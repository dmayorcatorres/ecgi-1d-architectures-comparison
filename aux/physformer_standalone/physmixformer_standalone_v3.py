# -*- coding: utf-8 -*-
"""
PhysMixFormer1D — standalone v2 (corregido)

Corrige physformer_standalone.py (v1, puerto fiel de physformer1.0.ipynb)
para que el código haga lo que la tesis y el artículo dicen que hace.
Ver /areas/physmixformer-code.md del historial de análisis para el detalle
completo de cada discrepancia encontrada en v1.

CAMBIOS RESPECTO A v1 (por orden de dependencia, no de aparición en el
archivo):

  0) DIRECCIÓN FIJA torso→corazón (Ec. 5 del artículo: f_θ: R^{Et×T} →
     R^{Ee×T}). El notebook original entrenaba corazón→torso (problema
     DIRECTO, bien planteado) por defecto; v1 "arreglaba" esto con un
     flag `direction` que transponía el kernel RBF de PhysicsMixer para
     usarlo en sentido inverso — el propio v1 documentaba que ese kernel
     transpuesto "NO es una solución regularizada del problema inverso".
     Aquí no hay flag de dirección: el modelo SIEMPRE mapea torso→corazón,
     y el operador físico Π SIEMPRE mapea corazón→torso (nunca al revés).

  1) GAUGE (Ec. 3-4). P_E^(ω) = I - (1 1^T Ω)/(1^T Ω 1), aplicado a la
     entrada (torso), al objetivo (corazón) y a la predicción final.
     Ausente por completo en v1.

  2) LAPLACIANO CON PESOS GAUSSIANOS (Ec. 7-8). a_ij = exp(-d(i,j)²/σ²)
     sobre k-NN, con distancia geodésica (Dijkstra sobre aristas de la
     malla) cuando hay caras, Euclídea si no. v1 usaba adyacencia BINARIA
     desde caras/kNN (sin pesos ni distancia).

  3) Π FILA-ESTOCÁSTICA SIEMPRE epicardio→torso (Ec. 9-10), parametrizada
     por softmax (garantiza Π≥0, Π·1=1 por construcción) anclada a un
     prior geométrico con temperatura recocida. v1 usaba un kernel RBF+
     top-k que, para el problema inverso, requería transponerse — ver (0).

  4) RAMA FÍSICA POR DIFUSIÓN CHEBYSHEV (Ec. 19-24): A_eff = Σ γ_j
     T_j(L_e)·Πᵀ, X_phys = A_eff · Y. Reemplaza el PhysicsMixer RBF de
     v1 (que no tenía relación con la formulación teórica del artículo).

  5) PÉRDIDA COMPUESTA DE 5 TÉRMINOS (Ec. 26-31): time (Ω_e-ponderado),
     spectral (FFT banda 0.5-100 Hz), corr (1-PCC), geom (forward-
     consistency ‖Πx̂-y‖², el término físico central), smooth (forma
     cuadrática x^T L_e x correcta — v1 usaba ‖Lx‖² y la dejaba inactiva
     con lambda_lap=0.0).

  6) SE ELIMINA `y_cls`/`use_label_concat`: el modelo es f_θ(Y) sin
     información de clase/protocolo concatenada a la entrada (v1 la
     concatenaba siempre, use_label_concat=True fijo).

  7) SPLIT: si el .mat trae un id de sujeto (`subject_key` en DATASETS)
     se hace split por sujeto (grupo completo a train o val). Si NO hay
     id de sujeto disponible se cae a estratificado por etiqueta con una
     ADVERTENCIA EXPLÍCITA de que esto puede permitir fuga de sujeto —
     v1 hacía esto último siempre y sin advertencia.

  8) Z-SCORE SIN FUGA: normalización de ENTRADA únicamente (torso), y
     solo con estadísticas del split de entrenamiento — nunca del
     dataset completo. El objetivo epicárdico y las señales que entran
     al bloque físico (Π, geom-loss) se mantienen en mV reales para que
     la mezcla convexa de Π y el forward-consistency tengan sentido
     físico. v1 calculaba mu/sigma sobre TODO el dataset antes del split
     (fuga) y normalizaba también el objetivo.

  9) FILTRO ESTÁNDAR: notch 50 Hz Y 60 Hz, low-pass único por dataset
     (según su fs), NO por etiqueta de clase. v1 usaba solo notch 50 Hz
     y un low-pass que dependía de `tag` (LP_CUTS indexado por etiqueta),
     lo que además usaba información de clase durante el preprocesado.

  10) MATRIZ DE ABLACIÓN: `use_multiscale` (Transformer1D vs MixFormer1D)
      y `fuse_physics` + `beta_geom`/`beta_smooth` (physics en la salida
      vs. solo en la pérdida vs. ambos vs. ninguno) permiten reproducir
      la Tabla 7 del artículo Y la matriz de 5 configuraciones que separa
      "physics-loss" de "physics-branch" (evita el problema señalado en
      la revisión: que los "baselines data-driven" también reciban
      términos físicos en Eq. 26 sin declararlo).

LO QUE **NO** SE INTENTA AQUÍ (a propósito, ver razonamiento previo):
  - PINN completo / residuo de PDE explícito.
  - "Convergence guarantees" para el operador aprendido A_eff.
  - Embeber en src/run_bohb.py (siguiente paso, fuera de este archivo).
  - Split subject-wise real: requiere que el usuario confirme el campo
    de id de sujeto en cada .mat (no verificado aquí porque los .mat no
    están disponibles en este entorno). Se deja el hook `subject_key` y
    la advertencia explícita si no se usa.
"""
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Union, Any, Callable
import argparse
import math
import os
import time
import warnings

import numpy as np
import scipy.io as sio
import scipy.signal as sps
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

# =======================
# Config básica
# =======================
SEED   = 121
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
torch.backends.cudnn.benchmark = True
MAX_WORKERS = min(8, (os.cpu_count() or 2))

REPO_ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path(".")
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"

# fs y low-pass POR DATASET (no por etiqueta — ver cambio #9). El low-pass
# se fija a fs/4 acotado a [150, 250] Hz salvo que se pase explícito;
# ajustar aquí si el criterio real difiere.
DATASETS = {
    "ckitpvc": {
        "path": REPO_ROOT / "src/data/ckitpvc/raw/ckitpvc.mat",
        "fs": 1000, "lp_cut": 250.0,
        "keys": {"y_torso": "ecg", "x_epi": "emg", "tag": "tag"},
        "subject_key": None,  # TODO: confirmar si el .mat trae id de sujeto
    },
    "bordeaux": {
        "path": REPO_ROOT / "src/data/bordeaux/raw/bordeaux.mat",
        "fs": 2048, "lp_cut": 500.0,
        "keys": {"y_torso": "ecg", "x_epi": "emg", "tag": "tag"},
        "subject_key": None,  # TODO: Bordeaux tiene 4 sujetos según el artículo
    },
    "auckland": {
        "path": REPO_ROOT / "src/data/auckland/raw/auckland.mat",
        "fs": 2048, "lp_cut": 500.0,
        "keys": {"y_torso": "ecg", "x_epi": "emg", "tag": "tag"},
        "subject_key": None,
    },
    "utah2002": {
        "path": REPO_ROOT / "src/data/utah2002/raw/utah2002.mat",
        "fs": 1000, "lp_cut": 250.0,
        "keys": {"y_torso": "ecg", "x_epi": "emg", "tag": "tag"},
        "subject_key": None,
    },
}

# =======================
# Utilidades generales
# =======================
def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def loadmat_smart(path: Union[str, Path]) -> Dict[str, Any]:
    path = str(path)
    try:
        return sio.loadmat(path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    except TypeError:
        return sio.loadmat(path, squeeze_me=True, struct_as_record=False)

def list_mat_keys(path: Path):
    mat = loadmat_smart(path)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    info = []
    for k in keys:
        v = mat[k]
        shape = getattr(v, "shape", None)
        info.append((k, shape, type(v).__name__))
    return info

# =======================
# 1) FILTRO — notch 50+60 Hz, low-pass ÚNICO por dataset (cambio #9)
# =======================
class ButterFilter:
    """HP 0.5 Hz (4º) -> notch 50 Hz -> notch 60 Hz -> LP fijo (6º), SOS."""
    def __init__(self, fs: float, lp_cut: float, hp_cut: float = 0.5,
                 notches: Tuple[float, ...] = (50.0, 60.0), notch_Q: float = 35.0):
        self.fs = fs
        self.hp_sos = sps.butter(4, hp_cut, btype="high", fs=fs, output="sos")
        self.lp_sos = sps.butter(6, min(lp_cut, fs / 2.0 * 0.98), btype="low", fs=fs, output="sos")
        self.notch_sos = []
        for f0 in notches:
            if f0 >= fs / 2.0:
                continue  # por encima de Nyquist para este fs, se omite
            try:
                b_no, a_no = sps.iirnotch(f0, notch_Q, fs=fs)
            except TypeError:
                w0 = f0 / (fs / 2.0)
                b_no, a_no = sps.iirnotch(w0, notch_Q)
            self.notch_sos.append(sps.tf2sos(b_no, a_no))

    def __call__(self, tr: torch.Tensor) -> torch.Tensor:
        """tr: (C,T). Ya NO depende de `tag` — se aplica igual a todo el dataset."""
        x = tr.detach().cpu().numpy()
        x = sps.sosfiltfilt(self.hp_sos, x, axis=-1)
        for sos in self.notch_sos:
            x = sps.sosfiltfilt(sos, x, axis=-1)
        x = sps.sosfiltfilt(self.lp_sos, x, axis=-1)
        x = np.ascontiguousarray(x).copy()
        return torch.from_numpy(x).to(tr.device, dtype=tr.dtype)


# =======================
# 2) Z-SCORE (solo para la entrada; ver cambio #8)
# =======================
class ZScore:
    def __init__(self, mu, sigma, eps: float = 1e-8):
        self.mu = mu
        self.sigma = sigma
        self.eps = eps
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mu = self.mu; sigma = self.sigma
        if not torch.is_tensor(mu):    mu = torch.tensor(mu, dtype=x.dtype, device=x.device)
        if not torch.is_tensor(sigma): sigma = torch.tensor(sigma, dtype=x.dtype, device=x.device)
        if mu.ndim == 1:    mu = mu.view(-1, 1)
        if sigma.ndim == 1: sigma = sigma.view(-1, 1)
        mu = mu.to(dtype=x.dtype, device=x.device)
        sigma = sigma.to(dtype=x.dtype, device=x.device)
        return (x - mu) / (sigma + self.eps)

# ---------- helpers robustos para structs MATLAB (sin cambios vs v1) ----------
def _unwrap_mat_struct(obj):
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, np.void):
        d = {}
        for name in (obj.dtype.names or []):
            d[name] = obj[name]
        return d
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        if obj.shape == ():
            return _unwrap_mat_struct(obj.item())
        if obj.size == 1:
            return _unwrap_mat_struct(obj.reshape(()).item())
    return obj

def _to_tensor_xy(obj, name="node"):
    arr = np.asarray(obj, dtype=np.float32 if name == "node" else np.int64)
    if arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] != 3:
        arr = arr.T
    return torch.from_numpy(arr)

def _find_nodes_faces(container: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    node_keys = ['node', 'nodes', 'Vertices', 'vertices', 'xyz', 'XYZ']
    face_keys = ['face', 'faces', 'tri', 'tris', 'Triangles', 'triangles', 'elements', 'Elements']

    def iter_dicts(d):
        if not isinstance(d, dict):
            return
        yield d
        for v in d.values():
            v = _unwrap_mat_struct(v)
            if isinstance(v, dict):
                yield from iter_dicts(v)
            elif isinstance(v, (list, tuple)):
                for el in v:
                    el = _unwrap_mat_struct(el)
                    if isinstance(el, dict):
                        yield from iter_dicts(el)

    d0 = _unwrap_mat_struct(container)
    for d in iter_dicts(d0 if isinstance(d0, dict) else {}):
        nk = next((k for k in node_keys if k in d), None)
        fk = next((k for k in face_keys if k in d), None)
        if nk and fk:
            return np.asarray(d[nk]), np.asarray(d[fk])
    return None

def _get_nodes_faces(struct_like) -> Tuple[torch.Tensor, torch.Tensor]:
    d = _unwrap_mat_struct(struct_like)
    nodes = faces = None
    if isinstance(d, dict) and ('node' in d and 'face' in d):
        nodes, faces = d['node'], d['face']
    else:
        found = _find_nodes_faces(d)
        if found is not None:
            nodes, faces = found
    if nodes is None or faces is None:
        keys = list(d.keys()) if isinstance(d, dict) else type(d).__name__
        raise ValueError(f"El struct de malla no contiene 'node' y 'face' (ni alias). Visto: {keys}")

    nodes_t = _to_tensor_xy(nodes, "node").float()
    faces_t = _to_tensor_xy(faces, "face").long()
    if faces_t.numel() > 0 and faces_t.min().item() >= 1:
        faces_t = faces_t - 1
    return nodes_t, faces_t

def _get_electrode_ids(struct_like, default_len: int) -> torch.Tensor:
    d = _unwrap_mat_struct(struct_like)
    candidates = []
    if isinstance(d, dict):
        for k in ['elec', 'electrodes', 'idx', 'indices']:
            if k in d:
                candidates.append(np.asarray(d[k]).squeeze())

    ids = candidates[0] if candidates else np.arange(1, default_len + 1)
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim == 0: ids = ids[None]
    if ids.size == 0: ids = np.arange(1, default_len + 1, dtype=np.int64)
    if ids.min(initial=1) >= 1:
        ids = ids - 1
    return torch.from_numpy(ids).long()

def submesh(nodes: torch.Tensor, faces: torch.Tensor, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    nodes_sel = nodes[idx]
    if faces.numel() == 0:
        return nodes_sel, faces.new_zeros((0, 3), dtype=torch.long)
    if hasattr(torch, "isin"):
        mask = torch.isin(faces, idx).all(dim=1)
    else:
        mask = (faces[..., None] == idx.view(1, 1, -1)).any(-1).all(-1)
    faces_sel_global = faces[mask]
    if faces_sel_global.numel() == 0:
        return nodes_sel, faces_sel_global
    remap = faces.new_full((nodes.shape[0],), -1)
    remap[idx] = torch.arange(idx.numel(), device=faces.device, dtype=torch.long)
    faces_sel = remap[faces_sel_global]
    return nodes_sel, faces_sel

# =======================
# 3) GEOMETRÍA: Laplaciano con pesos GAUSSIANOS + distancia geodésica
#    (cambio #2 — Ec. 7-8 del artículo)
# =======================
def _edges_from_faces(faces: torch.Tensor) -> torch.Tensor:
    f = faces.long()
    if f.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.long)
    return torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]], dim=0)

def geodesic_distance_matrix(coords: torch.Tensor, faces: torch.Tensor) -> Optional[torch.Tensor]:
    """Distancia geodésica aproximada: Dijkstra sobre las aristas de la
    malla, pesadas por longitud euclídea (Ec. 7: 'geodesic distance along
    the surface, computed via Dijkstra's algorithm')."""
    n = coords.shape[0]
    if faces.numel() == 0 or n == 0:
        return None
    edges = _edges_from_faces(faces).numpy()
    pts = coords.detach().cpu().numpy()
    w = np.linalg.norm(pts[edges[:, 0]] - pts[edges[:, 1]], axis=1)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    vals = np.concatenate([w, w])
    graph = csr_matrix((vals, (rows, cols)), shape=(n, n))
    D = dijkstra(graph, directed=False)
    finite = np.isfinite(D)
    if finite.any():
        D[~finite] = D[finite].max() * 2.0  # componentes desconectadas: penalización grande, no inf
    else:
        D[~finite] = 0.0
    return torch.from_numpy(D).float()

def gaussian_knn_laplacian(coords: torch.Tensor, faces: torch.Tensor,
                           sigma: float, k: int = 6,
                           prefer_geodesic: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Devuelve (L_normalizado, grado) siguiendo Ec. 7-8:
    a_ij = exp(-d(i,j)²/σ²) en la máscara k-NN, L = I - D^{-1/2} A D^{-1/2}.
    """
    n = coords.shape[0]
    if n == 0:
        z = torch.zeros((0, 0))
        return z, torch.zeros((0,))
    D = geodesic_distance_matrix(coords, faces) if prefer_geodesic else None
    if D is None:
        D = torch.cdist(coords.float(), coords.float(), p=2)
    D = D.clone()
    D.fill_diagonal_(float("inf"))
    k_eff = min(k, max(1, n - 1))
    knn_idx = torch.topk(-D, k=k_eff, dim=1).indices
    mask = torch.zeros((n, n), dtype=torch.bool)
    rows = torch.arange(n).unsqueeze(1).expand(-1, k_eff)
    mask[rows, knn_idx] = True
    mask = mask | mask.T
    D0 = D.clone()
    D0[torch.isinf(D0)] = 0.0
    A = torch.exp(-(D0 ** 2) / max(sigma ** 2, 1e-12)) * mask.float()
    A.fill_diagonal_(0.0)
    deg = A.sum(dim=1)
    d_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    Dis = torch.diag(d_inv_sqrt)
    L = torch.eye(n) - Dis @ A @ Dis
    return L, deg

def pairwise_dist_3d(P_a: torch.Tensor, P_b: torch.Tensor) -> torch.Tensor:
    return torch.cdist(P_a.float(), P_b.float(), p=2)

# =======================
# 4) SPLIT — subject-wise si hay id, si no estratificado CON ADVERTENCIA
#    (cambio #7)
# =======================
def _stratified_split_indices(labels: torch.Tensor, ratio: float, seed: int = SEED):
    rng = np.random.default_rng(seed)
    labels_np = labels.numpy()
    tr_idx, va_idx = [], []
    for lab in np.unique(labels_np):
        idx = np.where(labels_np == lab)[0]
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * ratio))) if len(idx) > 1 else len(idx)
        cut = min(cut, len(idx) - 1) if len(idx) > 1 else len(idx)
        tr_idx.extend(idx[:cut].tolist())
        va_idx.extend(idx[cut:].tolist())
    if len(va_idx) == 0 and len(tr_idx) > 1:  # asegurar val no vacío
        va_idx.append(tr_idx.pop())
    return tr_idx, va_idx

def _group_split_indices(groups: np.ndarray, ratio: float, seed: int):
    """Split por sujeto: cada sujeto completo va a train O a val (nunca a
    ambos), aproximando `ratio` en número de MUESTRAS (no de sujetos)."""
    rng = np.random.default_rng(seed)
    uniq_groups = np.unique(groups)
    rng.shuffle(uniq_groups)
    n_total = len(groups)
    target_train = int(round(n_total * ratio))
    tr_idx, va_idx = [], []
    n_train_so_far = 0
    for g in uniq_groups:
        idx = np.where(groups == g)[0].tolist()
        if n_train_so_far < target_train:
            tr_idx.extend(idx)
            n_train_so_far += len(idx)
        else:
            va_idx.extend(idx)
    if len(va_idx) == 0 and len(uniq_groups) > 1:
        # mover el último grupo agregado a val si el split se pasó de largo
        last_group_idx = np.where(groups == uniq_groups[-1])[0].tolist()
        tr_idx = [i for i in tr_idx if i not in last_group_idx]
        va_idx = last_group_idx
    return tr_idx, va_idx

def split_indices(labels: torch.Tensor, ratio: float, seed: int,
                  subject_ids: Optional[np.ndarray] = None, dataset_name: str = ""):
    if subject_ids is not None:
        return _group_split_indices(subject_ids, ratio, seed)
    warnings.warn(
        f"[{dataset_name}] Sin id de sujeto disponible: split ESTRATIFICADO "
        f"por etiqueta, no subject-wise. Esto puede permitir que segmentos "
        f"del mismo sujeto caigan en train Y val — declarar esta limitación "
        f"al reportar resultados (ver Sección 3.2 del artículo, que exige "
        f"split subject-wise). Confirmar el campo de sujeto en el .mat y "
        f"pasarlo via DATASETS[...]['subject_key'] para corregir esto.",
        stacklevel=2,
    )
    return _stratified_split_indices(labels, ratio, seed)

# =======================
# 5) DATASET — nombres torso/epicardio explícitos (cambio #0)
# =======================
class ECGDataset(Dataset):
    """
    y_torso: (N, Et, T)   BSPM — ENTRADA del modelo (Ec. 1: Y)
    x_epi:   (N, Ee, T)   HSPM — OBJETIVO del modelo (Ec. 1: X)
    tag:     (N,)         etiqueta de clase/protocolo (solo para split/plots,
                          NUNCA se concatena a la entrada del modelo — cambio #6)
    """
    def __init__(self,
                 path: Union[str, Path],
                 key_y_torso: str = "ecg",
                 key_x_epi: str = "emg",
                 key_tag: str = "tag",
                 key_torso_mesh: str = "torso",
                 key_heart_mesh: str = "heart",
                 key_subject: Optional[str] = None,
                 tf_y_torso: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 tf_filt: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        self.path = Path(path)
        mat: Dict[str, Any] = loadmat_smart(self.path)

        y_torso = torch.as_tensor(mat[key_y_torso]).float()
        x_epi = torch.as_tensor(mat[key_x_epi]).float()
        tag = torch.as_tensor(mat[key_tag]).long().squeeze()

        def _ensure_nct(t: torch.Tensor):
            if t.ndim == 2:
                return t.unsqueeze(1)
            if t.shape[1] > t.shape[2]:
                return t.transpose(1, 2).contiguous()
            return t
        y_torso = _ensure_nct(y_torso)
        x_epi = _ensure_nct(x_epi)

        uniq = torch.unique(tag)
        remap = {int(u): i for i, u in enumerate(sorted(uniq.tolist()))}
        tag = torch.tensor([remap[int(v)] for v in tag], dtype=torch.long)

        if x_epi.size(-1) != y_torso.size(-1):
            x_epi = F.interpolate(x_epi, size=y_torso.size(-1), mode="linear", align_corners=False)

        self.y_torso = y_torso     # (N, Et, T)
        self.x_epi = x_epi         # (N, Ee, T)
        self.tag = tag             # (N,)

        self.subject_ids = None
        if key_subject is not None and key_subject in mat:
            self.subject_ids = np.asarray(mat[key_subject]).astype(np.int64).squeeze()
            if self.subject_ids.shape[0] != y_torso.shape[0]:
                warnings.warn(f"subject_key '{key_subject}' tiene longitud distinta a N; se ignora.")
                self.subject_ids = None

        self.tf_y_torso = tf_y_torso
        self.tf_filt = tf_filt

        torso_obj = mat.get(key_torso_mesh, None)
        heart_obj = mat.get(key_heart_mesh, None)
        if torso_obj is None or heart_obj is None:
            raise ValueError("Faltan structs de malla: 'torso' y/o 'heart'.")

        self.torso_nodes, self.torso_faces = _get_nodes_faces(torso_obj)
        self.heart_nodes, self.heart_faces = _get_nodes_faces(heart_obj)

        Et = self.y_torso.shape[1]
        Ee = self.x_epi.shape[1]
        self.torso_elec_ids = _get_electrode_ids(torso_obj, Et)
        self.heart_elec_ids = _get_electrode_ids(heart_obj, Ee)

        if isinstance(_unwrap_mat_struct(torso_obj), dict) and \
           not any(k in _unwrap_mat_struct(torso_obj) for k in ['elec', 'electrodes', 'idx', 'indices']):
            print("[aviso] 'torso' no trae mapping 'elec'. Asumo canales <-> primeros nodos (0..Et-1).")
        if isinstance(_unwrap_mat_struct(heart_obj), dict) and \
           not any(k in _unwrap_mat_struct(heart_obj) for k in ['elec', 'electrodes', 'idx', 'indices']):
            print("[aviso] 'heart' no trae mapping 'elec'. Asumo canales <-> primeros nodos (0..Ee-1).")

        self.has_reliable_geometry = (
            self.torso_nodes.shape[0] == Et and self.heart_nodes.shape[0] == Ee
        )
        if not self.has_reliable_geometry:
            print(
                f"[aviso] malla NO está a resolución de electrodo "
                f"(torso: {self.torso_nodes.shape[0]} nodos vs Et={Et}; "
                f"corazón: {self.heart_nodes.shape[0]} nodos vs Ee={Ee}) — "
                f"mapeo electrodo<->nodo no verificado. La rama física "
                f"(gauge, Π, L_e) se desactiva por defecto para este dataset."
            )

        if len(self.torso_elec_ids) != Et:
            self.torso_elec_ids = self.torso_elec_ids[:Et]
        if len(self.heart_elec_ids) != Ee:
            self.heart_elec_ids = self.heart_elec_ids[:Ee]

    def __len__(self) -> int:
        return self.y_torso.size(0)

    def __getitem__(self, idx: int):
        y_torso = self.y_torso[idx]   # (Et,T) mV real
        tag = int(self.tag[idx])
        x_epi = self.x_epi[idx]       # (Ee,T) mV real, SIN normalizar (cambio #8)

        if self.tf_filt is not None:
            y_torso = self.tf_filt(y_torso)
            x_epi = self.tf_filt(x_epi)

        y_torso_in = self.tf_y_torso(y_torso) if self.tf_y_torso is not None else y_torso

        # Se devuelven ambas versiones de y_torso: la normalizada (entrada
        # de red) y la real en mV (para Π/geom-loss).
        return y_torso_in, y_torso, x_epi, tag

def build_geometry_pack(ds: "ECGDataset", sigma_t: float = 30.0, sigma_e: float = 30.0,
                        knn_k: int = 6) -> Dict[str, torch.Tensor]:
    P_t, F_t = submesh(ds.torso_nodes, ds.torso_faces, ds.torso_elec_ids)
    P_h, F_h = submesh(ds.heart_nodes, ds.heart_faces, ds.heart_elec_ids)

    L_t, deg_t = gaussian_knn_laplacian(P_t, F_t, sigma=sigma_t, k=knn_k)
    L_e, deg_e = gaussian_knn_laplacian(P_h, F_h, sigma=sigma_e, k=knn_k)

    D_te = pairwise_dist_3d(P_t, P_h)   # (Et, Ee) — distancias torso-epicardio

    return dict(
        coords_torso=P_t, faces_torso=F_t,
        coords_heart=P_h, faces_heart=F_h,
        L_torso=L_t, deg_torso=deg_t,
        L_heart=L_e, deg_heart=deg_e,
        D_te=D_te,
    )

# =======================
# 6) GAUGE — P_E^(w) (cambio #1 — Ec. 3-4)
# =======================
def gauge_projector_matrix(weights: torch.Tensor) -> torch.Tensor:
    """P_E^(w) = I - (1 1^T Ω) / (1^T Ω 1)   (Ec. 3). weights: (E,) > 0."""
    E = weights.shape[0]
    w = weights.view(-1).clamp_min(1e-12)
    ones = torch.ones(E, dtype=w.dtype, device=w.device)
    denom = (ones * w).sum().clamp_min(1e-12)
    return torch.eye(E, dtype=w.dtype, device=w.device) - torch.outer(ones, w) / denom

def apply_gauge(x: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """x: (B,E,T) -> P·x sobre el eje de electrodos."""
    return torch.einsum('ij,bjt->bit', P, x)

def gauge_weights_from_degree(deg: torch.Tensor) -> torch.Tensor:
    """Ω = D (grado), alineación espectral con el Laplaciano (Ec. 11).
    Si el grado es todo-cero (malla no confiable), cae a pesos uniformes."""
    if deg.numel() == 0 or deg.sum() <= 0:
        return torch.ones_like(deg) if deg.numel() > 0 else deg
    return deg.clamp_min(1e-8)

# =======================
# 7) Π FILA-ESTOCÁSTICA epicardio→torso, SIEMPRE (cambio #3 — Ec. 9-10)
# =======================
class RowStochasticCoupling(nn.Module):
    """Π ∈ R^{Et×Ee}: Π≥0, Π·1_Ee=1_Et por construcción (softmax por fila).
    SIEMPRE mapea epicardio→torso: y_torso ≈ Π · x_epi. Nunca se transpone
    para pretender resolver el problema inverso (ver cambio #0)."""
    def __init__(self, D_te: torch.Tensor, p: float = 2.0,
                 tau0: float = 1.0, tau_end: float = 0.3):
        super().__init__()
        self.register_buffer("D_te", D_te)            # (Et, Ee)
        self.p = p
        self.tau0 = tau0
        self.tau_end = tau_end
        self.tau = tau0
        self.logits = nn.Parameter(torch.zeros_like(D_te))  # residuo aprendible

    def anneal_tau(self, epoch: int, total_epochs: int):
        frac = min(max(epoch / max(1, total_epochs), 0.0), 1.0)
        self.tau = self.tau0 * (self.tau_end / self.tau0) ** frac

    def forward(self) -> torch.Tensor:
        prior_logits = -(self.D_te.clamp_min(0).pow(self.p)) / max(self.tau, 1e-6)
        logits = prior_logits + self.logits
        Pi = F.softmax(logits, dim=-1)  # (Et, Ee), fila-estocástica garantizada
        return Pi

# =======================
# 8) RAMA FÍSICA: difusión Chebyshev + Π (cambio #4 — Ec. 19-24)
# =======================
class PhysicsBranch(nn.Module):
    """A_eff = Σ_j γ_j · T_j(L_e)·Πᵀ  (Ec. 22-23);  X_phys = A_eff · Y (Ec. 24).

    Warm-up progresivo (Sección 2.3.3): los gate logits {β_j} que
    determinan {γ_j} vía softmax rampan LINEALMENTE desde un valor
    ~uniforme (β_j=warmup_uniform_logit, misma constante para las
    `n_bases` — softmax casi plano, todas las escalas de difusión
    contribuyen por igual) hasta los logits realmente APRENDIDOS
    (`gamma_logits`), durante las primeras `warmup_epochs` épocas. Sin
    esto, A_eff entra a la fusión residual desde la época 1 con una
    mezcla de bases sin calibrar — una de las causas identificadas de
    la regresión de RMSE de v2 frente a v1 (v1 sí tenía un ramp
    análogo, vía `bias_scale`, para su PhysicsMixer)."""
    def __init__(self, L_e: torch.Tensor, coupling: RowStochasticCoupling,
                 n_bases: int = 5, lmax: float = 2.0,
                 warmup_epochs: int = 8, warmup_uniform_logit: float = 0.1):
        super().__init__()
        Ee = L_e.shape[0]
        self.register_buffer("L_e", L_e)
        L_cheb = (2.0 / lmax) * L_e - torch.eye(Ee)  # Ec. 19: reescalado a [-1,1]
        self.register_buffer("L_cheb", L_cheb)
        self.coupling = coupling
        self.n_bases = max(1, n_bases)
        self.gamma_logits = nn.Parameter(torch.zeros(self.n_bases))
        self.warmup_epochs = max(1, warmup_epochs)
        self.warmup_uniform_logit = warmup_uniform_logit
        self.warmup_alpha = 1.0  # 1.0 = sin warm-up activo (100% gamma_logits aprendido);
                                  # se pisa cada época vía anneal_warmup() durante entrenamiento

    def anneal_warmup(self, epoch: int, total_epochs: Optional[int] = None):
        """Llamar una vez por época de ENTRENAMIENTO (igual que
        coupling.anneal_tau). `epoch` es 1-indexado. alpha=0 en la época 1
        (casi uniforme) -> alpha=1 al llegar a `warmup_epochs` (logits
        totalmente aprendidos), y se queda en 1.0 después."""
        w = self.warmup_epochs if total_epochs is None else min(self.warmup_epochs, max(1, total_epochs))
        self.warmup_alpha = min(max((epoch - 1) / max(1, w - 1) if w > 1 else 1.0, 0.0), 1.0)

    def _effective_gamma_logits(self) -> torch.Tensor:
        if self.warmup_alpha >= 1.0:
            return self.gamma_logits
        uniform = torch.full_like(self.gamma_logits, self.warmup_uniform_logit)
        return (1.0 - self.warmup_alpha) * uniform + self.warmup_alpha * self.gamma_logits

    def _chebyshev_bases(self, Pi_T: torch.Tensor) -> List[torch.Tensor]:
        # Pi_T: (Ee, Et).  T0=I -> T0@Pi_T=Pi_T ; T1=L_cheb ; recurrencia Ec. 20
        bases = [Pi_T]
        if self.n_bases > 1:
            Tk_prev, Tk = Pi_T, self.L_cheb @ Pi_T
            bases.append(Tk)
            for _ in range(2, self.n_bases):
                Tk_next = 2.0 * (self.L_cheb @ Tk) - Tk_prev
                bases.append(Tk_next)
                Tk_prev, Tk = Tk, Tk_next
        return bases

    def forward(self, y_torso_mV: torch.Tensor):
        """y_torso_mV: (B,Et,T) señal de torso en mV (gauge-proyectada, SIN
        z-score — ver cambio #8, la mezcla de Π solo es físicamente
        significativa en unidades reales compartidas)."""
        Pi = self.coupling()                 # (Et, Ee)
        Pi_T = Pi.transpose(0, 1)             # (Ee, Et)
        bases = self._chebyshev_bases(Pi_T)   # cada uno (Ee, Et)
        gamma = F.softmax(self._effective_gamma_logits(), dim=0)
        A_eff = sum(g * B for g, B in zip(gamma, bases))       # (Ee, Et)
        X_phys = torch.einsum('ec,bct->bet', A_eff, y_torso_mV)  # (B,Ee,T)
        return X_phys, Pi, A_eff

# =======================
# 9) BACKBONE — Transformer1D multiescala (sin y_cls, cambio #6)
#    Bloques reutilizados de v1: no tenían nada mal, solo se quita el
#    concat de etiqueta y el parámetro se vuelve puramente f_θ(Y).
# =======================
class PositionalEncodingSin(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0).transpose(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :, :x.size(-1)]

class RotaryPE(nn.Module):
    def __init__(self, head_dim: int, base_theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requiere head_dim par"
        self.head_dim = head_dim
        self.base_theta = base_theta
        self.register_buffer("_cached_t", torch.tensor(0), persistent=False)
        self.register_buffer("_cos", torch.empty(0), persistent=False)
        self.register_buffer("_sin", torch.empty(0), persistent=False)

    def _build_cache(self, T: int, device, dtype):
        half = self.head_dim // 2
        inv_freq = (self.base_theta ** (-torch.arange(0, half, dtype=torch.float32, device=device) / half))
        t = torch.arange(T, device=device, dtype=torch.float32)
        freqs = torch.einsum('t,f->tf', t, inv_freq)
        cos = torch.cos(freqs)[None, None, :, :]
        sin = torch.sin(freqs)[None, None, :, :]
        self._cached_t = torch.tensor(T, device=device)
        self._cos = cos.to(dtype=dtype)
        self._sin = sin.to(dtype=dtype)

    def apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        device, dtype = x.device, x.dtype
        if (self._cos.numel() == 0) or (self._cached_t.item() < T):
            self._build_cache(T, device, dtype)
        cos = self._cos[:, :, :T, :]
        sin = self._sin[:, :, :T, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.empty_like(x)
        x_rot[..., ::2] = x1 * cos - x2 * sin
        x_rot[..., 1::2] = x2 * cos + x1 * sin
        return x_rot

class MSWrapper(nn.Module):
    def __init__(self, module: nn.Module, scales=(1, 2, 4)):
        super().__init__()
        self.module = module
        self.scales = scales
        self.alpha = nn.Parameter(torch.zeros(len(scales)))

    def _down(self, x_btD: torch.Tensor, s: int) -> torch.Tensor:
        if s == 1: return x_btD
        return F.avg_pool1d(x_btD.transpose(1, 2), kernel_size=s, stride=s).transpose(1, 2)

    def _up(self, x_btD: torch.Tensor, T: int, s: int) -> torch.Tensor:
        if s == 1: return x_btD
        return F.interpolate(x_btD.transpose(1, 2), size=T, mode="linear", align_corners=False).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        outs = []
        for s in self.scales:
            xs = self._down(x, s)
            ys = self.module(xs)
            ys = self._up(ys, T, s)
            outs.append(ys)
        w = self.alpha.softmax(dim=0)
        return sum(w[i] * outs[i] for i in range(len(outs)))

class SelfAttention1D(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_drop: float = 0.1,
                 pe_type: str = "rope", rope_theta: float = 10000.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.pe_type = pe_type.lower()
        self.rope = RotaryPE(self.head_dim, base_theta=rope_theta) if self.pe_type == "rope" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.in_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def to_heads(t):
            return t.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = to_heads(q); k = to_heads(k); v = to_heads(v)
        if self.rope is not None:
            q = self.rope.apply_rope(q)
            k = self.rope.apply_rope(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        ctx = torch.matmul(attn, v)
        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return self.out_proj(ctx)

class InceptionMultiScale1D(nn.Module):
    """Kernels {7,15,31,41}/dilataciones {1,2,3,5} — motivados fisiológicamente
    (Sección 2.3.1) SOLO si T está a 1000 Hz; ver nota en Transformer1D."""
    def __init__(self, d_model: int, kernels=(7, 15, 31, 41), dilations=(1, 2, 3, 5),
                 use_maxpool: bool = True, norm: str = "gn", dropout: float = 0.1):
        super().__init__()
        assert len(kernels) == len(dilations)
        self.branches = nn.ModuleList()
        for k, d in zip(kernels, dilations):
            pad = (k // 2) * d
            self.branches.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=k, padding=pad, dilation=d, groups=d_model, bias=False),
                nn.Conv1d(d_model, d_model, kernel_size=1, bias=True),
                nn.GELU()
            ))
        self.pool_branch = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1, count_include_pad=False),
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=True), nn.GELU()
        )
        self.use_maxpool = use_maxpool
        if use_maxpool:
            self.max_branch = nn.Sequential(
                nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                nn.Conv1d(d_model, d_model, kernel_size=1, bias=True), nn.GELU()
            )
        n_branches = len(self.branches) + 1 + (1 if use_maxpool else 0)
        self.mix = nn.Conv1d(d_model * n_branches, d_model, kernel_size=1, bias=True)
        self.norm = nn.BatchNorm1d(d_model) if norm == "bn" else (nn.GroupNorm(1, d_model) if norm == "gn" else nn.Identity())
        self.dropout = nn.Dropout(dropout)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, max(1, d_model // 4), 1), nn.GELU(),
            nn.Conv1d(max(1, d_model // 4), d_model, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [b(x) for b in self.branches]
        outs.append(self.pool_branch(x))
        if self.use_maxpool:
            outs.append(self.max_branch(x))
        y = torch.cat(outs, dim=1)
        y = self.mix(y)
        y = self.norm(y)
        gate = self.se(y)
        y = y * gate
        return self.dropout(y)

class ConvSandwich1D(nn.Module):
    def __init__(self, d_model: int, kernel: int = 7, norm: str = "gn", dropout: float = 0.1):
        super().__init__()
        pad = kernel // 2
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=pad, groups=d_model, bias=False)
        self.pw = nn.Conv1d(d_model, d_model, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm1d(d_model) if norm == "bn" else (nn.GroupNorm(1, d_model) if norm == "gn" else nn.Identity())
        self.drop = nn.Dropout(dropout)

    def forward(self, x_bdT: torch.Tensor) -> torch.Tensor:
        y = self.dw(x_bdT)
        y = self.pw(y)
        y = self.act(y)
        y = self.norm(y)
        return self.drop(y)

class TransformerBlock1D(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1,
                 pe_type: str = "rope", rope_theta: float = 10000.0,
                 attn_ms_scales=(1, 2, 4), conv_kernel: int = 7, conv_norm: str = "gn"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        base_attn = SelfAttention1D(d_model, num_heads, attn_drop=dropout, pe_type=pe_type, rope_theta=rope_theta)
        self.attn = MSWrapper(base_attn, scales=attn_ms_scales)
        self.conv_sandwich = ConvSandwich1D(d_model, kernel=conv_kernel, norm=conv_norm, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        xc = self.conv_sandwich(x.permute(0, 2, 1))
        x = x + xc.permute(0, 2, 1)
        x = x + self.mlp(self.ln2(x))
        return x

class Transformer1D(nn.Module):
    """f_θ(Y): (B,Et,T) -> (B,Ee,T). SIN concat de etiqueta (cambio #6).
    `use_ms_temporal=False` da el backbone "Transformer1D" puro (ablación);
    `True` da "MixFormer1D" (multiescala sin física)."""
    def __init__(self, time_steps: int, input_channels: int, output_channels: int,
                 d_model: int = 128, num_heads: int = 4, num_layers: int = 4,
                 dropout: float = 0.1, pe_type: str = "rope", rope_theta: float = 10000.0,
                 use_ms_temporal: bool = True, ms_type: str = "inception",
                 ms_kernels=(7, 15, 31, 41), ms_dilations=(1, 2, 3, 5),
                 attn_ms_scales=(1, 2, 4)):
        super().__init__()
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)
        self.pos_enc = PositionalEncodingSin(d_model, max_len=time_steps) if pe_type.lower() == "sin" else None
        self.use_ms_temporal = use_ms_temporal
        if use_ms_temporal:
            if ms_type == "inception":
                self.ms_temporal = InceptionMultiScale1D(d_model, kernels=ms_kernels, dilations=ms_dilations,
                                                         use_maxpool=True, norm="gn", dropout=dropout)
            else:
                raise ValueError(f"ms_type desconocido: {ms_type}")
        else:
            self.ms_temporal = None
        self.layers = nn.ModuleList([
            TransformerBlock1D(d_model, num_heads, dropout=dropout, pe_type=pe_type, rope_theta=rope_theta,
                               attn_ms_scales=attn_ms_scales, conv_kernel=7, conv_norm="gn")
            for _ in range(num_layers)
        ])
        self.out_ln = nn.LayerNorm(d_model)
        self.output_proj = nn.Conv1d(d_model, output_channels, kernel_size=1)

    def forward(self, y_torso: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(y_torso)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        if self.ms_temporal is not None:
            x = x + self.ms_temporal(x)
        x = x.permute(0, 2, 1)
        for blk in self.layers:
            x = blk(x)
        x = self.out_ln(x)
        x = x.permute(0, 2, 1).contiguous()
        return self.output_proj(x)

# =======================
# 10) MODELO COMPLETO — gauge + backbone + rama física + fusión residual
# =======================
class PhysMixFormer1D(nn.Module):
    """f_θ: torso (BSPM, mV) -> corazón (HSPM, mV), fijo (cambio #0).

    fuse_physics=True  -> X_phys se suma a la salida (Ec. 25, fusión residual)
    fuse_physics=False -> X_phys NO se suma; Π/A_eff igual se calculan y
                          quedan disponibles para L_geom en modo
                          "regularization-only" (Ec. 25, párrafo final)
    physics_branch=None -> sin rama física en absoluto (Transformer1D/
                            MixFormer1D puros de la Tabla 7)
    """
    def __init__(self, backbone: Transformer1D,
                 physics_branch: Optional[PhysicsBranch] = None,
                 P_torso: Optional[torch.Tensor] = None,
                 P_epi: Optional[torch.Tensor] = None,
                 fuse_physics: bool = True,
                 lambda_phys: float = 0.2,
                 lambda_phys_trainable: bool = False):
        super().__init__()
        self.backbone = backbone
        self.physics = physics_branch
        self.fuse_physics = fuse_physics and (physics_branch is not None)
        if lambda_phys_trainable:
            self.lambda_phys = nn.Parameter(torch.tensor(float(lambda_phys)))
        else:
            self.register_buffer("lambda_phys", torch.tensor(float(lambda_phys)))
        Et = P_torso.shape[0] if P_torso is not None else None
        Ee = P_epi.shape[0] if P_epi is not None else None
        self.register_buffer("P_torso", P_torso if P_torso is not None else torch.tensor([]))
        self.register_buffer("P_epi", P_epi if P_epi is not None else torch.tensor([]))

    def forward(self, y_torso_z: torch.Tensor, y_torso_mV: torch.Tensor):
        """y_torso_z: entrada normalizada para el backbone (estabilidad de
        entrenamiento). y_torso_mV: señal real en mV, usada por la rama
        física y por el gauge (cambio #8)."""
        y_in = y_torso_mV
        if self.P_torso.numel() > 0:
            y_in = apply_gauge(y_in, self.P_torso)  # Ec. 4, gauge de entrada

        x_base = self.backbone(y_torso_z)            # (B,Ee,T), predicción en mV

        Pi = A_eff = None
        if self.physics is not None:
            x_phys, Pi, A_eff = self.physics(y_in)
            if self.fuse_physics:
                lam = torch.clamp(self.lambda_phys, 0.0, 1.0)
                x_comb = x_base + lam * x_phys
            else:
                x_comb = x_base
        else:
            x_comb = x_base

        if self.P_epi.numel() > 0:
            x_hat = apply_gauge(x_comb, self.P_epi)   # Ec. 4, gauge de salida
        else:
            x_hat = x_comb
        return x_hat, Pi, A_eff

# =======================
# 11) PÉRDIDA COMPUESTA — 5 términos (cambio #5 — Ec. 26-31)
# =======================
class CompositeLossTerms(nn.Module):
    """Ec. 27-31, SIN combinar. Devuelve {nombre: tensor_escalar con grad}.
    Separado de la combinación (ver UncertaintyWeightedCombiner /
    FixedWeightCombiner más abajo) para poder comparar A/B ponderación fija
    vs. adaptativa sin duplicar el cálculo de cada término — diagnóstico de
    v2 vs v1: el término 'spec' tiene una escala ~15-20x mayor que 'time'
    en la práctica, así que con betas fijos domina el gradiente aunque su
    beta nominal (0.3) sea menor que el de 'time' (1.0)."""
    def __init__(self, fs: float = 250.0, band: Tuple[float, float] = (0.5, 100.0)):
        super().__init__()
        self.fs = fs
        self.band = band

    def time_term(self, x_hat, x, omega_e):
        w = omega_e.view(1, -1, 1)
        return ((x_hat - x) ** 2 * w).mean()

    def spectral_term(self, x_hat, x):
        # torch.fft.rfft no soporta bfloat16 (RuntimeError: "Unsupported
        # dtype BFloat16"); bajo autocast bf16 x_hat/x llegan en ese dtype.
        # Se fuerza float32 solo para el tramo de FFT (mismo patrón que el
        # fix de AutoCorrelationLayer en Autoformer1D, commit de2b1d7d).
        Xh = torch.fft.rfft(x_hat.float(), dim=-1)
        Xt = torch.fft.rfft(x.float(), dim=-1)
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / self.fs).to(x.device)
        band_mask = (freqs >= self.band[0]) & (freqs <= self.band[1])
        if band_mask.sum() == 0:
            return x.new_tensor(0.0)
        diff = (Xh - Xt)[..., band_mask]
        return (diff.real ** 2 + diff.imag ** 2).mean()

    def corr_term(self, x_hat, x, eps: float = 1e-8):
        xh = x_hat - x_hat.mean(-1, keepdim=True)
        xt = x - x.mean(-1, keepdim=True)
        num = (xh * xt).sum(-1)
        den = torch.sqrt((xh ** 2).sum(-1) * (xt ** 2).sum(-1) + eps)
        rho = num / (den + eps)
        return (1.0 - rho).mean()

    def geom_term(self, x_hat, y_torso, Pi, omega_t):
        # Pi: (Et,Ee); x_hat: (B,Ee,T) -> y_pred: (B,Et,T)
        y_pred = torch.einsum('oe,bet->bot', Pi, x_hat)
        w = omega_t.view(1, -1, 1)
        return ((y_pred - y_torso) ** 2 * w).mean()

    def smooth_term(self, x_hat, L_e):
        # x^T L_e x  por muestra y tiempo (Ec. 31), NO ||Lx||^2
        xh = x_hat.permute(0, 2, 1)              # (B,T,Ee)
        Lx = torch.einsum('ij,btj->bti', L_e, xh)  # (B,T,Ee)
        val = (xh * Lx).sum(-1)                   # (B,T)
        return val.mean()

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor,
               y_torso: Optional[torch.Tensor] = None, Pi: Optional[torch.Tensor] = None,
               L_e: Optional[torch.Tensor] = None,
               omega_e: Optional[torch.Tensor] = None,
               omega_t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if omega_e is None:
            omega_e = torch.ones(x.shape[1], device=x.device, dtype=x.dtype)
        terms: Dict[str, torch.Tensor] = {
            "time": self.time_term(x_hat, x, omega_e),
            "spec": self.spectral_term(x_hat, x),
            "corr": self.corr_term(x_hat, x),
        }
        if (Pi is not None) and (y_torso is not None):
            if omega_t is None:
                omega_t = torch.ones(y_torso.shape[1], device=x.device, dtype=x.dtype)
            terms["geom"] = self.geom_term(x_hat, y_torso, Pi, omega_t)
        if L_e is not None:
            terms["smooth"] = self.smooth_term(x_hat, L_e)
        return terms


class UncertaintyWeightedCombiner(nn.Module):
    """L_total = Σ_k exp(-s_k)·L_k + s_k   (Kendall, Gal & Cipolla, CVPR 2018).
    s_k = log(σ_k²) es aprendible por término: reemplaza los β_k fijos.
    Un término con escala grande (p.ej. 'spec') empuja su propio s_k hacia
    arriba, bajando su peso efectivo exp(-s_k) automáticamente — sin grid
    search por dataset. El término +s_k evita que el optimizador anule un
    término subiendo su incertidumbre sin límite (también sube la pérdida)."""
    def __init__(self, task_names: List[str], init_log_var: float = 0.0,
                s_clamp: Tuple[float, float] = (-6.0, 6.0)):
        super().__init__()
        self.s_clamp = s_clamp
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(float(init_log_var))) for name in task_names
        })  # init=0.0 -> peso inicial=1 para todos (arranque neutro, sin salto respecto a betas=1)

    def forward(self, terms: Dict[str, torch.Tensor]):
        total = None
        log: Dict[str, Dict[str, float]] = {}
        for name, Lk in terms.items():
            if name not in self.log_vars:
                # término presente en este batch pero no registrado al construir
                # el combiner (p.ej. 'geom'/'smooth' solo aparecen si hay física)
                raise KeyError(f"'{name}' no está en UncertaintyWeightedCombiner.log_vars "
                               f"({list(self.log_vars.keys())}); pasar task_names completo al construirlo.")
            s_k = self.log_vars[name].clamp(*self.s_clamp)  # evita exp() runaway bajo AMP
            precision = torch.exp(-s_k)
            contrib = precision * Lk + s_k
            total = contrib if total is None else total + contrib
            log[name] = {"raw": Lk.item(), "weight": precision.item(), "s": s_k.item()}
        return total, log


class FixedWeightCombiner(nn.Module):
    """Los β_k fijos de siempre — se conserva para el A/B fixed-vs-adaptive."""
    def __init__(self, betas: Dict[str, float]):
        super().__init__()
        self.betas = betas

    def forward(self, terms: Dict[str, torch.Tensor]):
        total = None
        log: Dict[str, Dict[str, float]] = {}
        for name, Lk in terms.items():
            beta = self.betas.get(name, 1.0)
            contrib = beta * Lk
            total = contrib if total is None else total + contrib
            log[name] = {"raw": Lk.item(), "weight": beta, "s": None}
        return total, log

# =======================
# 12) build_loader — SIN fuga de z-score (cambio #8)
# =======================
def compute_channel_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=(0, 2))
    sd = x.std(dim=(0, 2)).clamp_min(1e-8)
    return mu, sd

def build_loader(path: Path, batch_size: int, device: str, split_ratio: float = 0.7,
                 seed: int = SEED, keys: Optional[dict] = None, apply_filter: bool = True,
                 sample_rate: Optional[float] = None, lp_cut: Optional[float] = None,
                 subject_key: Optional[str] = None, dataset_name: str = ""):
    keys = keys or {"y_torso": "ecg", "x_epi": "emg", "tag": "tag"}
    dataset = ECGDataset(path, key_y_torso=keys["y_torso"], key_x_epi=keys["x_epi"],
                         key_tag=keys["tag"], key_subject=subject_key)

    tr_idx, va_idx = split_indices(dataset.tag, split_ratio, seed,
                                   subject_ids=dataset.subject_ids, dataset_name=dataset_name)

    # Filtro (cambio #9): se aplica ANTES de calcular estadísticas de z-score,
    # igual para train y val (no depende del split).
    fs_eff = sample_rate if sample_rate is not None else 1000.0
    lp_eff = lp_cut if lp_cut is not None else fs_eff / 4.0
    dataset.tf_filt = ButterFilter(fs=fs_eff, lp_cut=lp_eff) if apply_filter else None

    # Estadísticas de z-score SOLO con las muestras de TRAIN (cambio #8),
    # y solo para la señal de entrada (torso) — el objetivo (x_epi) queda
    # en mV reales, sin normalizar.
    y_train_raw = dataset.y_torso[tr_idx]
    if dataset.tf_filt is not None:
        y_train_raw = torch.stack([dataset.tf_filt(y) for y in y_train_raw], dim=0)
    mu_y, sd_y = compute_channel_stats(y_train_raw)
    dataset.tf_y_torso = ZScore(mu_y, sd_y)

    train_ds, val_ds = Subset(dataset, tr_idx), Subset(dataset, va_idx)

    use_gpu = (device == "cuda")
    nw = MAX_WORKERS if use_gpu else 0
    common = dict(batch_size=batch_size, pin_memory=use_gpu, num_workers=nw)
    if use_gpu and nw > 0:
        common.update(dict(persistent_workers=True, prefetch_factor=4))

    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader, dataset

# =======================
# 13) Train / eval loops
# =======================
def rmse_metric(y_hat, y):
    with torch.no_grad():
        return torch.sqrt(torch.mean((y_hat - y) ** 2)).item()

def pcc_metric(y_hat, y, eps=1e-8):
    with torch.no_grad():
        yh = y_hat - y_hat.mean(dim=-1, keepdim=True)
        yt = y - y.mean(dim=-1, keepdim=True)
        num = (yh * yt).sum(dim=-1)
        den = torch.sqrt((yh ** 2).sum(dim=-1) * (yt ** 2).sum(dim=-1) + eps)
        return (num / (den + eps)).mean().item()

def train_one_epoch(model: PhysMixFormer1D, loader, optimizer, device,
                    terms_module: CompositeLossTerms,
                    combiner: Union["UncertaintyWeightedCombiner", "FixedWeightCombiner"],
                    L_e: Optional[torch.Tensor], omega_e: Optional[torch.Tensor],
                    omega_t: Optional[torch.Tensor], scaler=None, grad_clip: float = 1.0,
                    epoch: int = 0, total_epochs: int = 1,
                    uncertainty_optimizer: Optional[torch.optim.Optimizer] = None,
                    use_geom: bool = True, use_smooth: bool = True):
    """uncertainty_optimizer: optimizador SEPARADO (SGD) para los s_k del
    combiner, si loss_balance='uncertainty'. Necesario porque bajo AdamW
    (el mismo optimizer usado para el modelo) la normalización por segundo
    momento hace que todos los s_k se muevan a un ritmo casi idéntico
    independientemente de la magnitud del término (~lr por paso, sea cual
    sea el gradiente crudo) — eso anula justo la propiedad que hace útil
    la ponderación por incertidumbre (verificado empíricamente: con AdamW
    compartido, 'time' y 'spec' convergían a pesos casi iguales pese a una
    diferencia de escala 18x; con SGD dedicado se diferencian en <10 pasos)."""
    model.train()
    combiner.train()
    if model.physics is not None:
        model.physics.coupling.anneal_tau(epoch, total_epochs)  # recocido de temperatura (Sección 2.3.3)
        model.physics.anneal_warmup(epoch, total_epochs)        # ramp de gate logits γ (Sección 2.3.3)

    total_loss = 0.0
    use_amp = (device.type == "cuda")
    amp_dtype = AMP_DTYPE if use_amp else torch.float32

    for y_torso_z, y_torso_mV, x_epi, _tag in loader:
        y_torso_z = y_torso_z.to(device, non_blocking=True)
        y_torso_mV = y_torso_mV.to(device, non_blocking=True)
        x_epi = x_epi.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            x_hat, Pi, A_eff = model(y_torso_z, y_torso_mV)
            terms = terms_module(x_hat, x_epi, y_torso=y_torso_mV,
                                 Pi=(Pi if use_geom else None),
                                 L_e=(L_e if use_smooth else None),
                                 omega_e=omega_e, omega_t=omega_t)
        # Combinación SIEMPRE en float32, fuera del autocast: log_vars/exp(-s_k)
        # son escalares (costo nulo) y evitan overflow/underflow bajo bf16/fp16.
        terms_fp32 = {k: v.float() for k, v in terms.items()}
        loss, _ = combiner(terms_fp32)

        if uncertainty_optimizer is not None:
            uncertainty_optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if uncertainty_optimizer is not None:
            # SGD plano, SIN scaler/AMP (son 5 escalares en fp32, ver forward
            # del combiner) y SIN el grad_clip del modelo (escalas distintas).
            uncertainty_optimizer.step()

        total_loss += loss.item() * y_torso_z.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model: PhysMixFormer1D, loader, device,
            terms_module: CompositeLossTerms,
            combiner: Union["UncertaintyWeightedCombiner", "FixedWeightCombiner"],
            L_e: Optional[torch.Tensor], omega_e: Optional[torch.Tensor],
            omega_t: Optional[torch.Tensor],
            use_geom: bool = True, use_smooth: bool = True) -> Dict[str, float]:
    model.eval()
    combiner.eval()
    tot_loss = tot_rmse = tot_pcc = 0.0
    last_weight_log: Dict[str, Dict[str, float]] = {}
    for y_torso_z, y_torso_mV, x_epi, _tag in loader:
        y_torso_z = y_torso_z.to(device, non_blocking=True)
        y_torso_mV = y_torso_mV.to(device, non_blocking=True)
        x_epi = x_epi.to(device, non_blocking=True)

        x_hat, Pi, A_eff = model(y_torso_z, y_torso_mV)
        terms = terms_module(x_hat, x_epi, y_torso=y_torso_mV,
                             Pi=(Pi if use_geom else None),
                             L_e=(L_e if use_smooth else None),
                             omega_e=omega_e, omega_t=omega_t)
        terms_fp32 = {k: v.float() for k, v in terms.items()}
        loss, weight_log = combiner(terms_fp32)
        last_weight_log = weight_log

        tot_loss += loss.item() * y_torso_z.size(0)
        tot_rmse += rmse_metric(x_hat, x_epi) * y_torso_z.size(0)
        tot_pcc += pcc_metric(x_hat, x_epi) * y_torso_z.size(0)

    n = len(loader.dataset)
    return {"loss": tot_loss / n, "rmse": tot_rmse / n, "pcc": tot_pcc / n,
           "weights": last_weight_log}

# =======================
# 14) run_basic_training — matriz de ablación (cambio #10)
# =======================
def build_model_and_geometry(full_ds: ECGDataset, T: int, d_model: int, num_heads: int,
                             num_layers: int, dropout: float,
                             ablation: str = "physmixformer",
                             fuse_physics: bool = True,
                             lambda_phys: float = 0.2,
                             lambda_phys_trainable: bool = False,
                             n_cheb_bases: int = 5,
                             physics_warmup_epochs: int = 8,
                             use_physics_override: Optional[bool] = None):
    """ablation in {"transformer", "mixformer", "physmixformer"}."""
    Et = full_ds.y_torso.shape[1]
    Ee = full_ds.x_epi.shape[1]
    use_ms = ablation in ("mixformer", "physmixformer")

    backbone = Transformer1D(
        time_steps=T, input_channels=Et, output_channels=Ee,
        d_model=d_model, num_heads=num_heads, num_layers=num_layers, dropout=dropout,
        pe_type="rope", use_ms_temporal=use_ms, ms_type="inception",
    )

    geom = build_geometry_pack(full_ds)
    phys_ok = full_ds.has_reliable_geometry if use_physics_override is None else use_physics_override
    physics_branch = None
    P_torso = P_epi = None
    omega_e = omega_t = None

    if phys_ok:
        coupling = RowStochasticCoupling(geom["D_te"])
        physics_branch = PhysicsBranch(geom["L_heart"], coupling, n_bases=n_cheb_bases,
                                       warmup_epochs=physics_warmup_epochs)
        omega_t = gauge_weights_from_degree(geom["deg_torso"])
        omega_e = gauge_weights_from_degree(geom["deg_heart"])
        P_torso = gauge_projector_matrix(omega_t)
        P_epi = gauge_projector_matrix(omega_e)
    else:
        print(f"[aviso] geometría no confiable: gauge/Π/L_e desactivados para este dataset.")

    # La rama física (Π, A_eff) se ADJUNTA siempre que la geometría sea
    # confiable, sin importar el backbone (`ablation`) — así `fuse_physics`
    # y `beta_geom`/`beta_smooth` pueden variar independientemente del
    # backbone (transformer/mixformer) para reproducir la matriz de 5
    # configuraciones (neural-baseline, multiscale-baseline, physics-loss,
    # physics-branch, full-model) sin que el nombre de la ablación
    # determine implícitamente si hay física disponible para la pérdida.
    model = PhysMixFormer1D(
        backbone=backbone,
        physics_branch=physics_branch if phys_ok else None,
        P_torso=P_torso, P_epi=P_epi,
        fuse_physics=fuse_physics, lambda_phys=lambda_phys,
        lambda_phys_trainable=lambda_phys_trainable,
    )
    L_e = geom["L_heart"] if phys_ok else None
    return model, L_e, omega_e, omega_t

def run_basic_training(mat_path: Path, keys: dict, epochs: int = 20, batch_size: int = 8,
                       lr: float = 1e-3, d_model: int = 128, num_heads: int = 4, num_layers: int = 4,
                       dropout: float = 0.1, split_ratio: float = 0.7, seed: int = 125,
                       apply_filter: bool = True, dataset_name: str = "dataset",
                       output_dir: Optional[Path] = None, sample_rate: Optional[float] = None,
                       lp_cut: Optional[float] = None, subject_key: Optional[str] = None,
                       ablation: str = "physmixformer", fuse_physics: bool = True,
                       beta_geom: float = 1.0, beta_smooth: float = 1e-2,
                       lambda_phys: float = 0.2, lambda_phys_trainable: bool = False,
                       use_physics_override: Optional[bool] = None,
                       loss_balance: str = "uncertainty",
                       physics_warmup_epochs: int = 8):
    """
    ablation: "transformer" | "mixformer" | "physmixformer" — controla
    SOLO el backbone (sin/con multiescala). La rama física (Π, L_e) está
    disponible para la pérdida independientemente de `ablation`, así que
    la matriz de 5 configuraciones queda:
      1) neural-baseline:   ablation="transformer", fuse_physics=False, beta_geom=0, beta_smooth=0
      2) multiscale-baseline: ablation="mixformer",   fuse_physics=False, beta_geom=0, beta_smooth=0
      3) physics-loss:      ablation="mixformer",     fuse_physics=False, beta_geom>0, beta_smooth>0
      4) physics-branch:    ablation="physmixformer", fuse_physics=True,  beta_geom=0, beta_smooth=0
      5) full-model:        ablation="physmixformer", fuse_physics=True,  beta_geom>0, beta_smooth>0

    loss_balance: "uncertainty" (Kendall et al. 2018, recomendado — ver
    especificación Eje 1) o "fixed" (betas manuales, para el A/B contra lo
    anterior; usa beta_geom/beta_smooth de los argumentos y
    beta_time=1.0/beta_spec=0.3/beta_corr=0.3 fijos para time/spec/corr).
    """
    set_seed(seed)
    device = DEVICE

    train_loader, val_loader, full_ds = build_loader(
        path=mat_path, batch_size=batch_size, device=device.type,
        split_ratio=split_ratio, seed=seed, keys=keys, apply_filter=apply_filter,
        sample_rate=sample_rate, lp_cut=lp_cut, subject_key=subject_key, dataset_name=dataset_name,
    )
    T = full_ds.y_torso.shape[2]
    print(f"\nShapes: Et={full_ds.y_torso.shape[1]}, Ee={full_ds.x_epi.shape[1]}, T={T} "
         f"| device={device.type} | ablation={ablation} | fuse_physics={fuse_physics}")

    model, L_e, omega_e, omega_t = build_model_and_geometry(
        full_ds, T, d_model, num_heads, num_layers, dropout,
        ablation=ablation, fuse_physics=fuse_physics,
        lambda_phys=lambda_phys, lambda_phys_trainable=lambda_phys_trainable,
        use_physics_override=use_physics_override,
        physics_warmup_epochs=physics_warmup_epochs,
    )
    model = model.to(device)
    if L_e is not None: L_e = L_e.to(device)
    if omega_e is not None: omega_e = omega_e.to(device)
    if omega_t is not None: omega_t = omega_t.to(device)

    terms_module = CompositeLossTerms(fs=(sample_rate or 250.0)).to(device)
    # "¿existe el término?" (beta_geom/beta_smooth > 0, gatea el propio
    # cálculo del término en train_one_epoch/evaluate) es independiente de
    # "¿cómo se pondera?" (fixed vs uncertainty) — antes, en modo
    # 'uncertainty', beta_geom=0/beta_smooth=0 no tenía ningún efecto
    # porque task_names solo miraba si había geometría disponible, no si el
    # usuario había pedido apagar el término. Eso rompía la matriz de
    # ablación (configs 2 y 3 quedaban idénticas). Ya corregido.
    use_geom = (L_e is not None) and (beta_geom > 0.0)
    use_smooth = (L_e is not None) and (beta_smooth > 0.0)
    task_names = ["time", "spec", "corr"] + (["geom"] if use_geom else []) + (["smooth"] if use_smooth else [])
    if loss_balance == "uncertainty":
        combiner = UncertaintyWeightedCombiner(task_names).to(device)
    elif loss_balance == "fixed":
        betas = {"time": 1.0, "spec": 0.3, "corr": 0.3, "geom": beta_geom, "smooth": beta_smooth}
        combiner = FixedWeightCombiner(betas).to(device)
    else:
        raise ValueError(f"loss_balance desconocido: {loss_balance!r} (usa 'uncertainty' o 'fixed')")

    # Los s_k del combiner NO entran al AdamW del modelo (ver por qué en el
    # docstring de train_one_epoch): optimizador propio, SGD, para que la
    # ponderación por incertidumbre realmente diferencie por magnitud.
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if (p.ndim == 1 or n.endswith('bias')) else decay).append(p)
    opt = torch.optim.AdamW([
        {'params': decay, 'weight_decay': 1e-3},
        {'params': no_decay, 'weight_decay': 0.0},
    ], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=5e-5)

    uncertainty_opt = None
    if loss_balance == "uncertainty":
        uncertainty_opt = torch.optim.SGD(combiner.parameters(), lr=0.02)

    try:
        scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val = float("inf")
    best_metrics = None
    best_weights_log: Dict[str, Dict[str, float]] = {}
    out_dir = Path(output_dir) if output_dir is not None else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"physmixformer1d_{ablation}_seed{seed}.pt"

    t_start = time.time()
    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, device, terms_module, combiner,
                                  L_e, omega_e, omega_t, scaler=scaler, grad_clip=1.0,
                                  epoch=ep, total_epochs=epochs,
                                  uncertainty_optimizer=uncertainty_opt,
                                  use_geom=use_geom, use_smooth=use_smooth)
        val_metrics = evaluate(model, val_loader, device, terms_module, combiner, L_e, omega_e, omega_t,
                               use_geom=use_geom, use_smooth=use_smooth)
        scheduler.step(val_metrics["loss"])

        w = val_metrics.get("weights", {})
        w_str = " ".join(f"{k}:w={v['weight']:.3g}" for k, v in w.items()) if w else ""
        print(f"[{ep:03d}] train_loss={tr_loss:.4f} | val_loss={val_metrics['loss']:.4f} "
             f"rmse={val_metrics['rmse']:.4f} pcc={val_metrics['pcc']:.4f} "
             f"lr={opt.param_groups[0]['lr']:.3e} | {w_str}")

        if val_metrics["loss"] < best_val - 1e-6:
            best_val = val_metrics["loss"]
            best_metrics = {k: v for k, v in val_metrics.items() if k != "weights"}
            best_weights_log = w
            torch.save({"state_dict": model.state_dict(),
                       "combiner_state_dict": combiner.state_dict(),
                       "config": dict(ablation=ablation, fuse_physics=fuse_physics,
                                     beta_geom=beta_geom, beta_smooth=beta_smooth,
                                     lambda_phys=lambda_phys, loss_balance=loss_balance,
                                     physics_warmup_epochs=physics_warmup_epochs,
                                     lr=lr, seed=seed)},
                      best_path)

    elapsed = time.time() - t_start
    if best_weights_log:
        w_final = " ".join(f"{k}={v['weight']:.4g}" for k, v in best_weights_log.items())
        print(f"[pesos finales del mejor checkpoint] {w_final}")
    print(f"\n[OK] Mejor modelo guardado en: {best_path}  ({elapsed:.1f}s)")
    return {"lr": lr, "seed": seed, **(best_metrics or {}), "ckpt": str(best_path),
           "final_weights": best_weights_log}

# =======================
# 15) CLI
# =======================
def main():
    p = argparse.ArgumentParser(description="PhysMixFormer1D standalone v2 (corregido).")
    p.add_argument("--dataset", choices=list(DATASETS.keys()) + ["all"], default="all")
    p.add_argument("--ablation", choices=["transformer", "mixformer", "physmixformer"], default="physmixformer")
    p.add_argument("--no-fuse-physics", dest="fuse_physics", action="store_false", default=True,
                   help="Física solo en la pérdida (regularization-only, Ec. 25), no en la salida.")
    p.add_argument("--beta-geom", type=float, default=1.0)
    p.add_argument("--beta-smooth", type=float, default=1e-2)
    p.add_argument("--lambda-phys", type=float, default=0.2)
    p.add_argument("--lambda-phys-trainable", action="store_true", default=False)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--split-ratio", type=float, default=0.7)
    p.add_argument("--apply-filter", dest="apply_filter", action="store_true", default=True)
    p.add_argument("--no-apply-filter", dest="apply_filter", action="store_false")
    p.add_argument("--use-physics", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--loss-balance", choices=["uncertainty", "fixed"], default="uncertainty",
                   help="uncertainty (Kendall et al. 2018, recomendado): pesos s_k aprendidos "
                        "por término, ver especificación Eje 1. fixed: betas fijos (para el A/B).")
    p.add_argument("--physics-warmup-epochs", type=int, default=8,
                   help="Épocas de ramp de gate logits γ (Sección 2.3.3): de casi-uniforme a "
                        "aprendido. 1 = sin warm-up (γ aprendido desde la época 1).")
    args = p.parse_args()
    use_physics = {"auto": None, "on": True, "off": False}[args.use_physics]

    names = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for name in names:
        info = DATASETS[name]
        mat_path = Path(info["path"])
        if not mat_path.exists():
            print(f"[aviso] Salto '{name}': no encontrado en {mat_path}")
            continue
        print(f"\n{'=' * 80}\n>>> PhysMixFormer1D v2 — dataset={name}  ablation={args.ablation}\n{'=' * 80}")
        run_basic_training(
            mat_path=mat_path, keys=info["keys"], lr=args.lr, seed=args.seed,
            epochs=args.epochs, batch_size=args.batch_size, d_model=args.d_model,
            num_heads=args.num_heads, num_layers=args.num_layers, dropout=args.dropout,
            split_ratio=args.split_ratio, apply_filter=args.apply_filter,
            dataset_name=name, output_dir=OUTPUT_ROOT / name / args.ablation,
            sample_rate=info.get("fs"), lp_cut=info.get("lp_cut"),
            subject_key=info.get("subject_key"), ablation=args.ablation,
            fuse_physics=args.fuse_physics, beta_geom=args.beta_geom,
            beta_smooth=args.beta_smooth, lambda_phys=args.lambda_phys,
            lambda_phys_trainable=args.lambda_phys_trainable,
            use_physics_override=use_physics, loss_balance=args.loss_balance,
            physics_warmup_epochs=args.physics_warmup_epochs,
        )


if __name__ == "__main__":
    main()
