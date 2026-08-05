# -*- coding: utf-8 -*-
"""
Port standalone de physformer1.0.ipynb (celda de código principal).

Objetivo: reproducir EXACTAMENTE lo que hacía el notebook (backbone
Transformer1D + rama física PhysicsMixer + regularización geométrica +
restricción de conducción + preprocesado Butterworth/Z-score), fuera de
Colab y sin depender de nada en src/ (no importa run_bohb.py, ECGDataModule
ni MODEL_REGISTRY) — así queda auditable como réplica del notebook, aislada
del pipeline comparativo de las 37 arquitecturas.

Diferencias respecto al notebook, todas necesarias solo para correr fuera de
Colab (nada de lógica de modelo/entrenamiento cambia):
  - Se quita la celda 1 (mount de Google Drive): no aportaba nada real, la
    celda 2 ya redefine AMP_DTYPE/MAX_WORKERS por su cuenta.
  - DATASETS usa rutas locales del repo (src/data/<name>/raw/<name>.mat, los
    mismos .mat que ya usa src/run_bohb.py) en vez de /content/drive/...   Se
    agrega la entrada "utah2002" (mismas claves x=emg/y=tag/bspm=ecg que los
    otros 3; su .mat también trae los structs 'torso'/'heart').
  - run_basic_training() queda expuesto vía CLI (argparse) en vez de
    invocarse con valores fijos al final del notebook.
  - Los checkpoints se guardan en outputs/<dataset>/ en vez del cwd, y cada
    corrida agrega una fila a outputs/metrics_standalone.csv para poder
    comparar contra src/metrics_log.xlsx (ver compare_with_pipeline.py).
  - Se omite TransformerTimeEncoder: código muerto en el notebook (no lo usa
    run_basic_training, que instancia Transformer1D directamente).
"""
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Union, Any, Callable
import argparse
import csv
import math
import os
import time

import numpy as np
import scipy.io as sio
import scipy.signal as sps
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
torch.backends.cudnn.benchmark = True  # útil si T es fijo por batch
MAX_WORKERS = min(8, (os.cpu_count() or 2))

fs   = 2048  # Hz (ajusta si difiere)
LP_CUTS = {  # low-pass por etiqueta
    0: 200,
    1: 200,
    2: 200,
    3: 250,
}
alpha = 0.5

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"

DATASETS = {
    "ckitpvc": {
        "path": REPO_ROOT / "src/data/ckitpvc/raw/ckitpvc.mat",
        "fs": 1000,
        "keys": {"x": "emg", "y": "tag", "bspm": "ecg"}
    },
    "bordeaux": {
        "path": REPO_ROOT / "src/data/bordeaux/raw/bordeaux.mat",
        "fs": 2048,
        "keys": {"x": "emg", "y": "tag", "bspm": "ecg"}
    },
    "auckland": {
        "path": REPO_ROOT / "src/data/auckland/raw/auckland.mat",
        "fs": 2048,
        "keys": {"x": "emg", "y": "tag", "bspm": "ecg"}
    },
    "utah2002": {
        "path": REPO_ROOT / "src/data/utah2002/raw/utah2002.mat",
        "fs": 2048,
        "keys": {"x": "emg", "y": "tag", "bspm": "ecg"}
    },
}

# =======================
# Utilidades
# =======================
def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def loadmat_smart(path: Union[str, Path]) -> Dict[str, Any]:
    """Carga .mat soportando SciPy nuevos/antiguos."""
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
# 1)  FILTRO configurable (vectorizado + SOS)
# =======================
class ButterFilter:
    """HP 0.5 Hz (4º) -> notch 50 Hz -> LP por etiqueta (6º), vectorizado por canal con SOS."""
    def __init__(self, fs: float, lp_cuts: dict, hp_cut: float = 0.5,
                 notch: float = 50.0, notch_Q: float = 35.0):
        self.fs = fs
        self.lp_cuts = lp_cuts
        # HP y LP en SOS
        self.hp_sos = sps.butter(4, hp_cut, btype="high", fs=fs, output="sos")
        self.lp_cache = {tag: sps.butter(6, fc, btype="low", fs=fs, output="sos")
                         for tag, fc in lp_cuts.items()}
        # Notch: iirnotch -> (b,a) -> sos (compatible con todas las versiones)
        try:
            b_no, a_no = sps.iirnotch(notch, notch_Q, fs=fs)   # SciPy moderno
        except TypeError:
            w0 = notch / (fs / 2.0)                            # fallback: w0 normalizado
            b_no, a_no = sps.iirnotch(w0, notch_Q)
        self.no_sos = sps.tf2sos(b_no, a_no)

    def __call__(self, tr: torch.Tensor, tag: int) -> torch.Tensor:
        """tr: (C,T), tag: etiqueta entera."""
        lp_sos = self.lp_cache.get(tag, next(iter(self.lp_cache.values())))
        # A NumPy en CPU (sin copia aún)
        x = tr.detach().cpu().numpy()
        # Filtrado vectorizado por canal en el eje temporal (axis=-1)
        x = sps.sosfiltfilt(self.hp_sos, x, axis=-1)
        x = sps.sosfiltfilt(self.no_sos, x, axis=-1)
        x = sps.sosfiltfilt(lp_sos,   x, axis=-1)
        # Arreglo clave: garantizar contigüidad y strides positivos
        x = np.ascontiguousarray(x).copy()
        return torch.from_numpy(x).to(tr.device, dtype=tr.dtype)


# =======================
# 2)  Z-score (escalar o por canal)
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

# ---------- helpers robustos para structs MATLAB ----------
def _unwrap_mat_struct(obj):
    """
    Convierte formatos SciPy (np.void, ndarray dtype=object, dict) a dicts nativos.
    """
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
    """Convierte a tensor 2D; corrige orientación (N,3) y (F,3)."""
    arr = np.asarray(obj, dtype=np.float32 if name=="node" else np.int64)
    if arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] != 3:
        arr = arr.T
    return torch.from_numpy(arr)

def _find_nodes_faces(container: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    node_keys = ['node','nodes','Vertices','vertices','xyz','XYZ']
    face_keys = ['face','faces','tri','tris','Triangles','triangles','elements','Elements']

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
        faces_t = faces_t - 1  # a 0-based
    return nodes_t, faces_t

def _get_electrode_ids(struct_like, default_len: int) -> torch.Tensor:
    d = _unwrap_mat_struct(struct_like)
    candidates = []
    if isinstance(d, dict):
        for k in ['elec','electrodes','idx','indices']:
            if k in d:
                candidates.append(np.asarray(d[k]).squeeze())

    ids = candidates[0] if candidates else np.arange(1, default_len + 1)
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim == 0: ids = ids[None]
    if ids.size == 0: ids = np.arange(1, default_len + 1, dtype=np.int64)
    if ids.min(initial=1) >= 1:
        ids = ids - 1
    return torch.from_numpy(ids).long()

# ---------- submalla por electrodos (vectorizada) ----------
def submesh(nodes: torch.Tensor, faces: torch.Tensor, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    nodes: (V,3)   faces:(F,3)  0-based
    idx:   (C,)    índices a mantener
    """
    nodes_sel = nodes[idx]  # (C,3)
    if faces.numel() == 0:
        return nodes_sel, faces.new_zeros((0, 3), dtype=torch.long)

    if hasattr(torch, "isin"):
        mask = torch.isin(faces, idx).all(dim=1)   # (F,)
    else:
        # fallback
        mask = (faces[..., None] == idx.view(1, 1, -1)).any(-1).all(-1)
    faces_sel_global = faces[mask]
    if faces_sel_global.numel() == 0:
        return nodes_sel, faces_sel_global

    remap = faces.new_full((nodes.shape[0],), -1)
    remap[idx] = torch.arange(idx.numel(), device=faces.device, dtype=torch.long)
    faces_sel = remap[faces_sel_global]
    return nodes_sel, faces_sel

# ---------- adyacencia, kNN y laplaciano (sin bucles Python) ----------
def adjacency_from_faces(faces: torch.Tensor, n_nodes: Optional[int] = None) -> torch.Tensor:
    """
    Devuelve A (n x n) binaria y simétrica a partir de triángulos.
    """
    if faces.numel() == 0:
        return torch.zeros((0, 0), dtype=torch.float32)
    f = faces.long()
    if n_nodes is None:
        n_nodes = int(f.max().item()) + 1
    A = torch.zeros((n_nodes, n_nodes), dtype=torch.float32, device=f.device)
    i, j, k = f[:, 0], f[:, 1], f[:, 2]
    idx_row = torch.cat([i, i, j, j, k, k], dim=0)
    idx_col = torch.cat([j, k, i, k, i, j], dim=0)
    A[idx_row, idx_col] = 1.0
    A = torch.maximum(A, A.T)
    return A.cpu() if A.device.type != "cpu" else A

def knn_graph(coords: torch.Tensor, k: int = 6) -> torch.Tensor:
    """
    Construye adyacencia k-NN simétrica desde coords (N,3).
    """
    if coords.numel() == 0:
        return torch.zeros((0, 0), dtype=torch.float32, device=coords.device)
    with torch.no_grad():
        D = torch.cdist(coords.float(), coords.float(), p=2)  # (N,N)
        N = D.size(0)
        D[torch.arange(N, device=D.device), torch.arange(N, device=D.device)] = float("inf")
        knn_idx = torch.topk(-D, k=min(k, max(1, N-1)), dim=1).indices
        A = torch.zeros((N, N), dtype=torch.float32, device=D.device)
        rows = torch.arange(N, device=D.device).unsqueeze(1).expand_as(knn_idx)
        A[rows, knn_idx] = 1.0
        A = torch.maximum(A, A.T)
    return A.cpu() if A.device.type != "cpu" else A

def laplacian_from_adjacency(A: torch.Tensor, normalized: bool = True) -> torch.Tensor:
    deg = A.sum(dim=1)
    if normalized:
        d_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
        D_is = torch.diag(d_inv_sqrt)
        L = torch.eye(A.shape[0]) - D_is @ A @ D_is
    else:
        L = torch.diag(deg) - A
    return L

def laplacian_from_faces_or_knn(coords: torch.Tensor, faces: torch.Tensor,
                                normalized: bool = True, k: int = 6) -> torch.Tensor:
    if faces.numel() > 0:
        A = adjacency_from_faces(faces, n_nodes=coords.size(0))
    else:
        A = knn_graph(coords, k=k)
    return laplacian_from_adjacency(A, normalized=normalized)

# =======================
# 4)  SPLIT ESTRATIFICADO
# =======================
def _stratified_split_indices(labels: torch.Tensor, ratio: float, seed: int = SEED):
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    for i, lbl in enumerate(labels.tolist()):
        buckets[lbl].append(i)
    tr, va = [], []
    for idxs in buckets.values():
        idxs = np.array(idxs); rng.shuffle(idxs)
        k = int(len(idxs) * ratio)
        tr.extend(idxs[:k]); va.extend(idxs[k:])
    rng.shuffle(tr); rng.shuffle(va)
    return tr, va

# ---------- distancias y base RBF ----------
def pairwise_dist_3d(P_h: torch.Tensor, P_t: torch.Tensor) -> torch.Tensor:
    """
    Distancias Euclídeas entre nodos de corazón (M,3) y torso (N,3).
    Devuelve D: (M, N).
    """
    M = P_h.size(0)
    N = P_t.size(0)
    if P_h.numel() == 0 or P_t.numel() == 0:
        return torch.zeros((M, N), dtype=P_h.dtype, device=P_h.device)
    return torch.cdist(P_h.float(), P_t.float(), p=2)

def rbf_basis(D: torch.Tensor, sigmas: List[float]) -> torch.Tensor:
    """
    D: (M,N) -> (R,M,N) con R=len(sigmas) (broadcasting).
    """
    if D.numel() == 0:
        return torch.zeros((len(sigmas), D.size(0), D.size(1)))
    s = torch.tensor(sigmas, dtype=D.dtype, device=D.device).view(-1, 1, 1)  # (R,1,1)
    return torch.exp(- (D.unsqueeze(0) / s) ** 2)

# ---------- Dataset con geometría ----------
class ECGDataset(Dataset):
    """
    Señales:
      - x[key_x]   → (N, Cx, T)   (por defecto 'emg')
      - y_bspm     → (N, Ce, T)   (por defecto 'ecg')
      - y (clase)  → (N,)
    Geometría:
      - torso_nodes (Nt,3), torso_faces (Ft,3), ecg_ids (Ce,)
      - heart_nodes (Nh,3), heart_faces (Fh,3), emg_ids (Cx,)
    """
    def __init__(self,
                 path: Union[str, Path],
                 key_x: str = "emg",
                 key_y: str = "tag",
                 key_bspm: str = "ecg",
                 key_torso: str = "torso",
                 key_heart: str = "heart",
                 tf_x: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 tf_bspm: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 tf_filt: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None):
        self.path = Path(path)
        mat: Dict[str, Any] = loadmat_smart(self.path)

        # --- señales ---
        x = torch.as_tensor(mat[key_x]).float()               # (N,Cx,T) o (N,T,Cx)
        y = torch.as_tensor(mat[key_y]).long().squeeze()      # (N,)
        bspm = torch.as_tensor(mat[key_bspm]).float()         # (N,Ce,T) o (N,T,Ce) o (N,T)

        # Estandarizar ejes: (N,C,T)
        def _ensure_nct(t: torch.Tensor):
            if t.ndim == 2:           # (N,T) -> (N,1,T)
                return t.unsqueeze(1)
            if t.shape[1] > t.shape[2]:
                return t.transpose(1, 2).contiguous()
            return t
        x    = _ensure_nct(x)
        bspm = _ensure_nct(bspm)

        # remapeo de clases
        uniq = torch.unique(y)
        remap = {int(u): i for i, u in enumerate(sorted(uniq.tolist()))}
        y = torch.tensor([remap[int(v)] for v in y], dtype=torch.long)

        # ajustar longitudes temporales
        if bspm.size(-1) != x.size(-1):
            bspm = F.interpolate(bspm, size=x.size(-1), mode="linear", align_corners=False)

        self.x = x              # (N,Cx,T)
        self.y = y              # (N,)
        self.y_bspm = bspm      # (N,Ce,T)

        # transf (opcionales)
        self.tf_x    = tf_x
        self.tf_bspm = tf_bspm
        self.tf_filt = tf_filt

        # --- geometría ---
        torso_obj = mat.get(key_torso, None)
        heart_obj = mat.get(key_heart, None)
        if torso_obj is None or heart_obj is None:
            raise ValueError("Faltan structs de malla: 'torso' y/o 'heart'.")

        self.torso_nodes, self.torso_faces = _get_nodes_faces(torso_obj)  # (Nt,3), (Ft,3)
        self.heart_nodes, self.heart_faces = _get_nodes_faces(heart_obj)  # (Nh,3), (Fh,3)

        Ce = self.y_bspm.shape[1]
        Cx = self.x.shape[1]
        self.ecg_ids = _get_electrode_ids(torso_obj, Ce)  # (Ce,)
        self.emg_ids = _get_electrode_ids(heart_obj, Cx)  # (Cx,)

        if isinstance(_unwrap_mat_struct(torso_obj), dict) and \
           not any(k in _unwrap_mat_struct(torso_obj) for k in ['elec','electrodes','idx','indices']):
            print("[aviso] 'torso' no trae mapping 'elec'. Asumo canales ↔ primeros nodos (0..Ce-1).")
        if isinstance(_unwrap_mat_struct(heart_obj), dict) and \
           not any(k in _unwrap_mat_struct(heart_obj) for k in ['elec','electrodes','idx','indices']):
            print("[aviso] 'heart' no trae mapping 'elec'. Asumo canales ↔ primeros nodos (0..Cx-1).")

        if len(self.ecg_ids) != Ce:
            print(f"[aviso] ecg_ids({len(self.ecg_ids)}) != Ce({Ce}) → recorto.")
            self.ecg_ids = self.ecg_ids[:Ce]
        if len(self.emg_ids) != Cx:
            print(f"[aviso] emg_ids({len(self.emg_ids)}) != Cx({Cx}) → recorto.")
            self.emg_ids = self.emg_ids[:Cx]

    def __len__(self) -> int: return self.x.size(0)

    def __getitem__(self, idx: int):
        emg = self.x[idx]             # (Cx,T)
        tag = int(self.y[idx])        # ()
        ecg = self.y_bspm[idx]        # (Ce,T)

        if self.tf_x is not None:
            emg = self.tf_x(emg)
        if self.tf_filt is not None:
            emg = self.tf_filt(emg, tag)
        if self.tf_bspm is not None:
            ecg = self.tf_bspm(ecg)

        return emg, tag, ecg

# ---------- paquete geométrico listo para el modelo ----------
def build_geometry_pack(ds: "ECGDataset",
                        sigmas: List[float] = (20., 40., 80.),
                        knn_k: int = 6) -> Dict[str, torch.Tensor]:
    """
    Devuelve:
      - coords_torso_elec: (Ce,3), faces_torso_elec: (Ft_e,3)
      - coords_heart_elec: (Cx,3), faces_heart_elec: (Fh_e,3)
      - L_torso_elec: (Ce,Ce), L_heart_elec: (Cx,Cx)
      - D_ht: (Cx,Ce), B_rbf: (R,Cx,Ce)
    """
    P_t, F_t = submesh(ds.torso_nodes, ds.torso_faces, ds.ecg_ids)  # (Ce,3), (Ft_e,3)
    P_h, F_h = submesh(ds.heart_nodes, ds.heart_faces, ds.emg_ids)  # (Cx,3), (Fh_e,3)

    L_t = laplacian_from_faces_or_knn(P_t, F_t, normalized=True, k=knn_k)
    L_h = laplacian_from_faces_or_knn(P_h, F_h, normalized=True, k=knn_k)

    D = pairwise_dist_3d(P_h, P_t)             # (Cx,Ce)
    B = rbf_basis(D, list(sigmas))             # (R,Cx,Ce)

    return dict(
        coords_torso_elec=P_t, faces_torso_elec=F_t,
        coords_heart_elec=P_h, faces_heart_elec=F_h,
        L_torso_elec=L_t, L_heart_elec=L_h,
        D_ht=D, B_rbf=B
    )

def summarize_geometry(ds: "ECGDataset", geo: Dict[str, torch.Tensor]):
    Ce, Cx = ds.y_bspm.shape[1], ds.x.shape[1]
    print("\n🧩 Geometría:")
    print(f"  Torso:  nodes={tuple(ds.torso_nodes.shape)}  faces={tuple(ds.torso_faces.shape)}  ecg_ids={tuple(ds.ecg_ids.shape)} (Ce={Ce})")
    print(f"  Corazón: nodes={tuple(ds.heart_nodes.shape)}  faces={tuple(ds.heart_faces.shape)}  emg_ids={tuple(ds.emg_ids.shape)} (Cx={Cx})")
    print("  Submallas (por electrodos):")
    print(f"    coords_torso_elec: {tuple(geo['coords_torso_elec'].shape)}   faces_torso_elec: {tuple(geo['faces_torso_elec'].shape)}")
    print(f"    coords_heart_elec: {tuple(geo['coords_heart_elec'].shape)}   faces_heart_elec: {tuple(geo['faces_heart_elec'].shape)}")
    if geo['faces_torso_elec'].numel() == 0 or geo['faces_heart_elec'].numel() == 0:
        print("  ⚠️ Submalla sin caras detectada → Laplaciano por k-NN.")
    print("  Operadores:")
    print(f"    L_torso_elec: {tuple(geo['L_torso_elec'].shape)}   L_heart_elec: {tuple(geo['L_heart_elec'].shape)}")
    print(f"    D_ht: {tuple(geo['D_ht'].shape)}   B_rbf: {tuple(geo['B_rbf'].shape)}")

# =======================
# 5)  BUILD LOADER + RESUMEN (sin doble lectura)
# =======================
def compute_channel_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=(0, 2))
    sd = x.std(dim=(0, 2)).clamp_min(1e-8)
    return mu, sd

def build_loader(path: Path, batch_size: int, device: str,
                 split_ratio: float = 0.7, seed: int = SEED,
                 keys: Optional[dict] = None, apply_filter: bool = True):
    key_x = keys.get("x", "emg") if keys else "emg"
    key_y = keys.get("y", "tag") if keys else "tag"
    key_b = keys.get("bspm", "ecg") if keys else "ecg"

    # Cargar una sola vez
    dataset = ECGDataset(path, key_x=key_x, key_y=key_y, key_bspm=key_b)
    mu_x, sd_x = compute_channel_stats(dataset.x)
    mu_b, sd_b = compute_channel_stats(dataset.y_bspm)

    # Asignar transforms en la misma instancia
    dataset.tf_x = ZScore(mu_x, sd_x)
    dataset.tf_bspm = ZScore(mu_b, sd_b)
    dataset.tf_filt = ButterFilter(fs=fs, lp_cuts=LP_CUTS) if apply_filter else None

    tr_idx, va_idx = _stratified_split_indices(dataset.y, split_ratio, seed)
    train_ds, val_ds = Subset(dataset, tr_idx), Subset(dataset, va_idx)

    use_gpu = (device == "cuda")
    nw = MAX_WORKERS if use_gpu else 0
    common = dict(batch_size=batch_size, pin_memory=use_gpu, num_workers=nw)
    if use_gpu and nw > 0:
        common.update(dict(persistent_workers=True, prefetch_factor=4))

    train_loader = DataLoader(train_ds, shuffle=True,  **common)
    val_loader   = DataLoader(val_ds,   shuffle=False, **common)
    return train_loader, val_loader, dataset

def summarize_dataset(name: str, ds: ECGDataset):
    N = len(ds)
    Cx, Tx = ds.x.shape[1], ds.x.shape[2]
    Ce, Te = ds.y_bspm.shape[1], ds.y_bspm.shape[2]
    classes, counts = torch.unique(ds.y, return_counts=True)
    print(f"\n📦 Dataset: {name}")
    print(f"  Archivo: {ds.path.name}")
    print(f"  N={N}  |  X: (N={N}, Cx={Cx}, T={Tx})  |  Y_bspm: (N={N}, Ce={Ce}, T={Te})")
    print(f"  Clases (remap): {classes.tolist()}  -> counts: {counts.tolist()}")
    print(f"  Dtypes: X={ds.x.dtype}, Y_bspm={ds.y_bspm.dtype}, y={ds.y.dtype}")
    x_min, x_max = ds.x.min().item(), ds.x.max().item()
    y_min, y_max = ds.y_bspm.min().item(), ds.y_bspm.max().item()
    print(f"  Rangos: X[{x_min:.3f}, {x_max:.3f}]  |  Y_bspm[{y_min:.3f}, {y_max:.3f}]")

def peek_batch(loader: DataLoader, name: str):
    xb, yb, zb = next(iter(loader))  # xb: (B,Cx,T), yb: (B,), zb: (B,Ce,T)
    print(f"\n🔎 Primer batch ({name}):")
    print(f"  X_batch: {tuple(xb.shape)}  on {xb.device}  dtype={xb.dtype}")
    print(f"  y_batch: {tuple(yb.shape)}  on {yb.device}  dtype={yb.dtype}  (ej: {yb[:min(8, yb.numel())].tolist()})")
    print(f"  Y_batch: {tuple(zb.shape)}  on {zb.device}  dtype={zb.dtype}")

# ======= Energía local y restricción de conducción (optimizada) =======
def moving_rms(x: torch.Tensor, win: int = 17, eps: float = 1e-12) -> torch.Tensor:
    """
    x: (B,C,T) -> RMS local por canal usando avg_pool1d (cuDNN).
    """
    x2 = x * x
    try:
        avg = F.avg_pool1d(x2, kernel_size=win, stride=1, padding=win//2, count_include_pad=False)
    except TypeError:
        denom = torch.ones_like(x2)
        denom = F.avg_pool1d(denom, kernel_size=win, stride=1, padding=win//2)
        avg = F.avg_pool1d(x2, kernel_size=win, stride=1, padding=win//2) * denom.clamp_min(1e-6).reciprocal()
    return torch.sqrt(avg + eps)

def conduction_consistency(
    y_hat: torch.Tensor,   # (B,Cout,T)
    x: torch.Tensor,       # (B,Cin,T)
    A: torch.Tensor,       # (Cout,Cin)
    *, kappa: float = 1.0,
    win: int = 17,
    eps: float = 1e-12
) -> torch.Tensor:
    Ex = moving_rms(x,  win=win, eps=eps)            # (B,Cin,T)
    Ey = moving_rms(y_hat, win=win, eps=eps)         # (B,Cout,T)
    bound = torch.einsum('qc, bct -> bqt', A, Ex)    # (B,Cout,T)
    viol  = F.relu(Ey - kappa * bound)               # (B,Cout,T)
    return (viol * viol).mean()

# =======================
# 6)  MODELO (Transformer1D) con PE seleccionable
# =======================
class PositionalEncodingSin(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0).transpose(1, 2))     # (1,D,T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :, :x.size(-1)]

class RotaryPE(nn.Module):
    """RoPE para head_dim par."""
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

def _alibi_slopes(n_heads: int):
    def get_slopes(n):
        def power_of_two(n):  # mayor potencia de 2 <= n
            return 2 ** math.floor(math.log2(n))
        m = power_of_two(n)
        slopes = torch.pow(2, -torch.arange(1, m + 1, dtype=torch.float32))
        if m < n:
            extra = torch.pow(2, -torch.linspace(1, m, steps=n - m, dtype=torch.float32))
            slopes = torch.cat([slopes, extra], dim=0)
        return slopes
    return get_slopes(n_heads)

class AlibiBias(nn.Module):
    """Bias simétrico por |i-j| (no causal), por cabeza."""
    def __init__(self, n_heads: int):
        super().__init__()
        slopes = _alibi_slopes(n_heads)  # (H,)
        self.register_buffer("slopes", slopes.view(1, n_heads, 1, 1))
        self.register_buffer("_cached_T", torch.tensor(0), persistent=False)
        self.register_buffer("_bias", torch.empty(0), persistent=False)

    def _build_bias(self, T: int, device, dtype):
        t = torch.arange(T, device=device)
        dist = (t[None, :] - t[:, None]).abs().to(dtype)  # (T,T)
        bias = -dist[None, None, :, :] * self.slopes.to(device=device, dtype=dtype)  # (1,H,T,T)
        self._cached_T = torch.tensor(T, device=device)
        self._bias = bias

    def forward(self, T: int, device, dtype):
        if (self._bias.numel() == 0) or (self._cached_T.item() < T):
            self._build_bias(T, device, dtype)
        return self._bias[:, :, :T, :T]

class MSWrapper(nn.Module):
    """Piramide temporal multiescala para módulos (B,T,D)->(B,T,D)."""
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

class MultiScaleTemporal(nn.Module):
    """Bloque conv1d multiescala barato (depthwise+pointwise)."""
    def __init__(self, d_model: int, kernels=(5, 9, 21), dilations=(1, 2, 4), dropout: float = 0.1, depthwise: bool = True):
        super().__init__()
        assert len(kernels) == len(dilations)
        self.branches = nn.ModuleList()
        for k, d in zip(kernels, dilations):
            pad = (k // 2) * d
            if depthwise:
                branch = nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=k, padding=pad, dilation=d, groups=d_model, bias=False),
                    nn.Conv1d(d_model, d_model, kernel_size=1, bias=True),
                    nn.GELU()
                )
            else:
                branch = nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=k, padding=pad, dilation=d, bias=True),
                    nn.GELU()
                )
            self.branches.append(branch)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.zeros(len(kernels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [b(x) for b in self.branches]
        w = self.alpha.softmax(dim=0)
        y = sum(w[i] * outs[i] for i in range(len(outs)))
        return self.dropout(y)

class SelfAttention1D(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_drop: float = 0.1,
                 pe_type: str = "sin", rope_theta: float = 10000.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model debe ser múltiplo de num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.pe_type = pe_type.lower()
        self.rope = RotaryPE(self.head_dim, base_theta=rope_theta) if self.pe_type == "rope" else None
        self.alibi = AlibiBias(num_heads) if self.pe_type == "alibi" else None

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

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,T,T)
        if self.alibi is not None:
            bias = self.alibi(T, device=x.device, dtype=x.dtype)
            scores = scores + bias

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        ctx = torch.matmul(attn, v)                      # (B,H,T,head_dim)

        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        out = self.out_proj(ctx)
        return out


class InceptionMultiScale1D(nn.Module):
    """
    Variante Inception 1D multiescala:
      - ramas depthwise separables con kernels/dilataciones variados
      - ramas de avg-pool y (opcional) max-pool
      - concatenación + mezcla 1x1 para volver a d_model
      - SE-gating por muestra
      - normalización por canal (BN/GN/LN)
    Entrada/salida: (B, D, T)
    """
    def __init__(self,
                 d_model: int,
                 kernels=(7, 15, 31, 41),
                 dilations=(1, 2, 3, 5),
                 use_maxpool: bool = True,
                 norm: str = "gn",        # "bn" | "gn" | "ln"
                 dropout: float = 0.1):
        super().__init__()
        assert len(kernels) == len(dilations)
        self.branches = nn.ModuleList()
        for k, d in zip(kernels, dilations):
            pad = (k // 2) * d
            self.branches.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=k, padding=pad,
                          dilation=d, groups=d_model, bias=False),  # depthwise
                nn.Conv1d(d_model, d_model, kernel_size=1, bias=True),        # pointwise
                nn.GELU()
            ))

        # rama de avg-pool
        self.pool_branch = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1, count_include_pad=False),
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=True),
            nn.GELU()
        )

        self.use_maxpool = use_maxpool
        if use_maxpool:
            self.max_branch = nn.Sequential(
                nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                nn.Conv1d(d_model, d_model, kernel_size=1, bias=True),
                nn.GELU()
            )

        n_branches = len(self.branches) + 1 + (1 if use_maxpool else 0)
        self.mix = nn.Conv1d(d_model * n_branches, d_model, kernel_size=1, bias=True)

        if norm == "bn":
            self.norm = nn.BatchNorm1d(d_model)
        elif norm == "gn":
            self.norm = nn.GroupNorm(1, d_model)
        else:
            self.norm = nn.Identity()

        self.dropout = nn.Dropout(dropout)

        # Squeeze-Excitation (gating por canal)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, d_model // 4, 1), nn.GELU(),
            nn.Conv1d(d_model // 4, d_model, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, T)
        outs = [b(x) for b in self.branches]
        outs.append(self.pool_branch(x))
        if self.use_maxpool:
            outs.append(self.max_branch(x))

        y = torch.cat(outs, dim=1)   # (B, D*n_branches, T)
        y = self.mix(y)              # (B, D, T)
        y = self.norm(y)
        gate = self.se(y)            # (B, D, 1)
        y = y * gate
        return self.dropout(y)


class ConvSandwich1D(nn.Module):
    """
    Bloque conv ligero (depthwise + pointwise) para sesgo inductivo local.
    Trabaja en (B, D, T). Se recomienda GN con batch pequeño.
    """
    def __init__(self, d_model: int, kernel: int = 7, norm: str = "gn", dropout: float = 0.1):
        super().__init__()
        pad = kernel // 2
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=pad,
                            groups=d_model, bias=False)
        self.pw = nn.Conv1d(d_model, d_model, kernel_size=1, bias=True)
        self.act = nn.GELU()
        if norm == "bn":
            self.norm = nn.BatchNorm1d(d_model)
        elif norm == "gn":
            self.norm = nn.GroupNorm(1, d_model)
        else:
            self.norm = nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x_bdT: torch.Tensor) -> torch.Tensor:
        y = self.dw(x_bdT)
        y = self.pw(y)
        y = self.act(y)
        y = self.norm(y)
        return self.drop(y)


class TransformerBlock1D(nn.Module):
    """
    Bloque Transformer 1D:
      - Atención (SelfAttention1D) envuelta en pirámide multiescala (MSWrapper)
      - Convolución local (ConvSandwich1D)
      - MLP feed-forward
    I/O: (B, T, D)
    """
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 pe_type: str = "sin",
                 rope_theta: float = 10000.0,
                 attn_ms_scales=(1, 2, 4),
                 conv_kernel: int = 7,
                 conv_norm: str = "gn"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)

        base_attn = SelfAttention1D(
            d_model=d_model,
            num_heads=num_heads,
            attn_drop=dropout,
            pe_type=pe_type,
            rope_theta=rope_theta
        )
        self.attn = MSWrapper(base_attn, scales=attn_ms_scales)

        self.conv_sandwich = ConvSandwich1D(
            d_model=d_model,
            kernel=conv_kernel,
            norm=conv_norm,
            dropout=dropout
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.attn(self.ln1(x))                 # atención multiescala

        xc = self.conv_sandwich(x.permute(0, 2, 1))    # (B, D, T)
        x  = x + xc.permute(0, 2, 1)                   # residuo

        x = x + self.mlp(self.ln2(x))                  # MLP
        return x


class Transformer1D(nn.Module):
    """
    Entrada:  (B, C_in, T)
    Salida:   (B, C_out, T)
    Condicionamiento opcional: one-hot de etiqueta concatenado por canal.
    Multiescala seleccionable: "vanilla" (MultiScaleTemporal) o "inception" (InceptionMultiScale1D).
    """
    def __init__(self,
                 time_steps: int,
                 input_channels: int,
                 output_channels: int,
                 d_model: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 pe_type: str = "sin",
                 rope_theta: float = 10000.0,
                 use_ms_temporal: bool = True,
                 ms_type: str = "vanilla",          # "vanilla" | "inception"
                 ms_kernels=(3, 7, 15),
                 ms_dilations=(1, 2, 4),
                 attn_ms_scales=(1, 2, 4),
                 use_label_concat: bool = False,
                 num_classes: int | None = None):
        super().__init__()
        self.time_steps = time_steps
        self.pe_type = pe_type.lower()
        self.use_ms_temporal = use_ms_temporal
        self.ms_type = ms_type.lower()

        self.use_label_concat = bool(use_label_concat)
        self.num_classes = int(num_classes or 0) if self.use_label_concat else 0
        if self.use_label_concat and self.num_classes <= 0:
            raise ValueError("num_classes debe ser > 0 cuando use_label_concat=True")

        in_ch = input_channels + (self.num_classes if self.use_label_concat else 0)

        self.input_proj = nn.Conv1d(in_ch, d_model, kernel_size=1)
        self.pos_enc = PositionalEncodingSin(d_model, max_len=time_steps) if self.pe_type == "sin" else None

        if use_ms_temporal:
            if self.ms_type == "inception":
                self.ms_temporal = InceptionMultiScale1D(
                    d_model,
                    kernels=(7, 15, 31, 41),
                    dilations=(1, 2, 3, 5),
                    use_maxpool=True,
                    norm="gn",
                    dropout=dropout
                )
            elif self.ms_type == "vanilla":
                self.ms_temporal = MultiScaleTemporal(
                    d_model, kernels=ms_kernels, dilations=ms_dilations,
                    dropout=dropout, depthwise=True
                )
            else:
                raise ValueError(f"ms_type desconocido: {ms_type}")
        else:
            self.ms_temporal = None

        self.layers = nn.ModuleList([
            TransformerBlock1D(
                d_model, num_heads, dropout=dropout,
                pe_type=self.pe_type, rope_theta=rope_theta,
                attn_ms_scales=attn_ms_scales,
                conv_kernel=7,         # puedes bajar a 5 si activas Inception
                conv_norm="gn"
            )
            for _ in range(num_layers)
        ])

        self.out_ln = nn.LayerNorm(d_model)
        self.output_proj = nn.Conv1d(d_model, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, y_cls: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_label_concat:
            if y_cls is None:
                raise ValueError("Se requiere y_cls con use_label_concat=True")
            y_oh = F.one_hot(y_cls, num_classes=self.num_classes).to(dtype=x.dtype, device=x.device)
            y_oh = y_oh.unsqueeze(-1).expand(-1, -1, x.size(-1))
            x = torch.cat([x, y_oh], dim=1)

        x = self.input_proj(x)                      # (B, D, T)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        if self.ms_temporal is not None:
            x = x + self.ms_temporal(x)             # sesgo multiescala

        x = x.permute(0, 2, 1)                      # (B, T, D)
        for blk in self.layers:
            x = blk(x)
        x = self.out_ln(x)                          # (B, T, D)
        x = x.permute(0, 2, 1).contiguous()         # (B, D, T)
        return self.output_proj(x)                  # (B, C_out, T)


# =======================
# 7)  Métricas y pérdidas
# =======================
@torch.no_grad()
def rmse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((y_hat - y) ** 2))

@torch.no_grad()
def pcc(y_hat: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y_hat_c = y_hat - y_hat.mean(dim=-1, keepdim=True)
    y_c     = y - y.mean(dim=-1, keepdim=True)
    num = torch.sum(y_hat_c * y_c, dim=-1)                             # (B,C)
    den = torch.sqrt(torch.sum(y_hat_c**2, dim=-1) * torch.sum(y_c**2, dim=-1) + eps)
    r = num / (den + eps)
    return r.mean()

def pcc_mean_diff(y_hat: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y_hat_c = y_hat - y_hat.mean(dim=-1, keepdim=True)
    y_c     = y     - y.mean(dim=-1, keepdim=True)
    num = (y_hat_c * y_c).sum(dim=-1)  # (B,C)
    den = torch.sqrt((y_hat_c**2).sum(dim=-1) * (y_c**2).sum(dim=-1) + eps)
    r = num / (den + eps)
    return r.mean()

def loss_rmse_pcc(y_hat: torch.Tensor, y: torch.Tensor, alpha: float = 0.6,
                  eps: float = 1e-8, L_torso: Optional[torch.Tensor] = None,
                  lambda_lap: float = 0.0) -> torch.Tensor:
    rmse_term = torch.sqrt(((y_hat - y)**2).mean() + eps)
    pcc_term  = 1.0 - pcc_mean_diff(y_hat, y, eps=eps)
    loss = alpha * rmse_term + (1.0 - alpha) * pcc_term
    if (L_torso is not None) and (lambda_lap > 0.0):
        loss = loss + lambda_lap * laplacian_smoothness(y_hat, L_torso)
    return loss

def laplacian_smoothness(y: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    Ly = torch.matmul(L, y)       # (C,C) @ (B,C,T) -> (B,C,T)
    return (Ly ** 2).mean()

# ========= Rama física (completa: A, logP, mask) =========
class PhysicsMixer(nn.Module):
    """
    y_phys(t) = A @ x(t), con A fila-estocástica (Cout,Cin) derivada de mezcla RBF + top-k.
    También devuelve logP (prior geométrico) y máscara KNN.
    """
    def __init__(self, B_rbf: torch.Tensor, D_ht: torch.Tensor, topk: int = 16,
                 bias_scale_init: float = 0.05, prior_mode: str = "sigma0",
                 rbf_temperature: float = 5.0, dist_p: float = 1.0, eps: float = 1e-12):
        super().__init__()
        assert B_rbf.dim() == 3, "B_rbf debe ser (R, Cin, Cout)"
        R, Cin, Cout = B_rbf.shape
        B = B_rbf.permute(0, 2, 1).contiguous()  # (R, Cout, Cin)
        self.register_buffer("B", B, persistent=False)
        self.R, self.Cout, self.Cin = B.shape
        self.register_buffer("D_tc", D_ht.T.contiguous(), persistent=False)

        with torch.no_grad():
            k = max(1, min(topk, self.Cin))
            base0 = self.B[0]                     # (Cout, Cin)
            _, idx = torch.topk(base0, k=k, dim=1)
            mask = torch.zeros(self.Cout, self.Cin, dtype=torch.bool)
            mask.scatter_(1, idx, True)
        self.register_buffer("knn_mask", mask, persistent=False)

        self.w = nn.Parameter(torch.zeros(self.R))
        self.bias_scale = nn.Parameter(torch.tensor(bias_scale_init))
        self.prior_mode = prior_mode
        self.rbf_temperature = float(rbf_temperature)
        self.dist_p = float(dist_p)
        self.eps = float(eps)

    @staticmethod
    def _big_neg(dtype):
        if dtype in (torch.float16, torch.bfloat16):
            return -1e4
        return -1e9

    def _prior_logP(self, device, dtype):
        mask = self.knn_mask.to(device=device)
        big_neg = self._big_neg(dtype)

        if self.prior_mode == "uniform":
            logits = torch.zeros_like(self.D_tc, dtype=dtype, device=device)
            logits = logits.masked_fill(~mask, big_neg)
            return F.log_softmax(logits.float(), dim=-1).to(dtype)

        elif self.prior_mode == "sigma0":
            logits = (-self.D_tc.to(device=device, dtype=dtype) /
                      max(self.rbf_temperature, 1e-6))
            logits = logits.masked_fill(~mask, big_neg)
            return F.log_softmax(logits.float(), dim=-1).to(dtype)

        elif self.prior_mode == "learned_detach":
            w = self.w.detach().to(device=device, dtype=dtype)
            B = self.B.to(device=device, dtype=dtype)
            logits = torch.einsum('r,rqc->qc', w, B)
            logits = logits.masked_fill(~mask, big_neg)
            return F.log_softmax(logits.float(), dim=-1).to(dtype)

        else:
            raise ValueError(f"prior_mode desconocido: {self.prior_mode}")

    def forward(self, x: torch.Tensor):
        """
        x: (B, Cin, T) -> y_phys: (B,Cout,T), A:(Cout,Cin), logP:(Cout,Cin), mask:(Cout,Cin)
        """
        Bsz, Cin, T = x.shape
        assert Cin == self.Cin, f"Cin={Cin} != {self.Cin}"
        device, dtype = x.device, x.dtype

        w           = self.w.to(device=device, dtype=dtype)
        B           = self.B.to(device=device, dtype=dtype)
        bias_scale  = self.bias_scale.to(device=device, dtype=dtype)
        mask        = self.knn_mask.to(device=device)
        big_neg     = self._big_neg(dtype)

        logits_geo = torch.einsum('r,rqc->qc', w, B)         # (Cout,Cin)
        logits_geo = bias_scale * logits_geo
        logits_geo = logits_geo.masked_fill(~mask, big_neg)

        A = F.softmax(logits_geo.float(), dim=-1).to(device=device, dtype=dtype)  # (Cout,Cin)
        y_phys = torch.einsum('qc,bct->bqt', A, x)           # (B,Cout,T)
        logP = self._prior_logP(device, dtype)
        return y_phys, A, logP, mask

def geo_losses(A: torch.Tensor, logP: torch.Tensor, D_tc: torch.Tensor, mask: torch.Tensor,
               lambda_geo_kl: float = 0.0, lambda_geo_dist: float = 0.0,
               dist_p: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    """
    KL(A || P) + término de distancia A⊙D^p (ambos enmascarados).
    """
    if (lambda_geo_kl <= 0.0) and (lambda_geo_dist <= 0.0):
        return A.new_tensor(0.0)

    maskf = mask.float()
    A_m = A * maskf
    Z = A_m.sum(dim=-1, keepdim=True).clamp_min(eps)
    A_m = A_m / Z

    logA = (A_m + eps).log()
    logP_masked = torch.where(mask, logP, torch.zeros_like(logP))
    kl = (A_m * (logA - logP_masked)).sum(dim=-1).mean()

    dist_term = (A_m * (D_tc.pow(dist_p))).sum(dim=-1).mean()
    return lambda_geo_kl * kl + lambda_geo_dist * dist_term

# =======================
# 8)  Loops de train/eval (AMP + CUDA-friendly)
# =======================
def train_one_epoch(model, loader: DataLoader, optimizer, device, scaler=None, grad_clip: Optional[float] = 1.0,
                    *, phys: Optional[nn.Module] = None, lambda_phys: float = 0.2,
                    L_torso: Optional[torch.Tensor] = None, lambda_lap: float = 0.0,
                    lambda_geo_kl: float = 0.0, lambda_geo_dist: float = 0.0,
                    lambda_cond: float = 0.1, kappa_cond: float = 1.0, win_cond: int = 17,
                    swap_xy: bool = False):
    model.train()
    if phys is not None: phys.train()

    total_loss = 0.0
    use_amp = (device.type == "cuda")
    amp_dtype = AMP_DTYPE if use_amp else torch.float32

    for xb, yb_cls, yb in loader:  # xb:(B,Cin,T)  yb:(B,Cout,T)  yb_cls:(B,)
        if swap_xy:
            xb, yb = yb, xb   # dirección "ecg2emg": entrada=torso, objetivo=corazón
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        yb_cls = yb_cls.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            y_base = model(xb, yb_cls)  # (B,Cout,T)

            if phys is not None:
                y_phys, A, logP, mask = phys(xb)     # A:(Cout,Cin)
                y_hat = y_base + lambda_phys * y_phys
            else:
                y_hat = y_base
                A = logP = mask = None

            loss = loss_rmse_pcc(y_hat, yb, alpha=alpha, L_torso=L_torso, lambda_lap=lambda_lap)

            if (phys is not None) and (A is not None) and ((lambda_geo_kl > 0.0) or (lambda_geo_dist > 0.0) or (lambda_cond > 0.0)):
                geo_reg = geo_losses(
                    A=A, logP=logP,
                    D_tc=phys.D_tc.to(device=xb.device, dtype=xb.dtype),
                    mask=mask, lambda_geo_kl=lambda_geo_kl, lambda_geo_dist=lambda_geo_dist,
                    dist_p=phys.dist_p, eps=1e-12
                )
                if lambda_cond > 0.0:
                    geo_reg = geo_reg + lambda_cond * conduction_consistency(
                        y_hat=y_hat, x=xb, A=A, kappa=kappa_cond, win=win_cond
                    )
                loss = loss + geo_reg

        params_to_clip = list(model.parameters()) + (list(phys.parameters()) if phys is not None else [])
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if (grad_clip is not None) and (grad_clip > 0):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if (grad_clip is not None) and (grad_clip > 0):
                torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip)
            optimizer.step()

        total_loss += loss.item() * xb.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader: DataLoader, device, *, phys: Optional[nn.Module] = None, lambda_phys: float = 0.2,
             L_torso: Optional[torch.Tensor] = None, lambda_lap: float = 0.0,
             lambda_geo_kl: float = 0.0, lambda_geo_dist: float = 0.0,
             lambda_cond: float = 0.1, kappa_cond: float = 1.0, win_cond: int = 17,
             swap_xy: bool = False) -> Dict[str, float]:
    model.eval()
    if phys is not None: phys.eval()

    tot_loss = tot_rmse = tot_pcc = 0.0
    for xb, yb_cls, yb in loader:
        if swap_xy:
            xb, yb = yb, xb
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        yb_cls = yb_cls.to(device, non_blocking=True)

        y_base = model(xb, yb_cls)

        geo_reg = xb.new_tensor(0.0)
        if phys is not None:
            y_phys, A, logP, mask = phys(xb)
            y_hat = y_base + lambda_phys * y_phys
            if (lambda_geo_kl > 0.0) or (lambda_geo_dist > 0.0):
                geo_reg = geo_losses(
                    A=A, logP=logP,
                    D_tc=phys.D_tc.to(device=xb.device, dtype=xb.dtype),
                    mask=mask, lambda_geo_kl=lambda_geo_kl, lambda_geo_dist=lambda_geo_dist,
                    dist_p=phys.dist_p, eps=1e-12
                )
            if lambda_cond > 0.0:
                geo_reg = geo_reg + lambda_cond * conduction_consistency(
                    y_hat=y_hat, x=xb, A=A, kappa=kappa_cond, win=win_cond
                )
        else:
            y_hat = y_base

        loss = loss_rmse_pcc(y_hat, yb, alpha=alpha, L_torso=L_torso, lambda_lap=lambda_lap)
        loss = loss + geo_reg

        tot_loss += loss.item() * xb.size(0)
        tot_rmse += rmse(y_hat, yb).item() * xb.size(0)
        tot_pcc  += pcc(y_hat, yb).item()  * xb.size(0)

    n = len(loader.dataset)
    return {"loss": tot_loss/n, "rmse": tot_rmse/n, "pcc": tot_pcc/n}

# =======================
# 9)  Inspección y Entrenamiento
# =======================
def run_inspection():
    set_seed(SEED)
    for dataset_name, info in DATASETS.items():
        path = Path(info["path"])
        if not path.exists():
            print(f"\n⚠️  Archivo no encontrado para {dataset_name}: {path}")
            continue

        try:
            keys = list_mat_keys(path)
            print(f"\n🗂  Keys en {path.name}:")
            for k, shape, typ in keys:
                print(f"   - {k:<20} shape={shape}  type={typ}")
        except Exception as e:
            print(f"\n(No se pudieron listar keys: {e})")

        train_loader, val_loader, full_ds = build_loader(
            path=path,
            batch_size=8,
            device=DEVICE.type,
            split_ratio=0.7,
            seed=SEED,
            keys=info.get("keys", None),
            apply_filter=True,
        )

        summarize_dataset(dataset_name, full_ds)
        peek_batch(train_loader, "train")
        peek_batch(val_loader, "val")

        geo = build_geometry_pack(full_ds, sigmas=[20., 40., 80.])
        summarize_geometry(full_ds, geo)

def _fmt_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("+", "")

def _append_metrics_csv(csv_path: Path, row: Dict[str, Any]):
    """Agrega `row` a metrics_standalone.csv (lo crea con header si no existe)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def run_basic_training(mat_path: Path, keys: dict, epochs: int = 20, batch_size: int = 8,
                       lr: float = 1e-3, d_model: int = 128, num_heads: int = 4, num_layers: int = 4,
                       dropout: float = 0.1, split_ratio: float = 0.7, seed: int = 125,
                       apply_filter: bool = True, use_torch_compile: bool = False,
                       dataset_name: str = "dataset", output_dir: Optional[Path] = None,
                       direction: str = "emg2ecg"):
    """
    Entrena Transformer1D + rama física con regularizadores geométricos
    y la restricción de conducción (RMS_torso <= κ · A @ RMS_corazón).

    direction:
      - "emg2ecg" (default del notebook original): entrada=corazón (emg),
        objetivo=torso (ecg) — problema DIRECTO (propagación corazón→torso),
        bien planteado.
      - "ecg2emg" (el que usa src/run_bohb.py por defecto, y el problema real
        de ECGI): entrada=torso (ecg), objetivo=corazón (emg) — problema
        INVERSO, mal planteado. `PhysicsMixer` sigue usando el mismo kernel
        de mezcla ponderado por distancia (y_phys = A @ x); ese kernel es un
        prior físicamente razonable para el problema directo (conducción de
        volumen), pero en el sentido inverso NO es una solución regularizada
        del problema inverso (no hay Tikhonov ni nada equivalente) — es solo
        un promedio ponderado por cercanía geométrica, reinterpretado.

    Importante: `keys` (qué campo del .mat es "emg"/"ecg") NO se toca según
    `direction` — ECGDataset infiere las longitudes de electrodos de torso y
    corazón (Ce/Cx) a partir de cuál señal cae en x/y_bspm, así que swapear
    las claves de carga rompería el mapeo geométrico torso/corazón. El swap
    de dirección se hace después de cargar (qué tensor es entrada del modelo
    y cuál es objetivo de la pérdida), no en la carga del .mat.
    """
    set_seed(seed)
    if direction not in ("emg2ecg", "ecg2emg"):
        raise ValueError(f"direction desconocida: {direction!r} (usa 'emg2ecg' o 'ecg2emg')")
    swap_xy = (direction == "ecg2emg")

    train_loader, val_loader, full_ds = build_loader(
        path=mat_path, batch_size=batch_size, device=DEVICE.type,
        split_ratio=split_ratio, seed=seed, keys=keys, apply_filter=apply_filter
    )
    # full_ds.x = emg (corazón), full_ds.y_bspm = ecg (torso), siempre —
    # ver nota arriba. Cin/Cout del modelo sí dependen de `direction`.
    if swap_xy:
        Cin  = full_ds.y_bspm.shape[1]   # ecg/torso como entrada
        Cout = full_ds.x.shape[1]        # emg/corazón como objetivo
    else:
        Cin  = full_ds.x.shape[1]
        Cout = full_ds.y_bspm.shape[1]
    T = full_ds.x.shape[2]
    print(f"\nShapes: Cin={Cin}, Cout={Cout}, T={T} | device={DEVICE.type}  | direction={direction}")

    geo = build_geometry_pack(full_ds, sigmas=[20., 40., 80.])
    summarize_geometry(full_ds, geo)

    if direction == "ecg2emg":
        # geo["B_rbf"]/["D_ht"] están calculados como (heart→torso) =
        # (Cin,Cout) del problema DIRECTO; para el problema inverso hay que
        # transponerlos para que calcen con Cin=Ce(torso), Cout=Cx(corazón).
        B_rbf_in = geo["B_rbf"].permute(0, 2, 1).contiguous()
        D_ht_in  = geo["D_ht"].T.contiguous()
        L_target = geo["L_heart_elec"].to(DEVICE)
    else:
        B_rbf_in = geo["B_rbf"]
        D_ht_in  = geo["D_ht"]
        L_target = geo["L_torso_elec"].to(DEVICE)

    num_classes = int(full_ds.y.max().item() + 1)
    model = Transformer1D(
        time_steps=T, input_channels=Cin, output_channels=Cout,
        d_model=d_model, num_heads=num_heads, num_layers=num_layers, dropout=dropout,
        pe_type="rope", rope_theta=10000.0,
        use_ms_temporal=True,
        ms_type="inception",                  # <--- activa Inception aquí
        ms_kernels=(3, 7, 15), ms_dilations=(1, 2, 4),   # se usan si ms_type="vanilla"
        attn_ms_scales=(1, 2, 4),
        use_label_concat=True, num_classes=num_classes
    ).to(DEVICE)

    if use_torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    phys = PhysicsMixer(B_rbf_in, D_ht_in, topk=16, bias_scale_init=0.05,
                        prior_mode="sigma0", rbf_temperature=0.3, dist_p=1.0)

    lambda_phys     = 0.2
    lambda_lap      = 0.0
    lambda_geo_kl   = 0.05
    lambda_geo_dist = 0.02
    lambda_cond     = 0.1
    kappa_cond      = 1.0
    win_cond        = 17

    L_torso = L_target  # lambda_lap=0.0 abajo, así que este término está inactivo de todas formas

    decay, no_decay = [], []
    for n, p in list(model.named_parameters()) + list(phys.named_parameters()):
        (no_decay if (p.ndim==1 or n.endswith('bias')) else decay).append(p)
    opt = torch.optim.AdamW([
        {'params': decay, 'weight_decay': 1e-3},
        {'params': no_decay, 'weight_decay': 0.0},
    ], lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=lr*0.05)

    try:
        scaler = torch.amp.GradScaler(DEVICE.type, enabled=(DEVICE.type == "cuda"))
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    warmup_epochs = 8
    warmup_target = 0.7

    best_val = float("inf")
    best_metrics = None
    lr_tag = _fmt_lr(lr)
    out_dir = Path(output_dir) if output_dir is not None else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"t1d_phys_best_lr{lr_tag}_seed{seed}.pt"

    t_start = time.time()
    for ep in range(1, epochs + 1):
        # warmup de la escala de sesgo de la rama física
        t = min(max(ep, 0) / float(warmup_epochs), 1.0)
        with torch.no_grad():
            phys.bias_scale.copy_(torch.tensor(max(1e-6, t * warmup_target), device=phys.bias_scale.device, dtype=phys.bias_scale.dtype))

        tr_loss = train_one_epoch(
            model, train_loader, opt, DEVICE, scaler=scaler, grad_clip=1.0,
            phys=phys, lambda_phys=lambda_phys,
            L_torso=L_torso, lambda_lap=lambda_lap,
            lambda_geo_kl=lambda_geo_kl, lambda_geo_dist=lambda_geo_dist,
            lambda_cond=lambda_cond, kappa_cond=kappa_cond, win_cond=win_cond,
            swap_xy=swap_xy,
        )

        val_metrics = evaluate(
            model, val_loader, DEVICE,
            phys=phys, lambda_phys=lambda_phys,
            L_torso=L_torso, lambda_lap=lambda_lap,
            lambda_geo_kl=lambda_geo_kl, lambda_geo_dist=lambda_geo_dist,
            lambda_cond=lambda_cond, kappa_cond=kappa_cond, win_cond=win_cond,
            swap_xy=swap_xy,
        )
        scheduler.step(val_metrics["loss"])

        print(f"[{ep:03d}] train_loss={tr_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f}  rmse={val_metrics['rmse']:.4f}  "
              f"pcc={val_metrics['pcc']:.4f}  lr={opt.param_groups[0]['lr']:.3e}  "
              f"bias_scale={phys.bias_scale.item():.3f}")

        if val_metrics["loss"] < best_val - 1e-6:
            best_val = val_metrics["loss"]
            best_metrics = {"loss": val_metrics["loss"],
                            "rmse": val_metrics["rmse"],
                            "pcc":  val_metrics["pcc"]}
            torch.save({
                "state_dict": model.state_dict(),
                "phys_state_dict": phys.state_dict(),
                "config": dict(
                    T=T, Cin=Cin, Cout=Cout, d_model=d_model,
                    num_heads=num_heads, num_layers=num_layers, dropout=dropout,
                    lambda_phys=lambda_phys, lambda_lap=lambda_lap,
                    lambda_geo_kl=lambda_geo_kl, lambda_geo_dist=lambda_geo_dist,
                    lambda_cond=lambda_cond, kappa_cond=kappa_cond, win_cond=win_cond,
                    warmup_epochs=warmup_epochs, warmup_target=warmup_target,
                    lr=lr, seed=seed, prior_mode="sigma0", rbf_temperature=0.3, dist_p=1.0, topk=16
                )
            }, best_path)

    elapsed = time.time() - t_start
    print(f"\n✔ Mejor modelo+física guardados en: {best_path}")

    _append_metrics_csv(OUTPUT_ROOT / "metrics_standalone.csv", {
        "dataset": dataset_name,
        "direction": direction,
        "lr": lr,
        "seed": seed,
        "epochs": epochs,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dropout": dropout,
        "split_ratio": split_ratio,
        "apply_filter": apply_filter,
        "best_val_loss": (best_metrics or {}).get("loss"),
        "best_rmse": (best_metrics or {}).get("rmse"),
        "best_pcc": (best_metrics or {}).get("pcc"),
        "elapsed_sec": round(elapsed, 1),
        "ckpt_path": str(best_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    return {"lr": lr, "seed": seed, **(best_metrics or {}), "ckpt": str(best_path)}


# =======================
# 10)  CLI
# =======================
def main():
    p = argparse.ArgumentParser(
        description="PhysFormer standalone: réplica de physformer1.0.ipynb "
                     "(backbone Transformer1D + PhysicsMixer), aislado del "
                     "pipeline comparativo de src/run_bohb.py."
    )
    p.add_argument("--dataset", choices=list(DATASETS.keys()) + ["all"], default="all",
                    help="Dataset a entrenar, o 'all' para correr los 4 en secuencia.")
    p.add_argument("--direction", choices=["ecg2emg", "emg2ecg"], default="ecg2emg",
                    help="ecg2emg (default): torso→corazón, el problema inverso real de ECGI "
                         "y el que usa src/run_bohb.py por defecto. emg2ecg: corazón→torso "
                         "(problema directo), el que entrenaba physformer1.0.ipynb original.")
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
    p.add_argument("--use-torch-compile", action="store_true", default=False)
    p.add_argument("--inspect", action="store_true",
                    help="Solo corre run_inspection() (shapes/keys/geometría) y sale.")
    args = p.parse_args()

    if args.inspect:
        run_inspection()
        return

    names = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for name in names:
        info = DATASETS[name]
        mat_path = Path(info["path"])
        if not mat_path.exists():
            print(f"⚠️  Salto '{name}': no encontrado en {mat_path}")
            continue
        print(f"\n{'='*80}\n>>> PhysFormer standalone — dataset={name}  direction={args.direction}\n{'='*80}")
        run_basic_training(
            mat_path=mat_path,
            keys=info["keys"],
            lr=args.lr,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            d_model=args.d_model, num_heads=args.num_heads,
            num_layers=args.num_layers, dropout=args.dropout,
            split_ratio=args.split_ratio, apply_filter=args.apply_filter,
            use_torch_compile=args.use_torch_compile,
            dataset_name=name,
            output_dir=OUTPUT_ROOT / name / args.direction,
            direction=args.direction,
        )


if __name__ == "__main__":
    main()
