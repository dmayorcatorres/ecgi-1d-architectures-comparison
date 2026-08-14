# data/geometry.py
#
# Geometría de malla torso/corazón para PhysMixFormer1D (models/atbased/
# PhysMixFormer1D.py). Portado de aux/physformer_standalone/
# physmixformer_standalone_v3.py (helpers de lectura de structs MATLAB,
# submesh, laplaciano gaussiano k-NN con distancia geodésica y chequeo de
# fiabilidad de la malla) — self-contenido, no depende del resto del
# pipeline. Se llama UNA vez por dataset (no por trial BOHB), ver
# run_bohb.py::run_pipeline.
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy.io as sio
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


@dataclass
class GeometryPack:
    """Salida de load_geometry() — todo lo que PhysMixFormer1D.attach_geometry
    necesita para construir Π (RowStochasticCoupling), la rama física
    (PhysicsBranch) y el gauge de entrada/salida."""
    coords_torso: torch.Tensor  # (Et, 3)
    faces_torso:  torch.Tensor  # (Ft, 3)
    coords_heart: torch.Tensor  # (Ee, 3)
    faces_heart:  torch.Tensor  # (Fh, 3)
    L_torso: torch.Tensor       # (Et, Et) laplaciano normalizado
    deg_torso: torch.Tensor     # (Et,)
    L_heart: torch.Tensor       # (Ee, Ee) laplaciano normalizado
    deg_heart: torch.Tensor     # (Ee,)
    D_te: torch.Tensor          # (Et, Ee) distancias torso<->corazón


def loadmat_smart(path: Union[str, Path]) -> Dict[str, Any]:
    path = str(path)
    try:
        return sio.loadmat(path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    except TypeError:
        return sio.loadmat(path, squeeze_me=True, struct_as_record=False)


# ── helpers robustos para structs MATLAB ────────────────────────────────────
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


# ── Laplaciano gaussiano k-NN con distancia geodésica ───────────────────────
def _edges_from_faces(faces: torch.Tensor) -> torch.Tensor:
    f = faces.long()
    if f.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.long)
    return torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]], dim=0)


def geodesic_distance_matrix(coords: torch.Tensor, faces: torch.Tensor) -> Optional[torch.Tensor]:
    """Dijkstra sobre las aristas de la malla, pesadas por longitud euclídea."""
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
    """(L_normalizado, grado): a_ij = exp(-d(i,j)²/σ²) en la máscara k-NN,
    L = I - D^{-1/2} A D^{-1/2}."""
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


def gauge_weights_from_degree(deg: torch.Tensor) -> torch.Tensor:
    """Ω = D (grado). Si el grado es todo-cero (malla no confiable), cae a
    pesos uniformes."""
    if deg.numel() == 0 or deg.sum() <= 0:
        return torch.ones_like(deg) if deg.numel() > 0 else deg
    return deg.clamp_min(1e-8)


def gauge_projector_matrix(weights: torch.Tensor) -> torch.Tensor:
    """P_E^(w) = I - (1 1^T Ω) / (1^T Ω 1). weights: (E,) > 0."""
    E = weights.shape[0]
    w = weights.view(-1).clamp_min(1e-12)
    ones = torch.ones(E, dtype=w.dtype, device=w.device)
    denom = (ones * w).sum().clamp_min(1e-12)
    return torch.eye(E, dtype=w.dtype, device=w.device) - torch.outer(ones, w) / denom


def load_geometry(
    mat_path: Path,
    n_input_channels: int,
    n_output_channels: int,
    key_torso_mesh: str = "torso",
    key_heart_mesh: str = "heart",
    sigma_t: float = 30.0,
    sigma_e: float = 30.0,
    knn_k: int = 6,
) -> Optional[GeometryPack]:
    """Lee las mallas torso/corazón del .mat y construye el paquete de
    geometría para PhysMixFormer1D. Devuelve None si:
      - el .mat no trae los structs 'torso'/'heart' (o alias reconocidos), o
      - la malla no está a resolución de electrodo (nº de nodos != nº de
        canales reales) -- en ese caso el mapeo canal<->nodo asumido
        ("primeros N nodos") no está verificado y la física se desactiva
        (mismo criterio que aux/physformer_standalone/
        physmixformer_standalone_v3.py::ECGDataset.has_reliable_geometry).
    """
    mat = loadmat_smart(mat_path)
    torso_obj = mat.get(key_torso_mesh, None)
    heart_obj = mat.get(key_heart_mesh, None)
    if torso_obj is None or heart_obj is None:
        return None

    try:
        torso_nodes, torso_faces = _get_nodes_faces(torso_obj)
        heart_nodes, heart_faces = _get_nodes_faces(heart_obj)
    except ValueError:
        return None

    torso_elec_ids = _get_electrode_ids(torso_obj, n_input_channels)
    heart_elec_ids = _get_electrode_ids(heart_obj, n_output_channels)

    has_reliable_geometry = (
        torso_nodes.shape[0] == n_input_channels and heart_nodes.shape[0] == n_output_channels
    )
    if not has_reliable_geometry:
        return None

    if len(torso_elec_ids) != n_input_channels:
        torso_elec_ids = torso_elec_ids[:n_input_channels]
    if len(heart_elec_ids) != n_output_channels:
        heart_elec_ids = heart_elec_ids[:n_output_channels]

    P_t, F_t = submesh(torso_nodes, torso_faces, torso_elec_ids)
    P_h, F_h = submesh(heart_nodes, heart_faces, heart_elec_ids)

    L_t, deg_t = gaussian_knn_laplacian(P_t, F_t, sigma=sigma_t, k=knn_k)
    L_e, deg_e = gaussian_knn_laplacian(P_h, F_h, sigma=sigma_e, k=knn_k)

    D_te = pairwise_dist_3d(P_t, P_h)  # (Et, Ee) — distancias torso-epicardio

    return GeometryPack(
        coords_torso=P_t, faces_torso=F_t,
        coords_heart=P_h, faces_heart=F_h,
        L_torso=L_t, deg_torso=deg_t,
        L_heart=L_e, deg_heart=deg_e,
        D_te=D_te,
    )
