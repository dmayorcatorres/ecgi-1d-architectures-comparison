# ecgdl/data/preprocessing.py
import hashlib, mmap, pickle, torch
from pathlib import Path
import scipy.io as sio

def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()[:12]

def load_mat_cached(mat_path: Path, time_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    cache = mat_path.with_suffix(f".T{time_steps}.{_sha1(mat_path)}.pt")
    if cache.exists():
        return torch.load(cache, map_location="cpu", mmap=True, weights_only=True)

    d = sio.loadmat(mat_path)
    ecg = torch.as_tensor(d["ecg"]).float()      # (N, C_in, T_raw)
    emg = torch.as_tensor(d["emg"]).float()      # (N, C_out, T_raw)

    ecg = paa_reduce(ecg, time_steps)
    emg = paa_reduce(emg, time_steps)
    cache.parent.mkdir(exist_ok=True)
    torch.save((ecg, emg), cache)
    return ecg, emg

def paa_reduce(x: torch.Tensor, T: int) -> torch.Tensor:
    """Piece-wise Aggregate Approximation vectorizada"""
    B, C, T_raw = x.shape
    if T_raw == T: return x
    block = T_raw // T
    x = x[..., :block * T]            # truncar
    x = x.view(B, C, T, block)
    return x.mean(dim=-1)
