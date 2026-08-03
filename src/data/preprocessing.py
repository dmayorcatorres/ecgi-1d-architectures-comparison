# ecgdl/data/preprocessing.py
import hashlib, mmap, pickle, torch
from pathlib import Path
import scipy.io as sio
import torch.nn.functional as F

def _sha1(path: Path) -> str:
    """Calcular hash SHA1 del archivo para el cache."""
    import hashlib
    sha1 = hashlib.sha1()
    with open(path, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()[:8]

def load_mat_cached(mat_path: Path, time_steps: int, key_x="emg", key_y="tag", key_bspm="ecg") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache = mat_path.with_suffix(f".T{time_steps}.labels.{_sha1(mat_path)}.pt")
    if cache.exists():
        ecg, emg, labels = torch.load(cache, map_location="cpu", mmap=True, weights_only=True)
        # Revalidar por si el caché quedó corrupto
        if torch.isnan(ecg).any() or torch.isnan(emg).any():
            cache.unlink(missing_ok=True)
        else:
            return ecg, emg, labels

    d = sio.loadmat(mat_path)

    ecg = torch.as_tensor(d[key_bspm]).float()
    emg = torch.as_tensor(d[key_x]).float()
    # Sanea posibles no finitos
    ecg = torch.nan_to_num(ecg)
    emg = torch.nan_to_num(emg)
    
    # Cargar y procesar etiquetas
    y_raw = torch.as_tensor(d[key_y]).long().squeeze()
    uniq = torch.unique(y_raw)
    remap = {int(u): i for i, u in enumerate(sorted(uniq.tolist()))}
    labels = torch.tensor([remap[int(v)] for v in y_raw])

    # Reducir dimensionalidad temporal
    ecg = paa_reduce(ecg, time_steps)
    emg = paa_reduce(emg, time_steps)
    ecg = torch.nan_to_num(ecg)
    emg = torch.nan_to_num(emg)
    
    # Guardar en cache incluyendo etiquetas
    cache.parent.mkdir(exist_ok=True)
    torch.save((ecg, emg, labels), cache)
    
    return ecg, emg, labels

def paa_reduce(x: torch.Tensor, T: int) -> torch.Tensor:
    """
    Reduce o expande la longitud temporal a T.
    - Si T_raw == T: devuelve x.
    - Si T_raw > T: usa PAA cuando hay bloques enteros; si no, interpola.
    - Si T_raw < T: interpola (upsampling) para evitar bloques vacíos y NaNs.
    x: (B, C, T_raw)
    """
    B, C, T_raw = x.shape
    if T_raw == T:
        return x

    if T_raw > T:
        block = T_raw // T
        if block >= 2:
            x_trim = x[..., :block * T]
            return x_trim.view(B, C, T, block).mean(dim=-1)
        # Si block==1 (downsample leve) o no cuadra, usa interpolación
        return F.interpolate(x, size=T, mode="linear", align_corners=False)

    # T_raw < T -> upsampling por interpolación
    return F.interpolate(x, size=T, mode="linear", align_corners=False)

