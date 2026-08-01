# utils/logbook.py
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import torch

def _to_float(x):
    return float(x.item()) if isinstance(x, torch.Tensor) else float(x)

def log_metrics_to_excel(
    path: Path,
    rpt: Dict,
    n_p: int,
    lat: float,
    flops: float,
    macs: float,
    best_trial_str: str,
    method: str,
    dataset: str,
    extra: Optional[Dict] = None,
):
    """Añade una fila con resultados al Excel en 'path' (lo crea si no existe)."""
    row = {
        "dataset": dataset,
        "method": method,
        **{f"{k}_mean": _to_float(v[0]) if isinstance(v, tuple) else _to_float(v)
           for k, v in rpt.items()},
        **{f"{k}_std": _to_float(v[1])
           for k, v in rpt.items() if isinstance(v, tuple)},
        "num_params": n_p,
        "inference_latency_ms": float(lat),
        "FLOPs": float(flops),
        "MACs": float(macs),
        "best_trial": best_trial_str,
        "timestamp": pd.Timestamp.now()
    }
    if extra:
        row.update(extra)

    if path.exists():
        df = pd.read_excel(path)
        for c in row:
            if c not in df.columns:
                df[c] = pd.NA
        df.loc[len(df)] = row
    else:
        df = pd.DataFrame([row])

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)
