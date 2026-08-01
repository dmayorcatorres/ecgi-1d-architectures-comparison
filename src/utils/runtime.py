# utils/runtime.py
import torch
import random
import numpy as np
from typing import Any

STRICT_DETERMINISM = False  # puedes sobreescribir desde el main si quieres

def setup_torch_runtime(strict: bool = STRICT_DETERMINISM):
    """TF32, determinismo razonable y bench de cuDNN."""
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    torch.backends.cudnn.deterministic = strict
    torch.backends.cudnn.benchmark     = not strict
    try:
        torch.use_deterministic_algorithms(strict, warn_only=True)
    except Exception:
        pass

def amp_config(device: str):
    """Devuelve (enabled, dtype, use_scaler) según HW."""
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return dict(enabled=True, dtype=torch.bfloat16, use_scaler=False)
        else:
            return dict(enabled=True, dtype=torch.float16,  use_scaler=True)
    return dict(enabled=False, dtype=None, use_scaler=False)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def base_module(m: torch.nn.Module) -> torch.nn.Module:
    """Devuelve el módulo real si fue torch.compile()."""
    return getattr(m, "_orig_mod", m)

def free_all(*objs: Any):
    """Liberación agresiva de memoria (CPU/GPU)."""
    import gc
    for o in objs:
        try: del o
        except: pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    gc.collect()
