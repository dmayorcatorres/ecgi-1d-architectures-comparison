# utils/profile.py
import time
import math
import torch
from thop import profile as thop_profile

def _cleanup_thop_artifacts(model: torch.nn.Module):
    """Elimina residuos que THOP puede dejar en el modelo."""
    for m in model.modules():
        for name in ("total_ops", "total_params"):
            try:
                if hasattr(m, "_buffers") and name in m._buffers:
                    del m._buffers[name]
                elif hasattr(m, name):
                    delattr(m, name)
            except Exception:
                pass
        if hasattr(m, "_thop_hooks"):
            try:
                for h in m._thop_hooks:
                    try: h.remove()
                    except Exception: pass
                delattr(m, "_thop_hooks")
            except Exception:
                pass

def compute_model_stats(model: torch.nn.Module, sample: torch.Tensor, device: str = "cpu"):
    """
    Calcula #param, latencia media y (si es posible) MACs/FLOPs con THOP.
    Devuelve (num_params, latency_ms, flops, macs). FLOPs/MACs pueden ser NaN si falla.
    """
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())

    # warmup
    with torch.inference_mode():
        for _ in range(10):
            _ = model(sample)

    # latencia
    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); start.record()
        with torch.inference_mode():
            for _ in range(50):
                _ = model(sample)
        end.record(); torch.cuda.synchronize()
        latency_ms = start.elapsed_time(end) / 50.0
    else:
        t0 = time.time()
        with torch.inference_mode():
            for _ in range(50):
                _ = model(sample)
        latency_ms = (time.time() - t0) * 1000.0 / 50.0

    flops = float("nan"); macs = float("nan")
    try:
        _cleanup_thop_artifacts(model)
        macs, _ = thop_profile(model, inputs=(sample,), verbose=False)
        flops = 2 * macs
        _cleanup_thop_artifacts(model)
    except Exception as e:
        print(f"[THOP] fallo perfilando: {e}. Sigo sin FLOPs/MACs.")
    return num_params, float(latency_ms), float(flops), float(macs)
