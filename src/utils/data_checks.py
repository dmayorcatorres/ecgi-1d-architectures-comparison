# utils/data_checks.py
import torch
from typing import Tuple, List

def _tensor_summary(x: torch.Tensor):
    x_cpu = x.detach().cpu()
    n = x_cpu.numel()
    nan = torch.isnan(x_cpu).sum().item()
    inf = torch.isinf(x_cpu).sum().item()
    finite = torch.isfinite(x_cpu)
    mn = x_cpu[finite].min().item() if finite.any() else float("nan")
    mx = x_cpu[finite].max().item() if finite.any() else float("nan")
    mean = x_cpu[finite].mean().item() if finite.any() else float("nan")
    std  = x_cpu[finite].std().item()  if finite.any() else float("nan")
    return dict(n=n, nan=nan, inf=inf, min=mn, max=mx, mean=mean, std=std)

def assert_no_nan_batch(xb: torch.Tensor, yb: torch.Tensor, tag: str = ""):
    sx = _tensor_summary(xb); sy = _tensor_summary(yb)
    if sx["nan"] or sx["inf"] or sy["nan"] or sy["inf"]:
        raise ValueError(
            f"[NaN/Inf detectado] {tag} | "
            f"X(n={sx['n']}, nan={sx['nan']}, inf={sx['inf']}, "
            f"min={sx['min']:.3g}, max={sx['max']:.3g}, mean={sx['mean']:.3g}, std={sx['std']:.3g}); "
            f"Y(n={sy['n']}, nan={sy['nan']}, inf={sy['inf']}, "
            f"min={sy['min']:.3g}, max={sy['max']:.3g}, mean={sy['mean']:.3g}, std={sy['std']:.3g})"
        )

def dataset_quick_scan(dm, max_batches: int = 2):
    """
    Recorre unas pocas batches de train/val y reporta NaN/Inf y rangos.
    Útil para datasets problemáticos (p.ej., utah2002).
    """
    print("\n[DataScan] ----")
    try:
        tr = dm.train_dataloader()
        for bi, (xb, yb, _) in enumerate(tr):
            if bi >= max_batches: break
            try:
                assert_no_nan_batch(xb, yb, tag=f"train_b{bi}")
            except Exception as e:
                print(str(e))
            sx = _tensor_summary(xb); sy = _tensor_summary(yb)
            print(f"[train_b{bi}] X: {sx} | Y: {sy}")
    except Exception as e:
        print(f"[DataScan] error en train_dataloader: {e}")

    try:
        va = dm.val_dataloader()
        for bi, (xb, yb, _) in enumerate(va):
            if bi >= max_batches: break
            try:
                assert_no_nan_batch(xb, yb, tag=f"val_b{bi}")
            except Exception as e:
                print(str(e))
            sx = _tensor_summary(xb); sy = _tensor_summary(yb)
            print(f"[val_b{bi}]   X: {sx} | Y: {sy}")
    except Exception as e:
        print(f"[DataScan] error en val_dataloader: {e}")

def install_nan_detector(model: torch.nn.Module):
    """
    Registra hooks en cada módulo para detectar salidas con NaN/Inf y
    abortar con un mensaje que indica la capa responsable.
    Devuelve la lista de handles para poder removerlos después.
    """
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def _hook(module, inp, out):
        def _check(t: torch.Tensor, side: str):
            if t is None or not isinstance(t, torch.Tensor): return
            bad = ~torch.isfinite(t)
            if bad.any():
                where = bad.nonzero(as_tuple=False)
                raise RuntimeError(
                    f"[NaN/Inf en {module.__class__.__name__}] "
                    f"side={side}, shape={tuple(t.shape)}, bad_count={bad.sum().item()}"
                )
        if isinstance(out, (tuple, list)):
            for t in out: 
                if isinstance(t, torch.Tensor): _check(t, "out")
        elif isinstance(out, torch.Tensor):
            _check(out, "out")

    for m in model.modules():
        # opcional: limitar a capas con parámetros para menos ruido
        if any(p.requires_grad for p in m.parameters(recurse=False)) or hasattr(m, "weight"):
            h = m.register_forward_hook(_hook)
            handles.append(h)
    return handles
