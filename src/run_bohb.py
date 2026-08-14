#!/usr/bin/env python
# run_bohb.py — Pipeline AutoML reproducible y moderno
#
#  Split:     60 % train / 20 % val (HPO + early-stop) / 20 % test (métricas finales)
#  HPO:       BOHB — TuneBOHB + HyperBandForBOHB (reduction_factor=3)
#  Optimizer: AdamW (weight decay desacoplado)
#  Schedule:  Warmup lineal 5 ep → CosineAnnealingLR  (entrenamiento final)
#             Sin scheduler en trials HPO  (señal limpia para BOHB)
#  Métricas:  Calculadas exclusivamente sobre el TEST held-out (no contaminado)
#
#  Uso:
#    python run_bohb.py --dataset kitpvc --method resnet
#    python run_bohb.py --dataset kitpvc --method resnet --save-curves
#    python run_bohb.py --dataset kitpvc --method resnet --num-samples 30
#    SANITY_CHECKS=1 python run_bohb.py --dataset kitpvc --method mlp
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset, TensorDataset

import ray
from ray import tune
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*does not support bfloat16 compilation natively.*",
    category=UserWarning,
)

log = logging.getLogger(__name__)

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT    = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT))

# ── Constantes ────────────────────────────────────────────────────────────────
GLOBAL_SEED   = 42
# Split 3 vías estratificado
TRAIN_FRAC    = 0.60   # 60 % — entrenamiento
VAL_FRAC      = 0.20   # 20 % — señal HPO + early stopping
# TEST_FRAC   = 0.20   # 20 % — métricas finales (implícito: 1 - TRAIN - VAL)
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SANITY_CHECKS = bool(int(os.getenv("SANITY_CHECKS", "0")))

# Claves del dict de configuración que NO son hiperparámetros del modelo
_NONMODEL_KEYS = frozenset({
    "batch_size", "lr", "weight_decay", "epochs",
    "variant", "mat_path", "time_steps",
    "input_channels", "output_channels",
    "direction", "_best_trial_str",
})

# ── Imports del repo ──────────────────────────────────────────────────────────
from config import DataConfig
from models.factory import ModelFactory
from models.register import MODEL_REGISTRY, SPACE_REGISTRY
from hpo.adapters import space_to_tune
from data.datamodule import ECGDataModule
from data.geometry import load_geometry
from data.filtering import load_filtered, DATASET_FS, DATASET_LP_CUT
from utils.metrics import ECGMetricsCalculator
from utils.early_stopping import EarlyStopping
from utils.runtime import setup_torch_runtime, amp_config, set_seed, base_module, free_all
from utils.ray_helpers import bytes_from_shm, spilling_config
from utils.profile import compute_model_stats
from utils.split import stratified_split_indices_approx
from utils.logbook import log_metrics_to_excel
from utils.data_checks import install_nan_detector, assert_no_nan_batch

setup_torch_runtime()


# ═════════════════════════════ Helpers ════════════════════════════════════════

def _sanitize(cfg):
    """Convierte np.generic en tipos Python nativos (necesario tras Ray Tune)."""
    def _s(x):
        if isinstance(x, np.generic):         return x.item()
        if isinstance(x, dict):               return {k: _s(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):      return type(x)(_s(v) for v in x)
        return x
    return _s(cfg)


def _clip_to_min_len(pred: torch.Tensor, target: torch.Tensor):
    """Recorta temporalmente pred y target al mínimo si difieren (modelos con padding)."""
    if pred.size(-1) != target.size(-1):
        M = min(pred.size(-1), target.size(-1))
        pred, target = pred[..., :M], target[..., :M]
    return pred, target


def _worker_init(worker_id: int):
    """Semilla derivada para cada worker del DataLoader — garantiza reproducibilidad."""
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def _make_loader(
    dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int
) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_worker_init if num_workers > 0 else None,
        generator=g if shuffle else None,
    )


def _three_way_split(
    labels: torch.Tensor, seed: int
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split estratificado en dos pasos encadenados:
      Paso 1 — dev (80 %) vs test (20 %)          stratified por etiqueta
      Paso 2 — train (75 % de dev) vs val (25 %)  stratified por etiqueta

    Resultado: 60 % train / 20 % val / 20 % test del total.
    Verifica ausencia de solapamiento entre los tres conjuntos.
    """
    dev_frac   = TRAIN_FRAC + VAL_FRAC          # 0.80
    inner_frac = TRAIN_FRAC / dev_frac           # 0.75 dentro de dev

    dev_idx, test_idx = stratified_split_indices_approx(labels, dev_frac,   seed=seed)
    inner_tr, inner_va = stratified_split_indices_approx(
        labels[dev_idx], inner_frac, seed=seed
    )
    train_idx = [dev_idx[i] for i in inner_tr]
    val_idx   = [dev_idx[i] for i in inner_va]

    s_tr, s_va, s_te = set(train_idx), set(val_idx), set(test_idx)
    assert s_tr.isdisjoint(s_va), "Solapamiento train/val detectado"
    assert s_tr.isdisjoint(s_te), "Solapamiento train/test detectado"
    assert s_va.isdisjoint(s_te), "Solapamiento val/test detectado"

    return train_idx, val_idx, test_idx


def _save_loss_curves(
    train_losses: List[float], val_losses: List[float],
    variant: str, ds_name: str, out_dir: Path
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train")
    ax.plot(epochs, val_losses,   label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title(f"{variant} — {ds_name}")
    ax.legend(); ax.grid(True); fig.tight_layout()
    base = out_dir / f"{ds_name}_{variant}_loss_curve"
    fig.savefig(base.with_suffix(".png"), dpi=150)
    plt.close(fig)
    pd.DataFrame({
        "epoch": list(epochs),
        "train_loss": train_losses,
        "val_loss":   val_losses,
    }).to_csv(base.with_suffix(".csv"), index=False)
    print(f"Curvas guardadas: {base}.png / .csv")


# ── LR per-variante (espacio de búsqueda manual calibrado) ───────────────────
_LR_CHOICES: Dict[str, list] = {
    "alstm":              [1e-3, 2e-3, 3e-3, 4e-3],
    "autoformer1d":       [4e-3, 5e-3, 6e-3, 7e-3],
    "cnnbilstmbigrudil":  [2e-3, 3e-3, 4e-3, 1e-2],
    "cnngru":             [1e-4, 5e-4, 1e-3, 4e-3],
    "cnnlstm":            [3e-3, 5e-3, 9e-3],
    "cnnlstmgru":         [1e-3, 2e-3, 4e-3, 5e-3],
    "deepresidualbilstm": [2e-3, 3e-3, 6e-3, 8e-3],
    "dilatedcnn":         [1e-3, 2e-3, 3e-3, 1e-2],
    "disjointcnn":        [1e-4, 3e-3, 4e-3],
    "disjointlstm":       [5e-3, 6e-3, 8e-3],
    "disjointresnet":     [5e-4, 1e-3, 3e-3, 6e-3],
    "esn":                [3e-3, 4e-3, 6e-3, 7e-3],
    "fcn":                [1e-3, 2e-3, 4e-3, 5e-3],
    "inceptiontime":      [3e-4, 7e-4, 1e-3, 1.3e-3],
    "indrnn":             [1e-4, 4e-4, 4e-3, 7e-3],
    "informer1d":         [6e-4, 1e-3, 7e-3, 8e-3],
    "lite":               [3e-3, 5e-3, 7e-3, 9e-3],
    "lstmfcn":            [5e-4, 1e-3, 2e-3, 4e-3],
    "mlp":                [4e-4, 2e-3, 8e-3],
    "multichanneldcnn":   [8e-4, 1e-3, 3e-3],
    "multiscalecnn":      [4e-5, 5e-4, 2e-3, 3e-3],
    "patchtst":           [4e-4, 2e-3, 5e-3],
    "peepholelstm":       [2e-3, 5e-3, 8e-3, 9e-3],
    "physformer":         [3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3],
    "qrnn":               [3e-4, 5e-4, 3e-3, 6e-3],
    "resnet":             [8e-4, 2e-3, 3e-3, 5e-3],
    "resnetcnn":          [3e-4, 9e-4, 1e-3, 2e-3],
    "rnn_gru":            [4e-5, 2e-3, 6e-3, 8e-3],
    "rnn_lstm":           [1e-3, 2e-3, 3e-3, 5e-3],
    "tcn":                [7e-4, 1e-3, 3e-3, 7e-3],
    "tfrcnn":             [2e-3, 3e-3],
    "timenet1d":          [5e-4, 1e-3, 3e-3, 5e-3],
    "tlgru":              [2e-3, 3e-3, 9e-3],
    "transformer1d":      [8e-4, 1e-3, 1.5e-3],
    "vit1d":              [5e-4, 1e-3, 2e-3, 5e-3],
    "vsgrunet":           [1e-3, 2e-3, 4e-3, 8e-3],
    "zhaocnn":            [5e-5, 2e-3, 7e-3],
    "oscnn":              [4e-5, 5e-4, 2e-3, 3e-3],
}
_LR_DEFAULT = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

# Familia por variante — controla patience y warmup
_MODEL_FAMILY: Dict[str, str] = {
    # CNN (14)
    "dilatedcnn": "cnn", "disjointcnn": "cnn", "disjointresnet": "cnn",
    "fcn": "cnn", "inceptiontime": "cnn", "lite": "cnn", "mlp": "cnn",
    "multichanneldcnn": "cnn", "multiscalecnn": "cnn", "resnet": "cnn",
    "resnetcnn": "cnn", "tcn": "cnn", "tfrcnn": "cnn", "zhaocnn": "cnn",
    # RNN (8)
    "esn": "rnn", "indrnn": "rnn", "peepholelstm": "rnn", "qrnn": "rnn",
    "rnn_gru": "rnn", "rnn_lstm": "rnn", "tlgru": "rnn", "vsgrunet": "rnn",
    # Hybrid (7)
    "cnnbilstmbigrudil": "hybrid", "cnnlstm": "hybrid", "cnngru": "hybrid",
    "cnnlstmgru": "hybrid", "deepresidualbilstm": "hybrid",
    "disjointlstm": "hybrid", "lstmfcn": "hybrid",
    # Attention (8)
    "alstm": "attention", "autoformer1d": "attention", "informer1d": "attention",
    "patchtst": "attention", "physformer": "attention", "timenet1d": "attention",
    "transformer1d": "attention", "vit1d": "attention",
}

# Config por familia: patience_final (entrenamiento), patience_hpo (trials HPO), warmup_ratio
_FAMILY_CFG: Dict[str, Dict] = {
    "cnn":       {"patience_final": 15, "patience_hpo":  7, "warmup_ratio": 0.05},
    "rnn":       {"patience_final": 20, "patience_hpo": 10, "warmup_ratio": 0.02},
    "hybrid":    {"patience_final": 20, "patience_hpo": 10, "warmup_ratio": 0.05},
    "attention": {"patience_final": 25, "patience_hpo": 12, "warmup_ratio": 0.10},
}


def _make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> optim.AdamW:
    """
    AdamW con dos grupos: sin weight_decay para BN/LN y biases (Loshchilov 2019).

    Excluye "combiner.*" (PhysMixFormer1D.combiner, los log-vars de
    UncertaintyWeightedCombiner) -- ese submódulo tiene su propio
    optimizador SGD dedicado (ver final_train/train_holdout), porque bajo
    AdamW compartido la normalización por segundo momento anula la
    diferenciación por escala que hace útil la ponderación por
    incertidumbre. No-op para el resto de arquitecturas (no tienen
    submódulo "combiner").
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("combiner."):
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return optim.AdamW(
        [
            {"params": decay,    "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def _clip_grad_norm_excluding_combiner(model: nn.Module, max_norm: float) -> None:
    """torch.nn.utils.clip_grad_norm_ que excluye "combiner.*" (ver
    _make_optimizer) -- el standalone validado no aplica grad-clip del
    modelo a los log-vars de incertidumbre (escalas distintas). No-op
    para el resto de arquitecturas."""
    params = [p for n, p in model.named_parameters() if not n.startswith("combiner.")]
    torch.nn.utils.clip_grad_norm_(params, max_norm)


# ═════════════════════════ Entrenamiento final ════════════════════════════════

def final_train(
    cfg:         Dict,
    ds_train:    Subset,
    ds_val:      Subset,
    ds_test:     Subset,
    n_total:     int,
    factory:     ModelFactory,
    ds_name:     str,
    metrics_log: Path,
    save_curves: bool = False,
    geometry=None,
) -> None:
    """
    Entrena el modelo con la mejor configuración HPO y evalúa en el test held-out.

    Schedule:
      Épocas 1..W  — LinearLR  (warmup: 0.1·lr → lr)
      Épocas W+1.. — CosineAnnealingLR  (lr → 1e-6)

    La mejora, el early stopping y el checkpoint usan exclusivamente ds_val.
    Las métricas finales reportadas usan exclusivamente ds_test.
    """
    cfg = _sanitize(cfg)
    set_seed(GLOBAL_SEED)

    n_tr, n_va, n_te = len(ds_train), len(ds_val), len(ds_test)
    print(f"\nSplit efectivo — train: {n_tr}  val: {n_va}  test: {n_te}  "
          f"({100*n_tr/n_total:.0f}% / {100*n_va/n_total:.0f}% / {100*n_te/n_total:.0f}%)")

    tr_loader = _make_loader(ds_train, int(cfg["batch_size"]), shuffle=True,
                             num_workers=2, seed=GLOBAL_SEED)
    va_loader = _make_loader(ds_val,   int(cfg["batch_size"]), shuffle=False,
                             num_workers=2, seed=GLOBAL_SEED)
    te_loader = _make_loader(ds_test,  int(cfg["batch_size"]), shuffle=False,
                             num_workers=2, seed=GLOBAL_SEED)

    params    = {k: v for k, v in cfg.items() if k not in _NONMODEL_KEYS}
    model     = factory.create(cfg["variant"], params).to(DEVICE)
    composite = getattr(model, "USES_COMPOSITE_LOSS", False)
    if composite and geometry is not None:
        model.attach_geometry(geometry)

    handles: list = []
    if SANITY_CHECKS:
        handles = install_nan_detector(model)

    try:
        if DEVICE == "cuda":
            model = torch.compile(model, mode="default")
    except Exception as exc:
        log.warning("torch.compile omitido: %s", exc)

    def _forward(xb):
        """Forward con fallback a eager si torch.compile falla en runtime.

        El wrapping de torch.compile() de arriba es perezoso y casi nunca
        falla en sí mismo; el fallo real ocurre en la primera invocación
        real del modelo (p.ej. resnet + AdaptiveMaxPool1d + Inductor bajo
        bfloat16, ver MIGRATION_REPORT.md). Si eso pasa, se reintenta una
        vez en modo eager y el resto del entrenamiento sigue sin compilar.
        """
        nonlocal model
        try:
            return model(xb)
        except Exception as exc:
            eager = base_module(model)
            if eager is model:
                raise
            log.warning(
                "torch.compile falló en runtime (%s); reintentando en modo eager.",
                exc,
            )
            model = eager
            return model(xb)

    amp_      = amp_config(DEVICE)
    scaler    = GradScaler(device=DEVICE, enabled=amp_["use_scaler"])
    loss_fn   = nn.MSELoss()

    family        = _MODEL_FAMILY.get(cfg["variant"], "cnn")
    fcfg          = _FAMILY_CFG[family]
    optimizer     = _make_optimizer(model, cfg["lr"], cfg["weight_decay"])

    # Optimizador SGD dedicado para los log-vars de incertidumbre de
    # PhysMixFormer1D -- separado del AdamW del modelo (ver
    # UncertaintyWeightedCombiner en PhysMixFormer1D.py: bajo AdamW
    # compartido la normalización por segundo momento anula la
    # diferenciación por escala que hace útil la ponderación).
    uncertainty_opt = torch.optim.SGD(model.uncertainty_parameters(), lr=0.02) \
                       if composite else None

    max_epochs    = int(cfg.get("epochs", 50))
    warmup_epochs = max(1, round(max_epochs * fcfg["warmup_ratio"]))

    if composite:
        # ReduceLROnPlateau -- mismo schedule que
        # physmixformer_standalone_v3.py::run_basic_training (factor=0.5,
        # patience=5, min_lr=5e-5), en vez del warmup+cosine genérico del
        # resto del pipeline. Adaptativo a val_loss real, no a un
        # cronograma fijo de épocas -- así entrena en igualdad de
        # condiciones con el standalone.
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=5e-5
        )
    else:
        # Warmup lineal + CosineAnnealingLR — estándar moderno para convergencia estable
        warmup    = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine    = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, max_epochs - warmup_epochs), eta_min=1e-6
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )

    # EarlyStopping sobre val_loss; min_delta=1e-4 evita paradas por ruido numérico
    es = EarlyStopping(patience=fcfg["patience_final"], min_delta=1e-4, verbose=False)

    best_state: dict | None = None
    best_val_loss = float("inf")
    best_epoch    = 0
    train_losses: List[float] = []
    val_losses:   List[float] = []

    print(f"\n{'Ep':>4}  {'Train Loss':>11}  {'Val Loss':>11}  {'LR':>9}")
    print("─" * 43)

    for epoch in range(max_epochs):
        # ── Fase train ───────────────────────────────────────────
        if composite:
            model.anneal_epoch(epoch + 1, max_epochs)
        model.train()
        tr_sum, tr_n = 0.0, 0
        for i, (xb, yb, _) in enumerate(tr_loader):
            if SANITY_CHECKS:
                assert_no_nan_batch(xb, yb, tag=f"tr_ep{epoch}_b{i}")
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if uncertainty_opt is not None:
                uncertainty_opt.zero_grad(set_to_none=True)

            if amp_["enabled"]:
                with autocast(device_type=DEVICE, dtype=amp_["dtype"]):
                    if composite:
                        pred, Pi, _ = _forward(xb)
                        loss = model.compute_loss(pred, yb, xb, Pi=Pi)
                    else:
                        pred = _forward(xb)
                        pred, yb_c = _clip_to_min_len(pred, yb)
                        loss = loss_fn(pred, yb_c)
            else:
                if composite:
                    pred, Pi, _ = _forward(xb)
                    loss = model.compute_loss(pred, yb, xb, Pi=Pi)
                else:
                    pred = _forward(xb)
                    pred, yb_c = _clip_to_min_len(pred, yb)
                    loss = loss_fn(pred, yb_c)

            if amp_["use_scaler"]:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                _clip_grad_norm_excluding_combiner(model, 1.0)
                scaler.step(optimizer)
                if uncertainty_opt is not None:
                    scaler.unscale_(uncertainty_opt)
                    scaler.step(uncertainty_opt)
                scaler.update()
            else:
                loss.backward()
                _clip_grad_norm_excluding_combiner(model, 1.0)
                optimizer.step()
                if uncertainty_opt is not None:
                    uncertainty_opt.step()

            tr_sum += loss.item() * xb.size(0)
            tr_n   += xb.size(0)

        train_loss = tr_sum / max(tr_n, 1)
        if save_curves:
            train_losses.append(train_loss)

        # ── Fase validación ──────────────────────────────────────
        model.eval()
        va_sum, va_n = 0.0, 0
        with torch.inference_mode():
            for i, (xb, yb, _) in enumerate(va_loader):
                if SANITY_CHECKS:
                    assert_no_nan_batch(xb, yb, tag=f"va_ep{epoch}_b{i}")
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                if composite:
                    out, Pi, _ = _forward(xb)
                    va_sum += model.compute_loss(out, yb, xb, Pi=Pi).item() * xb.size(0)
                else:
                    out = _forward(xb)
                    out, yb_c = _clip_to_min_len(out, yb)
                    va_sum += loss_fn(out, yb_c).item() * xb.size(0)
                va_n   += xb.size(0)
        val_loss = va_sum / max(va_n, 1)
        if save_curves:
            val_losses.append(val_loss)

        # Actualizar schedule una vez por época (epoch-based)
        if composite:
            scheduler.step(val_loss)  # ReduceLROnPlateau necesita la métrica
        else:
            scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        # Checkpoint del mejor estado según val_loss
        mark = ""
        if val_loss + 1e-4 < best_val_loss:
            src = base_module(model)
            best_state     = {k: v.detach().cpu() for k, v in src.state_dict().items()}
            best_val_loss  = val_loss
            best_epoch     = epoch + 1
            mark = " ✓"

        print(f"{epoch+1:4d}  {train_loss:11.5f}  {val_loss:11.5f}  {lr_now:9.2e}{mark}")

        if es(val_loss):
            print(f"\n  Early stopping ep {epoch+1}  "
                  f"(mejor val={best_val_loss:.5f} @ ep{best_epoch})")
            break

    print("─" * 43)

    # Restaurar mejor estado (fallback: estado final si nunca mejoró)
    if best_state is None:
        log.warning("El modelo no mejoró. Se usa el estado final como checkpoint.")
        src = base_module(model)
        best_state = {k: v.detach().cpu() for k, v in src.state_dict().items()}
    base_module(model).load_state_dict(best_state, strict=True)

    # ── Guardar checkpoint y configuración ───────────────────────
    ckpt_dir  = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"{ds_name}_{cfg['variant']}_best.pt"
    torch.save({
        "model_state": best_state,
        "val_loss":    best_val_loss,
        "best_epoch":  best_epoch,
    }, ckpt_path)
    with open(ckpt_dir / f"{ds_name}_{cfg['variant']}_config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"\nCheckpoint: {ckpt_path}")

    # Curvas opcionales (train + val; el test no se incluye — evita mirar el futuro)
    if save_curves:
        _save_loss_curves(train_losses, val_losses, cfg["variant"], ds_name,
                          PROJECT_ROOT / "loss_curves")

    # ── Métricas finales — TEST held-out ─────────────────────────
    # El test set no participó en ningún paso anterior (ni HPO ni early stopping).
    preds_list, targs_list = [], []
    model.eval()
    with torch.inference_mode():
        for xb, yb, _ in te_loader:
            xb  = xb.to(DEVICE, non_blocking=True)
            out = _forward(xb)
            if composite:
                out = out[0]
            out = out.cpu()
            out, yb_c = _clip_to_min_len(out, yb)
            preds_list.append(out)
            targs_list.append(yb_c)
    preds = torch.cat(preds_list)
    targs = torch.cat(targs_list)
    assert len(preds) == n_te, \
        f"Métricas incompletas: {len(preds)} predicciones vs {n_te} muestras test."

    rpt = ECGMetricsCalculator.complete_report(targs, preds)
    print("\n=== Métricas finales — TEST held-out (20 %) ===")
    for k, v in rpt.items():
        if isinstance(v, tuple):
            print(f"  {k:24s}: {v[0]:.5f} ± {v[1]:.5f}")
        else:
            print(f"  {k:24s}: {v:.5f}")

    # ── Perfil del modelo ─────────────────────────────────────────
    sample        = next(iter(te_loader))[0][:1].to(DEVICE, non_blocking=True)
    profile_model = factory.create(cfg["variant"], params).to(DEVICE)
    if composite and geometry is not None:
        # best_state trae las claves de los buffers/parámetros de geometría
        # (P_torso, P_epi, coupling.logits, etc.) -- sin adjuntarla aquí
        # también, load_state_dict(strict=True) falla por claves inesperadas.
        profile_model.attach_geometry(geometry)
    profile_model.load_state_dict(best_state, strict=True)
    n_p, lat, flops, macs = compute_model_stats(profile_model, sample, device=DEVICE)
    print(f"\n  Parámetros           : {n_p:,}")
    print(f"  Latencia inferencia  : {lat:.2f} ms")
    if math.isfinite(flops) and math.isfinite(macs):
        print(f"  FLOPs                : {flops/1e9:.3f} G")
        print(f"  MACs                 : {macs/1e6:.3f} M")
    else:
        print("  FLOPs/MACs           : no disponibles (THOP falló)")

    # ── Registro en Excel ─────────────────────────────────────────
    log_metrics_to_excel(
        metrics_log, rpt, n_p, lat, flops, macs,
        best_trial_str=cfg["_best_trial_str"],
        method=cfg["variant"],
        dataset=ds_name,
        extra=dict(
            n_train=n_tr, n_val=n_va, n_test=n_te,
            train_frac=round(n_tr / n_total, 3),
            val_frac=round(n_va / n_total, 3),
            test_frac=round(n_te / n_total, 3),
            best_val_loss=round(best_val_loss, 6),
            best_epoch=best_epoch,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
        ),
    )

    for h in handles:
        try: h.remove()
        except Exception: pass
    free_all(profile_model, model)


# ═══════════════════════════ Pipeline HPO ════════════════════════════════════

def run_pipeline(
    variant:     str,
    mat_path:    Path,
    x:           torch.Tensor,
    y:           torch.Tensor,
    labels:      torch.Tensor,
    factory:     ModelFactory,
    data_cfg:    DataConfig,
    direction:   str,
    metrics_log: Path,
    save_curves: bool,
    num_samples: int,
    max_epochs:  int = 50,
) -> None:
    assert variant in MODEL_REGISTRY, f"'{variant}' no en MODEL_REGISTRY"
    assert variant in SPACE_REGISTRY, f"'{variant}' no en SPACE_REGISTRY"

    ds_name = mat_path.stem.lower()
    N       = len(labels)

    print(f"\n{'='*60}")
    print(f"  Pipeline AutoML — {variant.upper()}  |  dataset={ds_name}  |  N={N}")
    print(f"  Split 60/20/20 · BOHB {num_samples} trials · seed={GLOBAL_SEED}")
    print(f"  Dispositivo: {DEVICE.upper()}")
    print(f"{'='*60}")

    # Split reproducible — mismo resultado siempre dado (labels, GLOBAL_SEED)
    train_idx, val_idx, test_idx = _three_way_split(labels, seed=GLOBAL_SEED)

    full_ds  = TensorDataset(x, y, labels)
    ds_train = Subset(full_ds, train_idx)
    ds_val   = Subset(full_ds, val_idx)
    ds_test  = Subset(full_ds, test_idx)

    # Publicar datos una sola vez en el object store de Ray para todos los trials
    # (evita serializar tensores grandes N veces)
    ds_ref        = ray.put(full_ds)
    train_idx_ref = ray.put(train_idx)
    val_idx_ref   = ray.put(val_idx)

    # Geometría de malla torso/corazón (PhysMixFormer1D/"physformer"): se
    # computa una sola vez por dataset (no por trial) y se comparte vía el
    # object store, igual que ds_ref/train_idx_ref/val_idx_ref. None si el
    # dataset no tiene geometría confiable -- ver data/geometry.py.
    geometry     = load_geometry(mat_path, data_cfg.input_channels, data_cfg.output_channels) \
                   if variant == "physformer" else None
    geometry_ref = ray.put(geometry)

    # ── Espacio de búsqueda BOHB ──────────────────────────────────
    spec = SPACE_REGISTRY[variant]
    spec = spec() if callable(spec) else spec

    search_space = {
        **space_to_tune(spec),
        "lr":              tune.choice(_LR_CHOICES.get(variant, _LR_DEFAULT)),
        # weight_decay=0 válido para arqs. con BN / Dropout; desacoplado via AdamW
        "weight_decay":    tune.choice([0, 1e-6, 1e-5, 1e-4]),
        # batch_size hasta 32: seguro en 4 GB VRAM con series 1D
        "batch_size":      tune.choice([8, 16, 32]),
        "epochs":          max_epochs,   # budget máximo por trial para HyperBand
        "variant":         variant,
        "mat_path":        str(mat_path),
        "time_steps":      data_cfg.time_steps,
        "input_channels":  data_cfg.input_channels,
        "output_channels": data_cfg.output_channels,
        "direction":       direction,
    }

    def train_holdout(cfg: Dict):
        """
        Función objetivo para BOHB.
        Sin LR scheduler: la señal de val_loss debe reflejar la configuración pura,
        no una interacción con el schedule (que varía según epochs restantes).
        """
        cfg       = _sanitize(cfg)
        set_seed(GLOBAL_SEED)   # reproducibilidad de cada trial

        # Recuperar datos del object store (no deserializa el tensor)
        full_ds_  = ray.get(ds_ref)
        tr_idx    = ray.get(train_idx_ref)
        va_idx    = ray.get(val_idx_ref)
        ds_tr     = Subset(full_ds_, tr_idx)
        ds_va     = Subset(full_ds_, va_idx)

        tr_loader = _make_loader(ds_tr, int(cfg["batch_size"]), shuffle=True,
                                 num_workers=0, seed=GLOBAL_SEED)
        va_loader = _make_loader(ds_va, int(cfg["batch_size"]), shuffle=False,
                                 num_workers=0, seed=GLOBAL_SEED)

        params    = {k: v for k, v in cfg.items() if k not in _NONMODEL_KEYS}
        model     = factory.create(cfg["variant"], params).to(DEVICE)
        composite = getattr(model, "USES_COMPOSITE_LOSS", False)
        if composite:
            geom = ray.get(geometry_ref)
            if geom is not None:
                model.attach_geometry(geom)
        amp_      = amp_config(DEVICE)
        scaler    = GradScaler(device=DEVICE, enabled=amp_["use_scaler"])
        loss_fn   = nn.MSELoss()
        optimizer = _make_optimizer(model, cfg["lr"], cfg["weight_decay"])
        uncertainty_opt = torch.optim.SGD(model.uncertainty_parameters(), lr=0.02) \
                           if composite else None
        family    = _MODEL_FAMILY.get(cfg["variant"], "cnn")
        es        = EarlyStopping(patience=_FAMILY_CFG[family]["patience_hpo"],
                                  min_delta=1e-4, verbose=False)

        for epoch in range(cfg["epochs"]):
            # ── train ──
            if composite:
                model.anneal_epoch(epoch + 1, cfg["epochs"])
            model.train()
            for xb, yb, _ in tr_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if uncertainty_opt is not None:
                    uncertainty_opt.zero_grad(set_to_none=True)

                if amp_["enabled"]:
                    with autocast(device_type=DEVICE, dtype=amp_["dtype"]):
                        if composite:
                            pred, Pi, _ = model(xb)
                            loss = model.compute_loss(pred, yb, xb, Pi=Pi)
                        else:
                            pred = model(xb)
                            pred, yb_c = _clip_to_min_len(pred, yb)
                            loss = loss_fn(pred, yb_c)
                else:
                    if composite:
                        pred, Pi, _ = model(xb)
                        loss = model.compute_loss(pred, yb, xb, Pi=Pi)
                    else:
                        pred = model(xb)
                        pred, yb_c = _clip_to_min_len(pred, yb)
                        loss = loss_fn(pred, yb_c)

                if amp_["use_scaler"]:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    _clip_grad_norm_excluding_combiner(model, 1.0)
                    scaler.step(optimizer)
                    if uncertainty_opt is not None:
                        scaler.unscale_(uncertainty_opt)
                        scaler.step(uncertainty_opt)
                    scaler.update()
                else:
                    loss.backward()
                    _clip_grad_norm_excluding_combiner(model, 1.0)
                    optimizer.step()
                    if uncertainty_opt is not None:
                        uncertainty_opt.step()

            # ── val ──
            model.eval()
            va_sum, va_n = 0.0, 0
            with torch.inference_mode():
                for xb, yb, _ in va_loader:
                    xb = xb.to(DEVICE, non_blocking=True)
                    yb = yb.to(DEVICE, non_blocking=True)
                    if composite:
                        out, Pi, _ = model(xb)
                        va_sum += model.compute_loss(out, yb, xb, Pi=Pi).item() * xb.size(0)
                    else:
                        out = model(xb)
                        out, yb_c = _clip_to_min_len(out, yb)
                        va_sum += loss_fn(out, yb_c).item() * xb.size(0)
                    va_n   += xb.size(0)
            val_loss = va_sum / max(va_n, 1)

            # Reportar a BOHB — HyperBand puede cancelar el trial aquí
            tune.report({"val_loss": val_loss, "epoch": epoch + 1})

            if es(val_loss):
                break

        free_all(model, optimizer, scaler, tr_loader, va_loader)

    # ── BOHB + HyperBand ──────────────────────────────────────────
    # reduction_factor=3: estándar SHA (Li et al. 2018) — rungs en {50, 17, 6, 2}
    analysis = tune.run(
        train_holdout,
        resources_per_trial={"cpu": 1, "gpu": 1},
        config=search_space,
        num_samples=num_samples,
        search_alg=TuneBOHB(metric="val_loss", mode="min", seed=GLOBAL_SEED),
        scheduler=HyperBandForBOHB(
            metric="val_loss", mode="min", max_t=max_epochs, reduction_factor=3
        ),
        name=f"{variant}_bohb",
        verbose=1,
        reuse_actors=False,
    )

    best = analysis.get_best_config("val_loss", "min")
    if best is None:
        raise RuntimeError(
            f"[{variant}] Todos los trials fallaron. "
            f"Revisa ~/ray_results/{variant}_bohb/"
        )
    best = _sanitize(best)
    best["_best_trial_str"] = ", ".join(f"{k}={v}" for k, v in best.items())
    print(f"\n>>> Mejor config HPO ({variant}): {best['_best_trial_str']}")

    final_train(
        cfg=best,
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_test,
        n_total=N,
        factory=factory,
        ds_name=ds_name,
        metrics_log=metrics_log,
        save_curves=save_curves,
        geometry=geometry,
    )


# ══════════════════════════════ MAIN ═════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    p = argparse.ArgumentParser(
        description="AutoML pipeline: BOHB HPO + split 60/20/20 + warmup-cosine + test held-out"
    )
    p.add_argument("--dataset", required=True,
                   choices=["kitpvc", "ckitpvc", "auckland", "bordeaux", "utah2002"])
    p.add_argument("--method",  required=True,
                   help="Clave en MODEL_REGISTRY (ej. 'mlp', 'resnet', 'cnngru')")
    p.add_argument("--direction", choices=["ecg2emg", "emg2ecg"], default="ecg2emg")
    p.add_argument("--save-curves",  action="store_true",
                   help="Guarda curvas train/val en loss_curves/ (PNG + CSV)")
    p.add_argument("--num-samples",  type=int, default=15,
                   help="Número de trials HPO (default: 15)")
    p.add_argument("--max-epochs",   type=int, default=50,
                   help="Épocas máximas por trial y entrenamiento final (default: 50)")
    args = p.parse_args()

    _datasets = {
        "kitpvc":   DATA_ROOT / "kitpvc"  / "kitpvc.mat",
        "ckitpvc":  DATA_ROOT / "ckitpvc" / "raw" / "ckitpvc.mat",
        "auckland": DATA_ROOT / "auckland"/ "raw" / "auckland.mat",
        "bordeaux": DATA_ROOT / "bordeaux"/ "raw" / "bordeaux.mat",
        "utah2002": DATA_ROOT / "utah2002"/ "raw" / "utah2002.mat",
    }
    mat = _datasets[args.dataset]
    if not mat.exists():
        raise FileNotFoundError(f"Dataset '{args.dataset}' no encontrado en {mat}")

    # ── Carga única del dataset ───────────────────────────────────
    PIPELINE_TIME_STEPS = 256
    dm0 = ECGDataModule(
        str(mat), batch_size=1, time_steps=PIPELINE_TIME_STEPS,
        num_workers=2, train_frac=1.0, direction=args.direction,
    )
    dm0.prepare_data()

    if dm0.labels is None:
        raise ValueError("No se pudieron cargar las etiquetas del dataset.")

    x, y, labels = dm0.x, dm0.y, dm0.labels

    # "physformer" (PhysMixFormer1D): filtro Butterworth aplicado a la señal
    # cruda antes del PAA (ver data/filtering.py) -- el resto de las 36
    # arquitecturas siguen sin filtrar, sin cambios de comportamiento.
    if args.method.lower() == "physformer" and args.dataset in DATASET_FS:
        ecg_f, emg_f, labels_f = load_filtered(mat, PIPELINE_TIME_STEPS, args.dataset)
        if args.direction.lower() == "ecg2emg":
            x, y = ecg_f, emg_f
        else:
            x, y = emg_f, ecg_f
        labels = labels_f
        print(f"[physformer] filtro Butterworth aplicado (fs={DATASET_FS[args.dataset]} Hz, "
              f"lp_cut={DATASET_LP_CUT[args.dataset]} Hz) antes del PAA.")

    N = len(labels)
    unique, counts = torch.unique(labels, return_counts=True)
    print(f"Dataset cargado: {N} muestras  |  "
          f"Clases: {dict(zip(unique.tolist(), counts.tolist()))}")
    print(f"Forma de entrada: {tuple(x.shape)}  →  salida: {tuple(y.shape)}")

    data_cfg = DataConfig(
        time_steps=x.shape[2],
        input_channels=x.shape[1],
        output_channels=y.shape[1],
    )
    factory = ModelFactory(data_cfg)

    # ── Ray ───────────────────────────────────────────────────────
    ray.shutdown()
    ray.init(
        num_cpus=6,
        num_gpus=1,
        object_store_memory=bytes_from_shm(target_frac=0.75),
        _system_config=spilling_config() or None,
        log_to_driver=False,
    )

    run_pipeline(
        variant=args.method.lower(),
        mat_path=mat,
        x=x, y=y, labels=labels,
        factory=factory,
        data_cfg=data_cfg,
        direction=args.direction,
        metrics_log=PROJECT_ROOT / "metrics_log.xlsx",
        save_curves=args.save_curves,
        num_samples=args.num_samples,
        max_epochs=args.max_epochs,
    )
    ray.shutdown()
