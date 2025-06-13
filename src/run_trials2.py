#!/usr/bin/env python
# hyperparam_search.py — BOHB + 5-fold CV con weight_decay y estadísticas de modelo
# --------------------------------------------------------------------------------
"""
• Inner 5-fold CV en GPU (AMP + liberación VRAM)
• BOHB pruning (HyperBand reduction_factor=3)
• Entrenamiento final con early-stopping (patience = 10)
• Semilla global fija = 42

Ejecución:
    python hyperparam_search.py --dataset kitpvc --method mlp
"""
import os
import sys
#os.environ["TUNE_DISABLE_DFATAL_SIGNAL_HANDLERS"] = "1"  # evita bloqueos CUDA en Ray

import gc
import random
import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

import ray
from ray import tune, train
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB

from thop import profile  # para FLOPs/MACs

# ────────────────────────── Rutas proyecto ───────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT    = PROJECT_ROOT / "data"
SRC_ROOT     = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

# ────────────────────────── Globales ──────────────────────────────
GLOBAL_SEED = 42
TRAIN_FRAC  = 0.70
KFOLD       = 5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP     = DEVICE == "cuda"

# ────────────────────────── Imports del repo ──────────────────────
from config import DataConfig
from models.factory import ModelFactory
from models.register import MODEL_REGISTRY, SPACE_REGISTRY
from hpo.adapters import space_to_tune
from data.datamodule import ECGDataModule
from utils.metrics import ECGMetricsCalculator

# ────────────────────────── AMP Utils ────────────────────────────
from torch.amp import autocast, GradScaler

# ────────────────────────── Excel logging ─────────────────────────
METRICS_LOG = PROJECT_ROOT / "metrics_log.xlsx"

import pandas as pd
from pathlib import Path

METRICS_LOG = PROJECT_ROOT / "metrics_log.xlsx"

def log_metrics_to_excel(
    rpt: Dict,
    n_p: int,
    lat: float,
    flops: float,
    macs: float,
    best_trial_str: str,
    method: str
):
    """
    Añade una nueva fila al fichero Excel con:
      - método (method)
      - métricas rpt (mean/std) convertidas a float
      - estadísticas de modelo (n_p, lat, flops, macs)
      - texto del best_trial
      - timestamp
    """
    # Helper para extraer un valor python de un posible tensor
    def to_float(x):
        return float(x.item()) if isinstance(x, torch.Tensor) else float(x)

    # Preparar la fila con 'method' como primera columna
    row = {
        "method": method,
        # medias
        **{f"{k}_mean": to_float(v[0]) if isinstance(v, tuple) else to_float(v)
           for k, v in rpt.items()},
        # stds (solo para tuplas)
        **{f"{k}_std": to_float(v[1])
           for k, v in rpt.items() if isinstance(v, tuple)},
        "num_params":          n_p,
        "inference_latency_ms":lat,
        "FLOPs":               flops,
        "MACs":                macs,
        "best_trial":          best_trial_str,
        "timestamp":           pd.Timestamp.now()
    }

    if METRICS_LOG.exists():
        df = pd.read_excel(METRICS_LOG)
        # Asegurar todas las columnas
        for c in row:
            if c not in df.columns:
                df[c] = pd.NA
        df.loc[len(df)] = row
    else:
        df = pd.DataFrame([row])

    df.to_excel(METRICS_LOG, index=False)


# ────────────────────────── Utilidades ────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_loader(ds, bs, shuffle):
    return DataLoader(
        ds, batch_size=bs, shuffle=shuffle,
        num_workers=2, pin_memory=True, persistent_workers=True
    )

def free_all(*objs):
    for o in objs:
        try: del o
        except: pass
    torch.cuda.empty_cache()
    gc.collect()

def compute_model_stats(model: nn.Module, sample: torch.Tensor):
    """Devuelve num_params, latencia (ms), FLOPs y MACs del modelo."""
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    macs, _   = profile(model, inputs=(sample,), verbose=False)
    flops     = 2 * macs

    # Medir latencia
    with torch.no_grad():
        for _ in range(10):
            model(sample)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(50):
            model(sample)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.time() - t0) / 50 * 1000.0

    return num_params, latency_ms, flops, macs


# ────────────────────────── Entrenamiento final ───────────────────
def final_train(cfg: Dict, factory: ModelFactory):
    set_seed(GLOBAL_SEED)
    dm = ECGDataModule(
        mat_path=cfg["mat_path"], batch_size=cfg["batch_size"],
        time_steps=cfg["time_steps"], num_workers=2,
        train_frac=TRAIN_FRAC,
    )
    dm.prepare_data(); dm.setup()
    tr_loader, va_loader = dm.train_dataloader(), dm.val_dataloader()

    params = {k: v for k, v in cfg.items()
              if k not in {"batch_size","lr","weight_decay",
                           "epochs","variant","mat_path",
                           "time_steps","input_channels","output_channels"}}
    model  = factory.create(cfg["variant"], params).to(DEVICE)
    optim_ = optim.Adam(model.parameters(),
                        lr=cfg["lr"],
                        weight_decay=cfg["weight_decay"])
    loss_fn = nn.MSELoss()
    scaler  = GradScaler("cuda", enabled=USE_AMP)

    # Early-stopping
    best_loss, stall, patience, best_state = float("inf"), 0, 20, None
    for _ in range(cfg["epochs"]):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim_.zero_grad()
            with autocast(device_type="cuda", enabled=USE_AMP):
                loss = loss_fn(model(xb), yb)
            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.step(optim_); scaler.update()
            else:
                loss.backward(); optim_.step()

        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                tot += loss_fn(model(xb), yb).item() * xb.size(0)
                n   += xb.size(0)
        val_loss = tot / n

        if val_loss + 1e-6 < best_loss:
            best_loss, best_state, stall = val_loss, model.state_dict(), 0
        else:
            stall += 1
            if stall >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    # Métricas clásicas
    preds, targs = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in va_loader:
            preds.append(model(xb.to(DEVICE)).cpu())
            targs.append(yb)
    rpt = ECGMetricsCalculator.complete_report(
        torch.cat(targs), torch.cat(preds)
    )
    print("\n=== Métricas finales (70/30) ===")
    for k, v in rpt.items():
        if isinstance(v, tuple):
            print(f"{k}: mean={v[0]:.4f}, std={v[1]:.4f}")
        else:
            print(f"{k}: {v:.4f}")

    # Estadísticas del modelo
    sample = next(iter(va_loader))[0][:1].to(DEVICE)
    n_p, lat, flops, macs = compute_model_stats(model, sample)
    print(f"\nnum_params           : {n_p:,}")
    print(f"inference_latency_ms : {lat:.2f} ms")
    print(f"FLOPs                : {flops/1e9:.3f} GFLOPs")
    print(f"MACs                 : {macs/1e6:.3f} MMACs")

    log_metrics_to_excel(
        rpt,
        n_p,
        lat,
        flops,
        macs,
        best_trial_str=cfg["_best_trial_str"],
        method=cfg["variant"]
    )
    free_all(model, optim_, scaler, tr_loader, va_loader)


# ────────────────────────── Pipeline HPO + CV ─────────────────────
def run_pipeline(variant: str, mat: Path, factory: ModelFactory):
    print(f"\n=== Pipeline HPO para ‘{variant}’ ===")
    assert variant in MODEL_REGISTRY, f"‘{variant}’ no registrado"

    spec = SPACE_REGISTRY[variant]
    spec = spec() if callable(spec) else spec

    search_space = {
        **space_to_tune(spec),
        "lr":            tune.loguniform(1e-4, 1e-2),
        "weight_decay":  tune.loguniform(1e-6, 1e-3),
        "batch_size":    tune.choice([8, 16, 32]),
        "epochs":        20,
        "variant":       variant,
        "mat_path":      str(mat),
        "time_steps":    data_cfg.time_steps,
        "input_channels":data_cfg.input_channels,
        "output_channels":data_cfg.output_channels,
    }

    def train_cv(cfg: Dict):
        set_seed(GLOBAL_SEED)
        dm = ECGDataModule(
            mat_path=cfg["mat_path"], batch_size=cfg["batch_size"],
            time_steps=cfg["time_steps"], num_workers=2, train_frac=1.0
        )
        dm.prepare_data(); dm.setup()
        full_ds = dm.train_dataloader().dataset

        kf = KFold(n_splits=KFOLD, shuffle=True, random_state=GLOBAL_SEED)
        losses: List[float] = []

        for tr_idx, va_idx in kf.split(range(len(full_ds))):
            tr_loader = build_loader(Subset(full_ds, tr_idx), cfg["batch_size"], True)
            va_loader = build_loader(Subset(full_ds, va_idx), cfg["batch_size"], False)

            params = {k: v for k, v in cfg.items()
                      if k not in {"batch_size","lr","weight_decay",
                                   "epochs","variant","mat_path",
                                   "time_steps","input_channels","output_channels"}}
            model  = factory.create(cfg["variant"], params).to(DEVICE)
            optim_ = optim.Adam(model.parameters(),
                                lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            loss_fn = nn.MSELoss()
            scaler  = GradScaler("cuda", enabled=USE_AMP)

            for _ in range(cfg["epochs"]):
                model.train()
                for xb, yb in tr_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    optim_.zero_grad()
                    with autocast(device_type="cuda", enabled=USE_AMP):
                        loss = loss_fn(model(xb), yb)
                    if USE_AMP:
                        scaler.scale(loss).backward()
                        scaler.step(optim_); scaler.update()
                    else:
                        loss.backward(); optim_.step()

            model.eval()
            tot, n = 0.0, 0
            with torch.no_grad():
                for xb, yb in va_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    tot += loss_fn(model(xb), yb).item() * xb.size(0)
                    n   += xb.size(0)
            losses.append(tot / n)

            # report intermedio a HyperBand
            train.report({"val_loss": float(np.mean(losses))})
            free_all(model, optim_, scaler, tr_loader, va_loader)

    analysis = tune.run(
        train_cv,
        resources_per_trial={"cpu": 2, "gpu": 1},
        config=search_space,
        num_samples=50,
        search_alg=TuneBOHB(metric="val_loss", mode="min", seed=GLOBAL_SEED),
        scheduler=HyperBandForBOHB(metric="val_loss", mode="min",
                                   max_t=20, reduction_factor=2),
        local_dir=str(PROJECT_ROOT/"ray_results"),
        name=f"{variant}_bohb_k5",
        verbose=1,
    )

    # 1) Obtenemos la mejor configuración
    best = analysis.get_best_config("val_loss", "min")

    # 2) Formateamos la cadena del best trial
    best_str = ", ".join(f"{k}={v}" for k, v in best.items())

    print(f"\n>>> [{variant}] mejor config: {best_str}")

    # 3) Inyectamos ese texto en el dict para pasarlo a final_train
    best["_best_trial_str"] = best_str

    # 4) Llamamos a final_train con el nuevo campo
    final_train(best, factory)

# ───────────────────────────── MAIN ───────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["kitpvc","auckland","bordeux", "utah2002"])
    p.add_argument("--method",  required=True,
                   help="Clave en MODEL_REGISTRY (ej. 'mlp')")
    args = p.parse_args()

    # Selección de .mat
    if args.dataset == "kitpvc":
        mat = DATA_ROOT/"kitpvc"/"raw"/"kitpvc.mat"
    elif args.dataset == "auckland":
        mat = DATA_ROOT/"auckland"/"raw"/"auckland.mat"
    elif args.dataset == "bordeux":
        mat = DATA_ROOT/"bordeux"/"raw"/"bordeux.mat"
    elif args.dataset == "utah2002":
        mat = DATA_ROOT/"utah2002"/"raw"/"utah2002.mat"
    else:
        raise ValueError("Dataset desconocido")
    assert mat.exists(), f"No encontré {mat}"

    # Inferir shapes de un batch
    dm0 = ECGDataModule(str(mat), batch_size=1,
                        time_steps=256, num_workers=2, train_frac=1.0)
    dm0.prepare_data(); dm0.setup()
    xb0, yb0 = next(iter(dm0.train_dataloader()))
    real_ts, real_ic, real_oc = xb0.shape[2], xb0.shape[1], yb0.shape[1]

    data_cfg = DataConfig(
        time_steps=real_ts,
        input_channels=real_ic,
        output_channels=real_oc,
    )
    factory = ModelFactory(data_cfg)

    ray.init(num_cpus=6, num_gpus=1, log_to_driver=False)
    run_pipeline(args.method.lower(), mat, factory)
    ray.shutdown()
