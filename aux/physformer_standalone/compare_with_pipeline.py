# -*- coding: utf-8 -*-
"""
Compara las corridas del PhysFormer standalone (physformer_standalone.py,
réplica fiel de physformer1.0.ipynb: backbone + PhysicsMixer + preprocesado
Butterworth/Z-score) contra las filas method=="physformer" que ya existen en
el pipeline comparativo (src/metrics_log.xlsx, generadas por src/run_bohb.py).

Nota importante: RMSE NO es directamente comparable entre ambas corridas.
El pipeline integrado (src/data/preprocessing.py + datamodule.py) no aplica
el filtrado Butterworth ni el Z-score por canal que sí aplica el notebook /
script standalone, así que las señales de entrada están en escalas
distintas. Esto es una divergencia de preprocesado ya documentada, no un
bug — el pipeline general es intencionalmente agnóstico al modelo para
poder comparar las 37 arquitecturas en igualdad de condiciones.
PCC (correlación de Pearson) sí es comparable porque es invariante a escala.

Uso:
    python compare_with_pipeline.py [--dataset auckland]
"""
import argparse
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
STANDALONE_CSV = Path(__file__).resolve().parent / "outputs" / "metrics_standalone.csv"
PIPELINE_XLSX = REPO_ROOT / "src" / "metrics_log.xlsx"


def load_standalone(dataset: str | None, direction: str) -> pd.DataFrame:
    if not STANDALONE_CSV.exists():
        raise FileNotFoundError(
            f"No hay corridas standalone todavía: {STANDALONE_CSV} no existe. "
            f"Corré primero physformer_standalone.py."
        )
    df = pd.read_csv(STANDALONE_CSV)
    if "direction" in df.columns:
        df = df[df["direction"] == direction]
    if dataset:
        df = df[df["dataset"] == dataset]
    return df


def load_pipeline(dataset: str | None, direction: str) -> pd.DataFrame:
    if not PIPELINE_XLSX.exists():
        raise FileNotFoundError(f"No existe {PIPELINE_XLSX}")
    df = pd.read_excel(PIPELINE_XLSX)
    df = df[df["method"] == "physformer"]
    if dataset:
        df = df[df["dataset"] == dataset]
    # la dirección queda embebida en el string best_trial (run_bohb.py la
    # agrega como "direction=..."); si no aparece, asumimos el default
    # (ecg2emg) de src/run_bohb.py.
    def _row_direction(bt):
        m = re.search(r"direction=(\w+)", str(bt))
        return m.group(1) if m else "ecg2emg"
    df = df[df["best_trial"].apply(_row_direction) == direction]
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=None,
                    help="Filtra a un dataset (ckitpvc/bordeaux/auckland/utah2002). "
                         "Por defecto compara todos los que tengan corridas standalone.")
    p.add_argument("--direction", choices=["ecg2emg", "emg2ecg"], default="ecg2emg",
                    help="Dirección a comparar (default ecg2emg, el problema inverso real "
                         "y el default de src/run_bohb.py).")
    p.add_argument("--out-csv", default=None,
                    help="Si se indica, guarda la tabla comparativa ahí (CSV).")
    args = p.parse_args()

    standalone = load_standalone(args.dataset, args.direction)
    pipeline = load_pipeline(args.dataset, args.direction)

    if standalone.empty:
        print("No hay corridas standalone que mostrar (filtro:", args.dataset, ")")
        return

    rows = []
    for dataset_name in sorted(standalone["dataset"].unique()):
        sa = standalone[standalone["dataset"] == dataset_name].sort_values("timestamp").tail(1)
        pl = pipeline[pipeline["dataset"] == dataset_name]

        sa_row = sa.iloc[0] if not sa.empty else None
        if not pl.empty:
            # la corrida más reciente (evita mezclar con filas de antes del
            # fix de orientación de ejes en preprocessing.py)
            pl_row = pl.sort_values("timestamp").iloc[-1]
        else:
            pl_row = None

        rows.append({
            "dataset": dataset_name,
            "standalone_pcc": None if sa_row is None else round(float(sa_row["best_pcc"]), 4),
            "pipeline_pcc":   None if pl_row is None else round(float(pl_row["PCC_mean"]), 4),
            "standalone_rmse": None if sa_row is None else round(float(sa_row["best_rmse"]), 4),
            "pipeline_rmse":   None if pl_row is None else round(float(pl_row["RMSE_mean"]), 4),
            "standalone_epochs": None if sa_row is None else int(sa_row["epochs"]),
            "pipeline_n_runs": len(pl),
            "standalone_ckpt": None if sa_row is None else sa_row["ckpt_path"],
        })

    table = pd.DataFrame(rows)
    print("\n=== Comparación PhysFormer: standalone (notebook fiel) vs pipeline integrado ===")
    print(table.to_string(index=False))
    print(
        "\nPCC es la métrica comparable entre ambas corridas (invariante a escala).\n"
        "RMSE difiere porque el pipeline integrado no aplica el filtrado "
        "Butterworth ni el Z-score por canal que sí aplica el notebook/standalone "
        "— divergencia de preprocesado conocida, no error de este script.\n"
        "El bug de orientación de ejes (N,T,C) vs (N,C,T) en "
        "src/data/preprocessing.py::load_mat_cached, que afectaba ckitpvc/utah2002 "
        "para los 37 modelos, ya está corregido (ver _ensure_nct en ese archivo). "
        "Las filas 'pipeline_*' usan la corrida más reciente por dataset — si "
        "metrics_log.xlsx todavía tiene filas de otros métodos anteriores al fix "
        "para esos dos datasets, esas siguen inválidas hasta que se re-corran."
    )

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_path, index=False)
        print(f"\nGuardado en: {out_path}")


if __name__ == "__main__":
    main()
