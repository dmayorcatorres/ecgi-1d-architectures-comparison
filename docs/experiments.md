# Trazabilidad experimental

## Mecanismo

`experiments/experiment_registry.csv` es el registro central de
ejecuciones. Cada fila documenta una ejecución de `run_bohb.py`
(un `dataset` × `method` × `direction`). Ver esquema de columnas en el
propio CSV (encabezado) y ejemplo mínimo más abajo.

Este registro es **complementario** a `src/metrics_log.xlsx` (que ya
existía y sigue siendo actualizado automáticamente por
`utils/logbook.py::log_metrics_to_excel` en cada corrida). La diferencia:

| | `src/metrics_log.xlsx` | `experiments/experiment_registry.csv` |
|---|---|---|
| Quién lo escribe | Automático, en cada corrida de `run_bohb.py` | Manual (o automatizable a futuro), por quien ejecuta el experimento |
| Contenido | Métricas de evaluación | Metadatos de trazabilidad (commit, semilla, hardware, duración, checkpoint) + referencia a las métricas |
| Formato | Excel binario (difícil de diffear en Git) | CSV (versionable, diffeable) |

## Cómo registrar un experimento nuevo

1. Ejecutar `run_bohb.py` (ver `README.md`, sección 12).
2. Anotar manualmente una fila en `experiments/experiment_registry.csv`
   con: `run_id`, `date`, `git_commit` (`git rev-parse --short HEAD`),
   `dataset`, `direction`, `method`, `data_version` (hash del `.pt`
   cacheado, ver `docs/data_documentation.md`), `split`, `seed`,
   `config_file`, `framework_version` (`python -c "import torch;
   print(torch.__version__)"`), `hardware`, `duration_min`, `checkpoint_path`,
   `metrics_ref` (fila/hoja correspondiente en `metrics_log.xlsx`),
   `notes`.

```text
TODO: automatizar este registro (hoy manual) añadiendo, en una iteración
futura, un hook en run_bohb.py que escriba automáticamente una fila en
experiment_registry.csv al finalizar run_pipeline(). No se implementó en
esta migración por ser un cambio de comportamiento del pipeline activo —
ver MIGRATION_REPORT.md.
```

## Esquema de columnas de `experiment_registry.csv`

| Columna | Descripción |
|---|---|
| `run_id` | Identificador único (sugerido: `<dataset>_<method>_<direction>_<fecha>`) |
| `date` | Fecha ISO-8601 de la ejecución |
| `git_commit` | Hash corto del commit de código usado |
| `data_version` | Hash del `.pt` cacheado del dataset (ver `docs/data_documentation.md`) |
| `split` | Estrategia de partición (`60/20/20-stratified` por defecto) |
| `seed` | Semilla usada (`GLOBAL_SEED`, hoy fija en 42) |
| `config_file` | Ruta al YAML de referencia en `configs/` (documental, ver ADR 0003) |
| `architecture` | Clave en `MODEL_REGISTRY` (p. ej. `mlp`, `resnet`, `patchtst`) |
| `hyperparameters` | Resumen de hiperparámetros del mejor trial BOHB (o ruta al `_config.json` del checkpoint) |
| `framework_version` | Versión de PyTorch/Ray usada |
| `hardware` | GPU/CPU usados |
| `duration_min` | Duración total en minutos |
| `checkpoint_path` | Ruta al `.pt` resultante |
| `metrics` | Resumen de métricas clave (MAE/RMSE/CC) o referencia a `metrics_log.xlsx` |
| `notes` | Observaciones libres |

## Ejemplo mínimo (ver también la fila de ejemplo en el CSV)

```bash
cd src
python run_bohb.py --dataset kitpvc --method mlp --direction ecg2emg \
    --num-samples 2 --max-epochs 2
git rev-parse --short HEAD   # registrar este valor en experiment_registry.csv
```
