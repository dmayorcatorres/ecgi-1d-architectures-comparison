# Evaluación

## Procedimiento

La evaluación se ejecuta dentro de `run_pipeline()`
(`src/run_bohb.py`), sobre la partición `test` (held-out, 20% del split
60/20/20), **una sola vez**, tras el reentrenamiento final del mejor
trial encontrado por BOHB. No hay actualmente un comando de evaluación
desacoplado del entrenamiento (ver limitación en `README.md`).

## Métricas

Implementadas en `src/utils/metrics.py::ECGMetricsCalculator`:

| Métrica | Método | Qué mide |
|---|---|---|
| MAE | `compute_mae` | Error absoluto medio |
| MSE | `compute_mse` | Error cuadrático medio |
| RMSE | `compute_rmse` | Raíz del error cuadrático medio |
| CC | `compute_cc` | Coeficiente de correlación entre señal predicha y real |
| Similitud coseno espectral | `_spectral_cosine_similarity` | Similitud en el dominio de frecuencia |
| Correlación cruzada máxima | `_max_cross_correlation` | Máxima correlación cruzada (tolera desfase temporal) |

`ECGMetricsCalculator.complete_report()` agrega el reporte completo por
ejecución.

## Dónde se registran los resultados

`src/metrics_log.xlsx` (vía `utils/logbook.py::log_metrics_to_excel`),
más `src/loss_curves/*.png`/`*.csv` si se usa `--save-curves`. Ver
`docs/experiments.md` para el mecanismo de trazabilidad complementario.

## Umbrales / criterios de aceptación

TODO: no se encontró un umbral formal de aceptación en el repositorio
(ver `docs/model_card.md`, sección "criterios de aceptación").

## Evaluación por subgrupos

TODO: no implementada actualmente (ver `docs/model_card.md`).

## Limitaciones de la evaluación

- Una sola partición de test por ejecución (sin validación cruzada en la
  fase final, aunque `TrainingConfig.n_splits=5` sugiere que sí se usó
  CV en alguna configuración — TODO: confirmar si CV se aplica al
  entrenamiento final o solo a una vía alternativa no activa).
- Sin intervalo de confianza/varianza entre semillas (una sola semilla
  fija, `GLOBAL_SEED=42`).
