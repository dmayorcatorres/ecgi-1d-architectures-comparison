# Monitoreo

## Alcance actual

El sistema no sirve inferencia en producción, por lo que no aplica
monitoreo de servicio (latencia, disponibilidad, *drift* en tiempo real
sobre tráfico productivo). Lo que sí existe y debe monitorearse es la
**calidad del proceso de entrenamiento/HPO**.

## Señales disponibles hoy

| Señal | Fuente |
|---|---|
| Progreso de trials BOHB | Logs de consola de Ray Tune (stdout, redirigido a `batch_logs/` por `run_experiments.sh`/`run_smoke_test.sh`) |
| Detección de NaN/Inf en batches | `utils/data_checks.py::assert_no_nan_batch` (lanza excepción, detiene el trial) |
| Curvas de pérdida train/val | `--save-curves` → `src/loss_curves/*.png`, `*.csv` |
| Métricas finales por ejecución | `src/metrics_log.xlsx` |

## Recomendaciones (no implementadas en esta migración)

- Centralizar logs de `batch_logs/` con un dashboard simple (p. ej.
  TensorBoard, ya parcialmente soportado vía `ignite.contrib.handlers.
  TensorboardLogger` en el motor deprecado `training/engines.py`, pero no
  conectado al pipeline activo — ver ADR 0004).
- Alertas automáticas ante fallos repetidos de trials (hoy requiere
  revisión manual de `batch_logs/<direction>/*.log`, ya excluidos de Git).
- Si se despliega inferencia en el futuro: monitoreo de *distribution
  shift* de las señales de entrada respecto a los datos de entrenamiento
  (ver `docs/risks_and_limitations.md`).

```text
TODO: el equipo debe decidir si invertir en un backend de tracking
(MLflow/W&B, ya importados condicionalmente en training/engines.py pero
no activos) antes de agregarlo como dependencia obligatoria.
```
