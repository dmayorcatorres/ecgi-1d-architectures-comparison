# Experimentos

Ver `docs/experiments.md` para la explicación completa del mecanismo de
trazabilidad. Resumen:

- `experiment_registry.csv`: registro de ejecuciones (metadatos:
  commit, dataset, arquitectura, semilla, hardware, checkpoint, métricas
  de referencia). Registro **manual** por ahora (ver TODO en
  `docs/experiments.md` sobre automatización futura).
- Los resultados numéricos detallados siguen viviendo en
  `src/metrics_log.xlsx` (comportamiento preexistente, no modificado).

## Cómo añadir un experimento

1. Ejecutar `src/run_bohb.py` (ver `README.md`, sección 12, o
   `docs/user_manual.md`).
2. Añadir una fila a `experiment_registry.csv` con los campos descritos
   en `docs/experiments.md`.
