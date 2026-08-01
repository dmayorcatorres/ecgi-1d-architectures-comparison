# Tarjeta de modelo (Model Card)

Formato inspirado en Mitchell et al., *Model Cards for Model Reporting*
(2019) y alineado con los principios de ISO/IEC 23053 e ISO/IEC 24029
(robustez de redes neuronales). No implica certificación. Este proyecto
entrena **36 arquitecturas distintas**; esta tarjeta describe el sistema
de comparación como conjunto y los campos comunes a todas las variantes.
Para resultados de una arquitectura/dataset específico, ver
`experiments/experiment_registry.csv` y `src/metrics_log.xlsx`.

## Identificación del modelo

Familia de 36 arquitecturas 1D registradas en `src/models/register.py`
(`MODEL_REGISTRY`), agrupadas en 4 familias: CNN, RNN, híbridas
(CNN+RNN) y basadas en atención/transformers. Ver `docs/architecture.md`
y `configs/model/default.yaml` para el listado completo.

## Versión

No hay versionado semántico por modelo entrenado más allá del checkpoint
y su `*_config.json` asociado (`src/checkpoints/<dataset>_<variante>_best.pt`
+ `_config.json`). Se recomienda usar `experiments/experiment_registry.csv`
(nuevo, ver `docs/experiments.md`) como mecanismo de versionado por
ejecución.

## Propósito

Estimar una señal fisiológica (EMG) a partir de otra (ECG), o su inversa
(`emg2ecg`), como problema de regresión secuencia-a-secuencia, para
comparar el desempeño relativo de familias de arquitecturas 1D.

## Usos previstos

- Investigación y comparación metodológica de arquitecturas de deep
  learning para señales electrofisiológicas 1D.
- Referencia para selección de arquitectura en trabajos futuros sobre
  ECGi.

## Usos no recomendados

- **Uso clínico o diagnóstico directo.** No hay evidencia en el
  repositorio de validación clínica, aprobación regulatoria, ni de un
  proceso de aseguramiento de calidad más allá de las métricas de
  regresión reportadas.
- Despliegue en producción sin una evaluación de robustez adicional (ver
  `docs/risks_and_limitations.md` y ISO/IEC 24029).
- Uso sobre poblaciones o condiciones de adquisición distintas a las de
  los datasets de entrenamiento sin re-evaluación (riesgo de *distribution
  shift*, ver limitaciones).

## Arquitectura

36 variantes; cada una definida en su propio módulo bajo
`src/models/{cnnbased,rnnbased,hybridbased,atbased}/`. Parámetros comunes
a todas: `time_steps`, `input_channels`, `output_channels` (inyectados en
runtime por `ModelFactory` según la forma real de los datos cargados). Ver
`docs/architecture.md` para el diagrama de componentes.

## Entradas

Tensor `(batch, input_channels, time_steps)` — señal ECG o EMG según
`direction`, `time_steps=256` por defecto.

## Salidas

Tensor `(batch, output_channels)` o `(batch, output_channels, time_steps)`
según arquitectura — señal complementaria estimada (EMG o ECG).

```text
TODO: confirmar y documentar por familia si la salida es por-muestra
(vector) o por-secuencia (serie temporal completa) — varía entre
arquitecturas y no se generalizó aquí para no inventar un contrato único
que no esté verificado para las 36 variantes.
```

## Preprocesamiento

Ver `docs/data_documentation.md`, sección "Preprocesamiento": recorte a
`time_steps=256`, cacheo en `.pt`, verificación de NaN/Inf por batch.

## Datos de entrenamiento

Split 60/20/20 estratificado (ver `docs/data_documentation.md`); se
entrena únicamente sobre la partición `train`, con validación en `val`
durante HPO y evaluación final held-out en `test`.

## Función de pérdida

`nn.MSELoss()` (verificado en `src/run_bohb.py`, usado tanto en el bucle
de HPO como en el entrenamiento final).

## Optimizador

`torch.optim.AdamW` con weight decay desacoplado (Loshchilov & Hutter,
2019), con dos grupos de parámetros (sin `weight_decay` para
BatchNorm/LayerNorm y biases) — ver `_make_optimizer()` en
`src/run_bohb.py`. Scheduler: warmup lineal (`LinearLR`) seguido de
`CosineAnnealingLR`, combinados con `SequentialLR`.

## Hiperparámetros

Espacio de búsqueda por arquitectura definido en `SPACE_REGISTRY`
(`src/models/register.py`); hiperparámetros comunes a todos los trials:
`weight_decay ∈ {0, 1e-6, 1e-5, 1e-4}`, `batch_size ∈ {8, 16, 32}`,
`learning_rate` (por variante, `_LR_CHOICES`/`_LR_DEFAULT` en
`run_bohb.py`). Ver `configs/model/default.yaml` y
`configs/training/default.yaml`.

## Procedimiento de entrenamiento

1. HPO con BOHB (`num_samples` trials, hasta `max_epochs` épocas cada
   uno) sobre `train`/`val`.
2. Reentrenamiento del mejor trial hasta `final_epochs` (`TrainingConfig`,
   default 200, sobrescribible).
3. Evaluación única sobre `test` held-out.

## Métricas

MAE, MSE, RMSE, coeficiente de correlación (CC), similitud coseno
espectral, correlación cruzada máxima — implementadas en
`src/utils/metrics.py::ECGMetricsCalculator`. Registradas por ejecución en
`src/metrics_log.xlsx`.

## Criterios de aceptación

TODO: no se encontró en el repositorio un umbral de aceptación
formalmente definido (p. ej. "RMSE < X para considerar el modelo válido").
El equipo debe definirlo si se requiere para publicación o uso posterior.

## Resultados generales

```text
TODO: no se resumen aquí valores agregados de las 36 arquitecturas × 5
datasets para evitar transcribir manualmente cifras de un archivo binario
(metrics_log.xlsx) que cambia con cada ejecución y podría introducir
errores de transcripción. Ver docs/experiments.md para el procedimiento
de consulta de resultados vigentes y experiments/experiment_registry.csv
para el mecanismo de registro estructurado hacia adelante.
```

## Resultados por subgrupos

TODO: no hay evidencia de análisis de desempeño por subgrupo (p. ej. por
clase de `tag`, por dataset de origen) más allá de la comparación
agregada por dataset/arquitectura en `metrics_log.xlsx`.

## Robustez

TODO: no se encontró evaluación de robustez (perturbaciones, ruido,
ejemplos adversariales) en el repositorio — ver
`docs/risks_and_limitations.md` y principios de ISO/IEC 24029 como marco
de referencia para trabajo futuro, no como estado actual.

## Incertidumbre

TODO: el pipeline no reporta intervalos de confianza ni estimaciones de
incertidumbre por predicción (single-point estimates). BOHB sí evalúa
múltiples trials, pero eso es incertidumbre sobre hiperparámetros, no
sobre la predicción del modelo final.

## Sesgos

Ver `docs/data_documentation.md`, sección "Sesgos conocidos" (TODO,
pendiente de información del equipo).

## Limitaciones

Ver `docs/risks_and_limitations.md`.

## Requisitos computacionales

Ver `INSTALL.md`, sección "Requisitos de hardware". Entrenamiento diseñado
para 1 GPU + 6 CPUs (`ray.init(num_cpus=6, num_gpus=1)`, default en
`run_bohb.py`).

## Impacto ambiental

TODO: no se midió consumo energético/huella de carbono de los
entrenamientos (36 arquitecturas × 5 datasets × N trials de HPO). Si se
requiere, se recomienda instrumentar con herramientas como
`codecarbon` en una iteración futura — no se agrega como dependencia
nueva en esta migración sin que el equipo lo solicite explícitamente.

## Licencia

MIT para el código (ver `LICENSE`). Ver `data/README.md` para el estado
(pendiente) de licencia de los datos de entrenamiento.

## Responsable

Dago Mayorca Torres — `dmayorca@umariana.edu.co` (ver `CITATION.cff`).

## Historial de versiones

Ver `CHANGELOG.md`.
