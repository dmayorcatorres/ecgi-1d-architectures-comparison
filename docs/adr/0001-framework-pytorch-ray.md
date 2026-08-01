# ADR 0001 — Framework de deep learning y de HPO

## Estado

Aceptado (decisión heredada del código existente; documentada
retroactivamente en esta auditoría, no re-decidida).

## Contexto

El proyecto necesita entrenar ~36 arquitecturas 1D de deep learning con
búsqueda de hiperparámetros automatizada, sobre datos electrofisiológicos
de tamaño moderado, con uso de GPU.

## Problema

Elegir un framework de deep learning y un mecanismo de optimización de
hiperparámetros (HPO) que soporte múltiples arquitecturas heterogéneas
(CNN, RNN, atención) y búsqueda eficiente con early-stopping.

## Alternativas consideradas

TODO: no hay registro documental de qué alternativas se evaluaron
(TensorFlow/Keras, Optuna standalone, scikit-optimize, etc.) antes de
elegir PyTorch + Ray Tune. Se documenta aquí solo la decisión ya
implementada en el código.

## Decisión

- **PyTorch** como framework de modelado (`torch`, `torch.nn`).
- **Ray Tune + BOHB** (`TuneBOHB` + `HyperBandForBOHB`, backend
  HpBandSter/ConfigSpace) como mecanismo de HPO.
- **PyTorch Lightning** únicamente para `ECGDataModule`
  (`pl.LightningDataModule`), no para el bucle de entrenamiento completo
  (el bucle de entrenamiento vive directamente en `run_bohb.py`, no en un
  `pl.Trainer`).

## Justificación

Verificable en el código: `run_bohb.py` implementa el ciclo de
entrenamiento/evaluación directamente sobre PyTorch, y usa Ray Tune como
orquestador de trials con BOHB por su eficiencia frente a grid/random
search en espacios de búsqueda grandes con recursos limitados (GPU
única). `SPACE_REGISTRY` (`models/register.py`) permite definir un
espacio de búsqueda por arquitectura sin acoplar el orquestador a cada
modelo.

## Consecuencias

- Positivas: extensible (nuevo modelo = nueva entrada en
  `MODEL_REGISTRY`/`SPACE_REGISTRY`), HPO eficiente con early-stopping por
  bandas (Hyperband).
- Negativas: dependencia de un stack relativamente pesado (Ray, ConfigSpace,
  HpBandSter) con compatibilidad de versiones sensible entre sí (ver
  `requirements.txt`); coexiste con un motor de entrenamiento alternativo
  (`training/engines.py`, basado en PyTorch Ignite) que nunca se conectó al
  flujo activo — ver ADR 0004.
