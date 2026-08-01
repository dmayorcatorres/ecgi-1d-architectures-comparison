# ADR 0004 — Tratamiento del código no referenciado (factories y motor de entrenamiento duplicados)

## Estado

Superado — ver "Resolución" al final. Se conserva el resto del ADR como
registro histórico de la decisión original.

## Contexto

Trazando los imports reales desde el único punto de entrada activo
(`src/run_bohb.py`, invocado por `run_experiments.sh` y
`run_smoke_test.sh`), se verificó que:

- `src/models/model_factory.py` (factory con diccionario hardcodeado de
  36 modelos) **no es importado por ningún módulo activo**. El factory
  realmente usado es `src/models/factory.py` + `src/models/register.py`
  (`MODEL_REGISTRY`/`SPACE_REGISTRY`).
- `src/optimization/ray_optimizer.py` y `src/optimization/hpo_spaces.py`
  **no son importados por ningún módulo activo**.
- `src/training/engines.py` (motor de entrenamiento con integración a
  PyTorch Ignite, DDP, AMP, Apex, MLflow, W&B, Hydra) **solo es importado
  por `ray_optimizer.py`**, que a su vez no está conectado al pipeline
  activo. Es, en la práctica, un segundo motor de entrenamiento que nunca
  llegó a usarse desde `run_bohb.py`.

## Problema

¿Eliminar este código para reducir duplicación y confusión, o
conservarlo?

## Alternativas

1. Eliminar los archivos no referenciados.
2. Conservarlos sin cambios (riesgo: un colaborador nuevo podría editar el
   factory o el motor equivocado, como ya evidencia la existencia misma de
   la duplicación).
3. Conservarlos, pero marcarlos explícitamente como deprecados
   (docstring de advertencia + mención en `README.md`/`CONTRIBUTING.md`/
   `CHANGELOG.md`) sin moverlos ni borrarlos en esta migración.

## Decisión

Se adopta la alternativa 3.

## Justificación

La restricción explícita de esta migración es "no elimines archivos sin
justificarlo" y "no cambies la lógica científica sin justificarlo". Aunque
el análisis de imports es una evidencia fuerte de que este código no se
ejecuta desde el flujo automatizado, no se puede descartar que:

- Se invoque manualmente en notebooks/sesiones interactivas no
  versionadas.
- Represente trabajo en progreso de una migración a Ignite/MLflow que el
  equipo planea retomar.

Por eso se documenta la duplicación explícitamente y se recomienda al
equipo (ver `MIGRATION_REPORT.md`) decidir en un plazo definido si:
(a) retoma y conecta `training/engines.py` como motor de entrenamiento
alternativo real, migrando `run_bohb.py` a usarlo, o (b) confirma su
eliminación en un commit dedicado y reversible.

## Consecuencias

- Positivas: cero riesgo de romper un flujo de trabajo no visible en el
  historial de Git; trazabilidad completa de la decisión.
- Negativas: la duplicación persiste hasta que el equipo decida; se
  agregan advertencias de docstring para mitigar el riesgo de que se edite
  el archivo equivocado mientras tanto.

## Resolución (2026-08-01)

El responsable del proyecto confirmó explícitamente la eliminación
(alternativa 1) en una sesión de refactoring posterior, una vez que:

1. El pipeline activo (`run_bohb.py` y `src/utils/*.py`) quedó respaldado
   en Git (commit `4d43b825`) — ya no dependía únicamente del working
   tree local.
2. Se volvió a verificar (trazado de imports) que `model_factory.py`,
   `optimization/hpo_spaces.py`, `optimization/ray_optimizer.py` y
   `training/engines.py` seguían sin ser referenciados por ningún módulo
   activo.

Se eliminaron los 5 archivos (`src/models/model_factory.py`,
`src/optimization/hpo_spaces.py`, `src/optimization/ray_optimizer.py`,
`src/training/engines.py`, `src/training/__init__.py` — este último
quedó vacío de propósito al eliminar `engines.py`, su único contenido)
en un commit dedicado y reversible (`git log` / `git revert` si el
equipo necesita recuperarlos). Verificado tras la eliminación: `pytest`
29/29 en verde e importación limpia de `run_bohb.py`.

No se retomó ni conectó `training/engines.py` — no había evidencia,
tras revisión, de que estuviera en uso en ningún flujo no versionado.
