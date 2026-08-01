# ADR 0005 — Almacenamiento de datos y modelos

## Estado

Aceptado.

## Contexto

Los datasets (`.mat`, hasta cientos de MB cada uno) y los checkpoints
entrenados (`.pt`, 745 MB en total observados históricamente) no deben
versionarse en Git — ya estaban correctamente excluidos por `.gitignore`
en el repositorio original (`*.mat`, `*.pt`, `*.pth`), aunque
`src/ray_results/` (resultados intermedios de Ray Tune) sí quedó
versionado históricamente por 2,432 archivos antes de que se ampliara el
`.gitignore`.

## Problema

¿Cómo declarar y documentar el almacenamiento de datos/modelos pesados sin
introducir una dependencia nueva (DVC, Git LFS, bucket cloud) que el
equipo no ha decidido adoptar?

## Alternativas

1. Adoptar Git LFS o DVC ahora mismo para versionar datos/checkpoints.
2. Mantener almacenamiento local/institucional fuera de Git (estado
   actual), documentando la convención de rutas y añadiendo `data/README.md`
   y `models/README.md` como "datasheet"/ficha, sin mover binarios.

## Decisión

Se adopta la alternativa 2 para esta migración; se recomienda Git LFS o
DVC como paso futuro opcional.

## Justificación

Introducir DVC/LFS implica decisiones de infraestructura (dónde vive el
remoto de almacenamiento, credenciales, costos) que exceden el alcance de
una migración de organización de repositorio y no deben decidirse
unilateralmente. Se documenta la recomendación en `data/README.md` y
`docs/data_documentation.md` para que el equipo la evalúe.

## Consecuencias

- Positivas: no se introduce infraestructura nueva sin decisión explícita
  del equipo; los binarios permanecen donde ya estaban, sin riesgo de
  pérdida de datos por un movimiento incorrecto.
- Negativas: reproducir el entorno de datos completo sigue requiriendo
  coordinación manual (copiar `.mat` a las rutas documentadas) en lugar de
  un `dvc pull`/`git lfs pull` automatizado.
