# ADR 0006 — Estrategia de "despliegue"

## Estado

Aceptado.

## Contexto

El proyecto es un pipeline de investigación batch (entrenamiento + HPO),
no un servicio que atienda solicitudes en producción. El `dockerfile`
original (renombrado a `Dockerfile` en esta migración) ya definía una
imagen basada en `nvcr.io/nvidia/pytorch:22.05-py3` para ejecutar el
proyecto de forma reproducible con GPU.

## Problema

¿Qué se entiende por "despliegue" en este proyecto, y qué mecanismo
formalizar?

## Alternativas

1. Diseñar un despliegue de servicio de inferencia (API REST, contenedor
   de serving) — no corresponde al estado actual del proyecto (no existe
   siquiera un script de inferencia standalone, ver `README.md` sección
   14).
2. Formalizar el "despliegue" como **entorno reproducible de ejecución
   batch**: imagen Docker + `compose.yaml` para lanzar
   entrenamientos/HPO de forma consistente entre máquinas.

## Decisión

Se adopta la alternativa 2.

## Justificación

Corresponde al uso real y verificado del proyecto. Diseñar una API de
serving sería anticipar una funcionalidad que no existe y podría
contradecir la restricción de "no inventes información faltante" /
"no agregues dependencias innecesarias".

## Consecuencias

- Positivas: `Dockerfile`/`compose.yaml` reflejan el uso real (entrenamiento
  batch con GPU montando datos/checkpoints como volúmenes).
- Negativas: si el proyecto evoluciona hacia servir un modelo entrenado en
  producción, se requerirá un ADR nuevo y el desarrollo de
  `scripts/predict.py` (ver pendiente en `MIGRATION_REPORT.md`).
