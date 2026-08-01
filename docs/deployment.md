# Despliegue

Ver `docs/adr/0006-estrategia-despliegue.md` para la decisión y su
justificación completa.

## Alcance actual

Este proyecto **no está desplegado como servicio**. "Despliegue" aquí se
refiere a ejecutar el pipeline de entrenamiento/HPO de forma reproducible
mediante contenedor.

## Imagen Docker

```bash
docker build -t ecgi-1d-architectures-comparison:local .
```

Basada en `nvcr.io/nvidia/pytorch:22.05-py3` (heredada del proyecto
original, no cambiada en esta migración — ver ADR 0006). Usuario no root
(`ecgi`), healthcheck de importación de dependencias críticas (ver
`Dockerfile`, `SECURITY.md`).

## Ejecución con Docker Compose

```bash
docker compose build
docker compose run --rm ecgi bash
docker compose run --rm ecgi python src/run_bohb.py \
    --dataset kitpvc --method mlp --num-samples 2 --max-epochs 2
```

Requiere NVIDIA Container Toolkit para exponer GPU al contenedor. Datos,
checkpoints y reportes se montan como volúmenes (`compose.yaml`) — nunca
se incluyen en la imagen (`.dockerignore`).

## Entornos

| Entorno | Descripción |
|---|---|
| Desarrollo/pruebas | CPU, sin GPU, `pytest tests/unit` |
| Entrenamiento de referencia | GPU única + 6 CPUs (`ray.init` defaults en `run_bohb.py`) |
| Producción | No aplica (no hay servicio desplegado) |

## Futuro: despliegue de inferencia

TODO: si el proyecto evoluciona a servir predicciones (API/batch scoring),
requiere: `scripts/predict.py`, un ADR nuevo, y definición de SLA/latencia
— ninguno de estos existe hoy. No se implementa en esta migración para no
inventar funcionalidad no solicitada.
