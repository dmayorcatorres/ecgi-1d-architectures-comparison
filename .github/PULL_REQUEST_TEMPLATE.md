## Qué cambia

<!-- Descripción breve del cambio. -->

## Por qué

<!-- Motivación / issue relacionado. -->

## Tipo de cambio

- [ ] `feat` — nueva funcionalidad
- [ ] `fix` — corrección de bug
- [ ] `docs` — solo documentación
- [ ] `refactor` — sin cambio de comportamiento
- [ ] `test` — solo pruebas
- [ ] `chore` — mantenimiento/CI/dependencias

## ¿Afecta reproducibilidad o resultados científicos?

- [ ] No.
- [ ] Sí — describir el impacto (semillas, splits, hiperparámetros por
      defecto, versiones de dependencias) y adjuntar comparación
      antes/después de métricas si aplica (ver `docs/experiments.md`).

## Checklist

- [ ] `ruff check src tests` sin errores nuevos.
- [ ] `pytest tests/unit tests/data tests/model tests/integration -q` en verde.
- [ ] Documentación actualizada si el cambio lo requiere (`README.md`,
      `docs/`, `CHANGELOG.md`).
- [ ] Si se añadió una arquitectura/dataset: registrado en
      `models/register.py` (no en `model_factory.py`, deprecado) y
      documentado en `docs/architecture.md`.
