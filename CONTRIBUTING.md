# Guía de contribución

Gracias por tu interés en contribuir a `ecgi-1d-architectures-comparison`.
Este documento describe cómo proponer cambios de forma consistente con la
organización del repositorio (ver `docs/architecture.md` y
`docs/adr/` para el contexto de las decisiones de diseño).

## Antes de empezar

1. Lee `README.md` e `INSTALL.md` para levantar el entorno.
2. Revisa `docs/risks_and_limitations.md` si tu cambio afecta datos,
   métricas o arquitectura de modelos — el proyecto es de investigación y
   los resultados reportados deben mantenerse trazables.
3. Si tu cambio afecta el comportamiento científico (arquitecturas,
   hiperparámetros por defecto, splits, métricas), ábrelo primero como
   issue para discutir el impacto antes de implementarlo.

## Flujo de trabajo

1. Crea una rama descriptiva: `feat/<tema>`, `fix/<tema>`, `docs/<tema>`.
2. Sigue la convención de commits [Conventional Commits](https://www.conventionalcommits.org/):
   `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
3. Ejecuta antes de abrir el PR:
   ```bash
   ruff check src tests
   pytest tests/unit tests/data tests/model -q
   ```
4. Abre el pull request usando la plantilla de
   `.github/PULL_REQUEST_TEMPLATE.md` y describe:
   - qué cambia y por qué;
   - si afecta reproducibilidad (semillas, splits, versiones de
     dependencias);
   - si afecta métricas reportadas (adjuntar comparación antes/después).
5. Espera revisión: al menos un cambio en `src/models/`, `src/training/`
   o `src/data/` debe ser revisado por otra persona antes de fusionarse
   (ver reglas recomendadas de protección de rama en
   `.github/workflows/`).

## Estilo de código

- Python ≥ 3.10, `ruff` para lint/formato (`pyproject.toml`).
- Rutas mediante `pathlib.Path`, no concatenación de strings.
- `logging` en lugar de `print` en código nuevo bajo `src/` (el código
  existente mezcla ambos; no es necesario migrar código no tocado, pero sí
  el código nuevo o modificado).
- Funciones públicas nuevas con docstring breve y, cuando sea razonable,
  type hints.
- No introducir rutas absolutas del entorno personal.

## Añadir una nueva arquitectura de modelo

1. Implementar el módulo en `src/models/<familia>/NuevoModelo.py`
   (`cnnbased/`, `rnnbased/`, `hybridbased/` o `atbased/`, según
   corresponda).
2. Registrar la clase y su espacio de búsqueda de hiperparámetros en
   `src/models/register.py` usando `register_model()` y
   `@register_space(...)`. Si necesitas un bloque reutilizable
   (SE, residual, etc.), colócalo en `src/models/base/` e impórtalo con
   `from models.base.X import Y` — no dupliques el archivo dentro de tu
   familia de arquitectura (ver `docs/adr/0004-codigo-deprecado.md`).
3. Añadir una prueba en `tests/model/` que valide forma de entrada/salida
   con un tensor sintético pequeño (ver ejemplos existentes).
4. Documentar el modelo (familia, referencia/paper si aplica) en
   `docs/architecture.md`.

## Añadir o modificar un dataset

Ver `docs/data_documentation.md` antes de integrar un dataset nuevo:
formato esperado (`.mat` con claves `ecg`/`emg`/`tag`), ubicación
(`src/data/<nombre>/raw/`), y actualización de `_datasets` en
`src/run_bohb.py`.

## Pruebas

Ver `tests/README.md` (dentro de `tests/`) para la organización de
unit/integration/data/model/smoke tests. Las pruebas que requieren GPU
están marcadas con `@pytest.mark.gpu` y se omiten automáticamente si no
hay CUDA disponible.

## Contacto

Ver responsable del proyecto en `CITATION.cff`.
