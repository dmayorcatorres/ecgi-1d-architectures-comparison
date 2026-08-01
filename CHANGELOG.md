# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/),
versionado según [Semantic Versioning](https://semver.org/lang/es/).

## [Unreleased]

### Changed (refactor, 2026-08-01)
- Se respaldó en Git el pipeline activo que nunca había sido commiteado
  (`src/run_bohb.py`, `src/run_smoke_test.sh`, `src/test_gpu.py`,
  `src/utils/{data_checks,logbook,profile,ray_helpers,runtime,split,
  metrics}.py`), verificado funcionando de punta a punta con datos reales
  y GPU antes del commit.
- Deduplicados `ConvSEBlock.py`, `DisjointBase.py`, `ResidualBlock.py` y
  `SqueezeExciteBlock.py`: existían copias idénticas en
  `models/base/`, `models/cnnbased/` y `models/hybridbased/` (12 archivos
  → 4). Se corrigió además un import absoluto roto en
  `models/base/ConvSEBlock.py` (`from src.models.base...`, incompatible
  con el import root plano usado en todo el proyecto) y el comentario de
  cabecera incorrecto en `models/base/__init__.py`. Los 8 módulos que
  usaban las copias locales (`DisjointCNN`, `DisjointResNet`, `FCN`,
  `ResNet`, `TFRCNN`, `CNN_GRU`, `CNN_LSTM_GRU`, `CNN_LSTM`) ahora
  importan desde `models.base`. Verificado con `pytest` (29/29) y forward
  pass real de los 8 modelos afectados.

### Added
- Reorganización del repositorio alineada con principios de ISO/IEC 5338,
  ISO/IEC/IEEE 12207/15289/42010, ISO/IEC 23053, ISO/IEC 5259,
  ISO/IEC 25010, ISO/IEC/IEEE 26514, ISO/IEC 23894, ISO/IEC 24029, Model
  Cards, Datasheets for Datasets y Citation File Format (ver
  `COMPLIANCE_MATRIX.md`).
- Documentación de gobernanza: `README.md` reescrito, `INSTALL.md`,
  `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`,
  `LICENSE` (MIT).
- Documentación técnica en `docs/`: arquitectura, ficha de datos, tarjeta
  de modelo, reproducibilidad, experimentos, riesgos y limitaciones,
  evaluación, despliegue, monitoreo, manual de usuario, y registros de
  decisiones arquitectónicas (`docs/adr/`).
- `configs/` con YAML para datos, modelo, entrenamiento y evaluación,
  externalizando valores antes fijados/comentados en código
  (`src/configs/base_config.py`).
- Suite de pruebas inicial en `tests/` (unit, integration, data, model,
  smoke) ejecutable con `pytest`.
- Integración continua en `.github/workflows/` (lint, tests, validación de
  Dockerfile, verificación de enlaces, análisis de seguridad, generación
  de SBOM).
- `experiments/experiment_registry.csv` y `docs/experiments.md` como
  mecanismo de trazabilidad experimental.
- `sbom/README.md` y generación de SBOM (`cyclonedx-bom`).
- `pyproject.toml` como fuente central de metadatos y configuración de
  herramientas (ruff, mypy, pytest).
- `.dockerignore`, `compose.yaml`, `Dockerfile` (renombrado desde
  `dockerfile`) reforzado con usuario no root, healthcheck y caché de
  capas de dependencias.

### Changed
- `.gitignore` reforzado: cobertura explícita de `src/checkpoints/`,
  `src/data/**/raw/`, entornos virtuales, cachés de herramientas e
  instaladores sueltos (`*.deb`, `*.pin`).

### Deprecated
- `src/models/model_factory.py` (duplicado no referenciado del factory
  activo `src/models/factory.py` + `register.py`).
- `src/optimization/ray_optimizer.py` y `src/optimization/hpo_spaces.py`
  (no referenciados por el pipeline activo `run_bohb.py`).
- `src/training/engines.py` (motor de entrenamiento alternativo con
  Ignite/MLflow/W&B, solo referenciado por el `ray_optimizer.py`
  deprecado, nunca conectado al flujo real).

Ver `MIGRATION_REPORT.md` para la justificación completa de cada hallazgo
y por qué no se eliminó código sin confirmación explícita.

### Security
- Ver `SECURITY.md` para el detalle de hallazgos (sin credenciales
  expuestas; uso correcto de `weights_only=True` en `torch.load`).

## [0.1.0] - Histórico (antes de esta auditoría)

Historia de desarrollo previa a esta reorganización documentada en el
historial de Git (`git log`); no se reconstruyen aquí versiones no
etiquetadas formalmente en el repositorio original.
