# Política de seguridad

Este documento resume el estado de seguridad verificado del repositorio al
momento de esta auditoría (2026-08-01) y las prácticas que deben seguirse.
No es una certificación; es un registro de hallazgos y recomendaciones
alineado con los principios de ISO/IEC 23894 (gestión de riesgos de IA) y
buenas prácticas generales de seguridad de la cadena de suministro de
software.

## Reporte de vulnerabilidades

TODO: definir un canal de reporte responsable (correo dedicado o
`Security` de GitHub) para este repositorio. Mientras tanto, reportar
directamente al responsable del proyecto (ver `CITATION.cff`).

## Hallazgos verificados en esta auditoría

| Ítem | Estado | Detalle |
|---|---|---|
| Secretos/credenciales en código | ✅ Sin hallazgos | Escaneo de patrones `api_key/secret/password/token` sobre todo `src/*.py` no encontró credenciales; los únicos resultados fueron docstrings sobre *tokens* de secuencias en modelos de atención (`PatchTST.py`, `BERT_BIGRU.py`), no relacionados con autenticación. |
| Archivos `.env` versionados | ✅ Sin hallazgos | No existen archivos `.env` en el repositorio. Se agregó `.env.example` como plantilla sin valores sensibles. |
| Deserialización insegura (`pickle`) | ✅ Mitigado | `src/data/preprocessing.py` usa `torch.load(cache, map_location="cpu", mmap=True, weights_only=True)`. `weights_only=True` evita la ejecución de código arbitrario embebido en checkpoints/pickles maliciosos — es la práctica recomendada por PyTorch desde la CVE de deserialización de `torch.load`. |
| Checkpoints de terceros | ⚠️ Sin política explícita | El proyecto genera sus propios checkpoints (`src/checkpoints/*.pt`); no se detectó descarga de pesos preentrenados de terceros. Si en el futuro se incorporan pesos externos (p. ej. modelos pre-entrenados de HuggingFace/torchvision), deben cargarse siempre con `weights_only=True` y verificarse su procedencia (hash/firma) antes de usarse. |
| Dependencias sin versión acotada | ⚠️ Corregido parcialmente | `setup.py` y `README.md` declaraban dependencias sin versión y de forma inconsistente entre sí. Se centralizó en `requirements.txt`/`requirements-dev.txt` con rangos mínimos. Pendiente: fijar versiones exactas y ejecutar `pip-audit` periódicamente (ver workflow `security.yml`). |
| Artefactos binarios pesados en historial de Git | ⚠️ Ver `MIGRATION_REPORT.md` | `src/ray_results/` quedó versionado históricamente (~2400 archivos, `.git` de 1.2 GB). No representa una vulnerabilidad de seguridad per se, pero es un riesgo de higiene/cadena de suministro (dificulta auditar qué se distribuye). Ver plan de purga de historial. |
| Instaladores `.deb` de terceros en el árbol de trabajo | ⚠️ Recomendación | `claude-desktop_*.deb` y `cuda-repo-*.deb` (no versionados) conviven con el código. Se agregó `*.deb`/`*.pin` a `.gitignore` para evitar que se versionen accidentalmente; se recomienda eliminarlos del árbol de trabajo local (fuera del alcance de este repositorio). |
| CI que ejecute análisis de seguridad | ❌ No existía | Se agrega `.github/workflows/security.yml` con `pip-audit` sobre `requirements.txt`. |

## Prácticas obligatorias para colaboradores

- **No** ejecutar ni cargar checkpoints (`*.pt`) recibidos de fuentes no
  verificadas sin `weights_only=True` y revisión previa.
- **No** commitear datasets clínicos ni derivados que puedan contener
  información identificable (ver `docs/data_documentation.md`, sección de
  privacidad).
- **No** incluir credenciales, tokens de APIs (Ray, MLflow, W&B) ni rutas
  absolutas del entorno personal en código o configuración versionada; usar
  variables de entorno (`.env`, no versionado) siguiendo `.env.example`.
- Ejecutar `pip-audit -r requirements.txt` antes de introducir una
  dependencia nueva.
- Revisar `git status`/`git diff` antes de cada commit para evitar subir
  artefactos pesados o generados (`__pycache__`, `*.pt`, `*.mat`, logs).

## Confianza y procedencia de pesos preentrenados

Actualmente el proyecto **no consume pesos preentrenados de terceros**;
todos los checkpoints se generan localmente vía `src/run_bohb.py`. Si esto
cambia:

1. Documentar la fuente exacta (repositorio, commit/tag, autor) en
   `docs/model_card.md`.
2. Verificar integridad (hash SHA-256 publicado por la fuente, si existe).
3. Cargar siempre con `torch.load(..., weights_only=True)`.
4. Registrar la incorporación como una entrada nueva en `CHANGELOG.md` y,
   si aplica, un ADR en `docs/adr/`.

## Alcance de esta política

Esta política cubre el código y la configuración de este repositorio. No
cubre la infraestructura de cómputo (clústeres GPU, almacenamiento
institucional) donde se ejecutan los entrenamientos, que está fuera del
alcance de este documento.
