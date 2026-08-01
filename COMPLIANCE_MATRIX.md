# Matriz de alineación con estándares y buenas prácticas

Esta matriz relaciona artefactos del repositorio con los principios de
los estándares/prácticas de referencia. **No implica certificación** bajo
ninguno de estos estándares — es una declaración de alineación de
organización y documentación, verificable por inspección directa de cada
artefacto listado.

Estados: **Implementado** · **Parcial** · **No aplica** · **Pendiente de
información**.

| Área | Artefacto del repositorio | Referencia | Estado | Evidencia | Pendientes |
|---|---|---|---|---|---|
| Ciclo de vida del sistema de IA | `docs/architecture.md`, `docs/experiments.md`, `docs/model_card.md`, `CHANGELOG.md` | ISO/IEC 5338 | Parcial | Flujo completo (datos→modelo→evaluación) documentado con diagramas Mermaid | Falta etapa de monitoreo/retiro post-despliegue (no aplica hoy, sin servicio en producción) |
| Procesos del ciclo de vida del software | `CONTRIBUTING.md`, `.github/workflows/`, `CHANGELOG.md` | ISO/IEC/IEEE 12207 | Parcial | Flujo de contribución, CI, versionado documentados | Sin proceso formal de gestión de requisitos/aceptación (`docs/requirements.md` no se generó — no había requisitos formales que documentar, ver Pendientes) |
| Documentación del ciclo de vida | `docs/*.md`, `README.md`, `INSTALL.md` | ISO/IEC/IEEE 15289 | Implementado | Cobertura de arquitectura, datos, modelo, experimentos, riesgos, reproducibilidad, evaluación, despliegue, monitoreo, manual de usuario | Mantener sincronizado con cambios de código (riesgo documentado en cada ADR) |
| Descripción de arquitectura | `docs/architecture.md`, `docs/adr/0001`–`0006` | ISO/IEC/IEEE 42010 | Implementado | Contexto, interesados, componentes, diagramas Mermaid, 6 ADR con contexto/problema/alternativas/decisión/justificación/consecuencias/estado | — |
| Marco de sistemas ML | `docs/architecture.md`, `docs/model_card.md` | ISO/IEC 23053 | Parcial | Pipeline de datos→modelo→evaluación descrito | Sin taxonomía formal de componentes ML más allá de lo descriptivo |
| Calidad de datos para ML | `docs/data_documentation.md`, `data/README.md` | ISO/IEC 5259 | Pendiente de información | Estructura de ficha de datos (Datasheets for Datasets) completa; origen/licencia/consentimiento marcados TODO explícitamente | El equipo debe completar procedencia, consentimiento ético, sesgos, criterios de inclusión/exclusión |
| Calidad del producto software | `pyproject.toml` (ruff/mypy), `.github/workflows/lint.yml`, `tests/` | ISO/IEC 25010 | Parcial | Mantenibilidad (lint), fiabilidad (tests), portabilidad (Docker) cubiertas parcialmente | Sin medición formal de atributos de calidad (usabilidad, eficiencia de desempeño) más allá de pruebas funcionales |
| Documentación de instalación/usuario | `INSTALL.md`, `docs/user_manual.md` | ISO/IEC/IEEE 26514 | Implementado | Requisitos de SO/HW/SW, instalación, solución de errores, actualización, desinstalación, manual de usuario con flujo típico y FAQ | Versión exacta de CUDA/cuDNN de la imagen Docker marcada TODO (no verificable sin acceso a la imagen) |
| Gestión de riesgos de IA | `docs/risks_and_limitations.md` | ISO/IEC 23894 | Parcial | Matriz de 11 riesgos con causa/probabilidad/impacto/evidencia/mitigación/responsable/residual | Varios riesgos con "evidencia: TODO" pendientes de confirmación del equipo (etiquetado, privacidad) |
| Robustez de redes neuronales | `docs/risks_and_limitations.md` (fila "ataques adversariales"), `docs/model_card.md` (sección "robustez") | ISO/IEC 24029 | Pendiente de información | Riesgo identificado explícitamente como no evaluado | Sin evaluación de robustez implementada; queda como trabajo futuro documentado |
| Model Cards | `docs/model_card.md` | Mitchell et al. 2019 | Parcial | Estructura completa (propósito, usos, arquitectura, entrenamiento, métricas, limitaciones) | Resultados agregados no transcritos (ver `docs/model_card.md`, evitar desactualización — referencia a `metrics_log.xlsx`) |
| Datasheets for Datasets | `docs/data_documentation.md` | Gebru et al. 2018 | Pendiente de información | Estructura completa de las 21 secciones solicitadas | Mayoría de campos de procedencia/ética marcados TODO — datos no documentados en el repositorio original |
| MLOps | `configs/`, `experiments/experiment_registry.csv`, `.github/workflows/`, `docs/experiments.md`, `docs/reproducibility.md` | Prácticas MLOps generales | Parcial | Configuración externalizada (documental), trazabilidad experimental, CI | Configuración YAML no es aún la fuente de verdad en runtime (ver ADR 0003); registro de experimentos es manual, no automatizado |
| FAIR (Findable, Accessible, Interoperable, Reusable) | `README.md`, `CITATION.cff`, `docs/reproducibility.md`, `data/README.md` | Principios FAIR | Parcial | Findable (README, CITATION), Reusable (MIT, docs) | Accessible/Interoperable limitados: datos no publicados ni con identificador persistente (DOI) |
| SPDX / SBOM | `sbom/sbom.json`, `sbom/README.md` | SPDX / CycloneDX | Implementado | SBOM CycloneDX generado y verificado (13 componentes de nivel superior), regenerado en CI | Sin SBOM de dependencias transitivas versionado (generable bajo demanda, ver `sbom/README.md`) |
| Citación de software | `CITATION.cff` | Citation File Format 1.2.0 | Parcial | Metadatos de autoría, licencia, palabras clave | Afiliación institucional y referencia de publicación marcadas TODO |
| Licencia | `LICENSE` (MIT) | — | Implementado | Texto MIT estándar, con nota de alcance explícita sobre datos (no cubiertos) | Confirmación institucional del titular de derechos (ver TODO en `LICENSE`) |
| Seguridad de la cadena de suministro | `SECURITY.md`, `.github/workflows/security.yml`, `.dockerignore`, `Dockerfile` (usuario no root) | Buenas prácticas generales | Implementado | Hallazgos verificados (sin secretos, `weights_only=True`), `pip-audit` en CI, imagen sin credenciales | Imagen base Docker de 2022 sin actualizaciones recientes (ver `docs/adr/0006`) |
| Reproducibilidad experimental | `docs/reproducibility.md`, `experiments/experiment_registry.csv` | Prácticas de ciencia reproducible | Parcial | Semillas centralizadas y verificadas, split determinista, ejemplo mínimo reproducible | Semilla no parametrizable por CLI (fija en código); determinismo GPU no estricto por defecto |

## Alcance de esta matriz

Cubre el estado al cierre de esta migración (Fase 1 + Fase 2, ver
`MIGRATION_REPORT.md`). Debe revisarse cada vez que se complete un ítem
"Pendiente de información" o se agregue/quite un artefacto relevante.
