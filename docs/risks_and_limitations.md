# Riesgos y limitaciones de IA

Alineado con los principios de ISO/IEC 23894 (gestión de riesgos de IA) e
ISO/IEC 24029 (robustez de redes neuronales). Matriz basada en hallazgos
verificados en el código y datos disponibles en esta auditoría; los
riesgos marcados "evidencia: TODO" requieren confirmación del equipo, no
se afirma su materialización.

| Riesgo | Causa | Probabilidad | Impacto | Evidencia | Mitigación | Responsable | Riesgo residual |
|---|---|---|---|---|---|---|---|
| Fuga de datos train/test por sujeto | `stratified_split_indices_approx()` particiona por muestra/ventana, no se verificó agrupación por sujeto | Media | Alto (infla métricas reportadas) | Código de `src/utils/split.py` revisado; sin mecanismo de `group_id` visible | Confirmar con el equipo si cada muestra proviene de un sujeto distinto; si no, migrar a `GroupShuffleSplit`-equivalente | Equipo de datos | Alto hasta confirmación |
| Edición del código muerto en vez del activo | Dos factories (`factory.py` vs `model_factory.py`) y dos motores de entrenamiento coexisten | Media | Medio (esfuerzo perdido, confusión) | Verificado por trazado de imports desde `run_bohb.py` | Marcado como deprecado + `MIGRATION_REPORT.md` + `CONTRIBUTING.md` | Mantenedores | Bajo tras esta documentación |
| Determinismo no garantizado en GPU | `cudnn.benchmark=True`, TF32 habilitado por defecto | Alta | Bajo-Medio (variación menor entre corridas) | `src/utils/runtime.py` | Documentado en `docs/reproducibility.md`; activar `strict=True` si se requiere | Quien ejecuta el experimento | Medio (trade-off velocidad/determinismo) |
| Datos fuera de distribución (*distribution shift*) | 5 datasets de origen distinto (`auckland`, `bordeaux`, `ckitpvc`, `kitpvc`, `utah2002`); un modelo entrenado en uno puede no generalizar a otro | Media-Alta | Alto si se usa fuera del dataset de entrenamiento | No hay evaluación cruzada entre datasets documentada en el repo | Evaluar explícitamente generalización cross-dataset antes de cualquier uso fuera del dataset de origen | Equipo de investigación | Alto sin evaluación cruzada |
| Ataques adversariales / robustez | Sin evaluación de robustez frente a perturbaciones en la señal de entrada | Baja (no es el foco de un pipeline de investigación offline) | Medio si se despliega en producción | Sin evidencia de pruebas de robustez en el repo | No aplica mientras no haya despliegue; evaluar si se planea servir el modelo | Equipo de investigación | Bajo en el uso actual (offline) |
| Incertidumbre no cuantificada | Predicciones puntuales (single-point), sin intervalos de confianza | Alta (es el diseño actual) | Medio (usuarios podrían sobre-confiar en una predicción puntual) | `ECGMetricsCalculator` no reporta incertidumbre | Documentado en `docs/model_card.md`; considerar ensembles/MC-dropout a futuro | Equipo de investigación | Medio |
| Degradación del modelo en el tiempo | No aplica a un pipeline offline sin servicio desplegado | N/A | N/A | No hay servicio en producción | N/A | N/A | N/A (revisar si se despliega inferencia) |
| Errores de etiquetado | Etiquetas (`tag`) usadas para estratificar el split; sin documentación del proceso de etiquetado | Evidencia: TODO | Medio-Alto si el etiquetado tiene errores sistemáticos | No verificable desde el código | El equipo debe documentar el proceso de etiquetado (`docs/data_documentation.md`) | Equipo de datos | Desconocido hasta documentación |
| Uso fuera del alcance previsto (clínico/diagnóstico) | El modelo no está validado clínicamente | Media (riesgo si se comunica mal el propósito del proyecto) | Alto si ocurre | Sin evidencia de validación clínica en el repo | Explícito en `docs/model_card.md`, sección "usos no recomendados" | Responsable del proyecto | Bajo si se respeta la documentación |
| Riesgos de privacidad | Señales electrofisiológicas humanas; estado de anonimización no documentado | Evidencia: TODO | Alto si los datos no están anonimizados y se distribuyen | No verificable desde el código | El equipo debe confirmar consentimiento informado/aprobación ética antes de cualquier distribución (ver `docs/data_documentation.md`) | Responsable del proyecto / comité de ética | Alto hasta confirmación |
| Dependencia de hardware/software específico | Imagen Docker fijada a `nvcr.io/nvidia/pytorch:22.05-py3` (2022), sin actualizaciones de seguridad recientes | Alta | Medio (vulnerabilidades no parcheadas en la imagen base) | `Dockerfile` original | Evaluar migración a una imagen soportada (ver `docs/adr/0006`); mientras tanto, aislar el contenedor de redes no confiables | Mantenedores / DevOps | Medio hasta actualización |
| Sobreajuste por HPO extensivo | BOHB explora múltiples configuraciones sobre el mismo `val` split | Media | Medio (métricas de val optimistas respecto a test real) | Diseño estándar de HPO; mitigado parcialmente por evaluación final en `test` held-out separado | Ya mitigado por el split 60/20/20 con test held-out no usado en HPO | Equipo de investigación | Bajo (mitigación ya presente en el diseño) |

## Limitaciones generales del sistema

- Sin script de inferencia independiente (ver `README.md`, sección 14).
- Sin versionado semántico de datasets/checkpoints más allá de hashes de
  caché (ver `docs/data_documentation.md`).
- Sin CI previo a esta migración (ver `.github/workflows/`).
- Cobertura de pruebas automatizadas históricamente nula (un solo smoke
  test manual, `src/test_gpu.py`); se añade una suite inicial en esta
  migración (`tests/`), pero no cubre las 36 arquitecturas exhaustivamente
  (ver `docs/reproducibility.md` y `MIGRATION_REPORT.md`).

## Revisión de esta matriz

Se recomienda revisar esta matriz cada vez que se incorpore un dataset
nuevo, se cambie el mecanismo de split, o se planee cualquier forma de
despliegue más allá del uso batch de investigación actual.
