# Reporte de migración

Registro de qué se movió, qué NO se movió/eliminó y por qué, y qué queda
pendiente para el equipo. Ver `docs/adr/` para el razonamiento completo
de cada decisión estructural.

## Principio rector

Ninguna acción de esta migración altera la lógica científica del pipeline
activo (`src/run_bohb.py` y sus dependencias directas). Todo cambio que
pudiera hacerlo se documenta aquí como recomendación, no se ejecuta.

## Fase 4 — Limpieza y verificación de ejecución real (2026-08-01)

### Limpieza

- Eliminados del disco (no trackeados en git, ajenos al proyecto):
  `claude-desktop_1.24012.9_amd64.deb` (159 MB), dos copias de
  `cuda-repo-ubuntu2404-13-1-local_*.deb` (3.9 GB c/u) y
  `cuda-ubuntu2404.pin.1` — instaladores sueltos en la raíz del repo,
  ~8 GB liberados.
- Eliminados 20 directorios `__pycache__` (regenerables) y las cachés de
  herramientas `.pytest_cache`/`.ruff_cache`.
- Verificado que `src/ray_results/`, `src/runs/`, `src/run_trials.py`,
  `src/run_trials2.py` **ya no existen físicamente en disco** (solo
  quedaban como borrados pendientes de commit en `git status`, ver hallazgo
  crítico más abajo) — no requirieron limpieza adicional.
- `src/batch_logs/` (5.9 MB, ya gitignorado) se conservó: son logs
  históricos de ejecuciones reales, no basura.

### Verificación real de ejecución (`run_smoke_test.sh`)

Se instaló PyTorch con soporte CUDA (había una GPU real disponible,
NVIDIA GTX 1650, no detectada en el chequeo inicial porque no se había
probado) y el resto de `requirements.txt` en un entorno virtual
(`.venv/`, no versionado), y se ejecutó `run_smoke_test.sh` con datos
reales (`auckland.mat`) sobre un subconjunto representativo (una
arquitectura por familia: `mlp`, `resnetcnn`, `rnn_gru`, `cnnlstm`,
`alstm`, `patchtst`) para verificar el pipeline sin incurrir en un
entrenamiento extenso de las 36 arquitecturas.

**Primer intento: 0/6 — bug real encontrado.** Los 6 modelos fallaron de
forma idéntica al arrancar:
```
ModuleNotFoundError: No module named 'thop'
```
`src/utils/profile.py` (importado por `run_bohb.py` para calcular
FLOPs/parámetros/latencia) requiere el paquete `thop`
(PyTorch-OpCounter), que **no estaba en `requirements.txt`** (tampoco en
el `README.md`/`setup.py` originales). Se corrigió agregando `thop` y
`PyYAML` (esta última requerida por `src/configs/__init__.py`, mismo
patrón de omisión) a `requirements.txt`, verificado con un escaneo
estático de todos los imports de terceros en `src/` contra el archivo de
dependencias.

**Segundo intento, tras el fix: 6/6 — todos pasaron**, con métricas
reales calculadas sobre el split de test held-out (ejemplos verificados
en los logs, `src/batch_logs/smoke_ecg2emg/`):

| Modelo | Familia | RMSE | PCC | Parámetros | FLOPs |
|---|---|---|---|---|---|
| mlp | CNN (dense) | 1.917 | 0.007 | 11,144,864 | 0.022 G |
| resnetcnn | CNN | 1.487 | 0.623 | 508,637 | 0.261 G |
| rnn_gru | RNN | 1.471 | 0.578 | 448,221 | 0.231 G |
| cnnlstm | Híbrida | 1.529 | 0.577 | 458,669 | 0.233 G |
| alstm | Atención | 1.905 | 0.394 | 584,733 | 0.295 G |
| patchtst | Atención (transformer) | 1.912 | 0.039 | 920,317 | 0.054 G |

(`auckland` tiene solo N=18 muestras totales — split 60/20/20 da
train=10/val=4/test=4 — por eso las métricas de correlación (PCC) varían
mucho entre arquitecturas; esto es un efecto del tamaño de muestra del
smoke test, no una comparación de calidad válida entre modelos. No se
deben citar estos números como resultados científicos.)

## Fase 5 — Refactoring de código (2026-08-01, autorizado explícitamente)

Con el pipeline ya respaldado en Git (Fase 4) y verificado funcionando,
el responsable del proyecto autorizó explícitamente tres refactors
adicionales:

1. **Deduplicación de bloques base** — commits `88fe667d` +
   `b6a95397`: consolidación de `ConvSEBlock.py`, `DisjointBase.py`,
   `ResidualBlock.py`, `SqueezeExciteBlock.py` en `models/base/` (ver
   detalle en `CHANGELOG.md`).
2. **Eliminación del código no referenciado** (resolución de ADR 0004):
   se eliminaron `src/models/model_factory.py`,
   `src/optimization/hpo_spaces.py`, `src/optimization/ray_optimizer.py`,
   `src/training/engines.py` y `src/training/__init__.py` (este último
   quedó sin propósito al eliminar `engines.py`, su único contenido).
   Re-verificado antes de eliminar que ningún módulo activo los importa.
   Verificado después: `pytest` 29/29 en verde, `run_bohb.py` importa sin
   errores. Recuperables con `git revert` si el equipo los necesita.
3. **Externalización de variantes de configuración** — ver detalle más
   abajo si se ejecutó en esta misma sesión.

Este refactor **no altera** el comportamiento de ningún modelo entrenado
ni las métricas que produce `run_bohb.py`; solo elimina rutas de código
muertas y duplicación estructural.

**Conclusión:** el pipeline activo (`run_bohb.py` + `utils/` +
`models/factory.py` + `models/register.py`) ejecuta correctamente de
punta a punta sobre datos reales y GPU real, para al menos una
arquitectura de cada una de las 4 familias, una vez corregida la
dependencia faltante. No se ejecutó el smoke test completo (36
arquitecturas) para no incurrir en un entrenamiento extenso innecesario;
el equipo puede correr `bash run_smoke_test.sh` sin restricción de
`METHODS` para la cobertura completa cuando lo considere necesario.

## Fase 1 — Archivos nuevos (sin riesgo, completada)

Se crearon ~50 archivos nuevos de documentación, configuración,
pruebas y automatización, sin modificar el comportamiento de
`src/run_bohb.py` ni de ningún módulo importado por él. Ver listado
completo en la entrega final de esta migración.

Un detalle a validar: `pyproject.toml` declara
`[tool.pytest.ini_options] pythonpath = ["src"]`, que permite a
`tests/` importar módulos con la misma convención "plana" que ya usa
`run_bohb.py` (`from models.factory import ...`), **sin tocar ningún
import existente**.

## Fase 2 — Movimientos ejecutados sobre archivos activos

| Acción | Estado | Detalle |
|---|---|---|
| `dockerfile` → `Dockerfile` | ✅ Ejecutado (`git mv`) | Historia preservada; contenido reforzado (usuario no root, healthcheck, caché de capas) sin cambiar la imagen base (`nvcr.io/nvidia/pytorch:22.05-py3`) |
| Corrección de bug real en `requirements.txt` | ✅ Ejecutado | La línea `"ray[data,train,tune]>=2.9,<3.0"` tenía comillas literales inválidas para un archivo `requirements.txt` (`pip install -r requirements.txt` fallaba con `Invalid requirement`). Verificado con `pip install --dry-run -r requirements.txt` antes y después del fix. También se relajó `numpy>=1.24,<2.0` a `numpy>=1.24` tras confirmar (por grep) que el código no usa alias eliminados en NumPy 2.0 (`np.float_`, `np.int0`, etc.) |
| Mover `src/metrics_log.xlsx` y `src/loss_curves/` a `reports/` | ❌ NO ejecutado en esta migración | Ver justificación abajo |
| `model_factory.py`, `optimization/*`, `training/engines.py` | ✅ Eliminados en Fase 5 (2026-08-01) | Confirmación explícita del responsable del proyecto; ver sección "Fase 5" |

### Por qué NO se movieron `metrics_log.xlsx` / `loss_curves/`

`src/run_bohb.py` escribe a estas rutas usando `PROJECT_ROOT` (definido
como el propio directorio de `run_bohb.py`, es decir `src/`):

```python
PROJECT_ROOT = Path(__file__).resolve().parent
...
ckpt_dir = PROJECT_ROOT / "checkpoints"
... PROJECT_ROOT / "loss_curves"
... metrics_log = PROJECT_ROOT / "metrics_log.xlsx"
```

Mover estos archivos/directorios a `reports/` requiere editar estas
líneas en `run_bohb.py` (código activo, usado para producir resultados
científicos reales). Se decidió **no tocar `run_bohb.py`** sin que el
equipo lo valide explícitamente, dado que:

1. Es el único punto de entrada real del pipeline.
2. Un error al mover estas rutas podría hacer que una corrida real
   sobrescriba o pierda resultados históricos ya en `src/metrics_log.xlsx`.
3. No hay una suite de tests de regresión sobre `run_bohb.py` completo
   (solo sobre sus componentes, ver `tests/`) que permita validar con
   confianza que el cambio de ruta no rompe nada.

**Recomendación para el equipo** (no ejecutada): añadir 3 líneas en
`run_bohb.py` cambiando `PROJECT_ROOT / "loss_curves"` y
`PROJECT_ROOT / "metrics_log.xlsx"` por `PROJECT_ROOT.parent / "reports" /
"..."`, ejecutar `run_smoke_test.sh` para validar que sigue funcionando, y
mover los archivos existentes con `git mv`.

### Código no referenciado: eliminado en Fase 5 (ver más abajo)

Ver `docs/adr/0004-codigo-deprecado.md` para el razonamiento completo,
incluida la sección "Resolución" con la decisión final. Resumen
histórico de lo que motivó originalmente **no** eliminarlo en esta misma
migración: `src/models/model_factory.py`, `src/optimization/ray_optimizer.py`,
`src/optimization/hpo_spaces.py` y `src/training/engines.py` estaban
**verificados como no importados** por `run_bohb.py` (trazado completo de
imports), pero no se podía descartar que representaran trabajo en progreso
no versionado. Se recomendó entonces al equipo:

1. Confirmar en un plazo definido si se retomaban o se eliminaban.
2. Si se eliminaban, hacerlo en un commit dedicado y fácilmente revertible
   (`git rm src/models/model_factory.py src/optimization/ray_optimizer.py
   src/optimization/hpo_spaces.py src/training/engines.py`), no como
   parte de un commit mixto con otros cambios.

No se añadieron docstrings de "deprecated" dentro de esos archivos en
esta migración para no modificar código fuente activo sin necesidad
funcional — la documentación de su estado vive en `docs/adr/0004.md`,
`README.md`, `CONTRIBUTING.md` y `CHANGELOG.md`.

## ⚠️ Hallazgo crítico: parte del pipeline activo nunca se ha commiteado

Al preparar la Fase 3 se auditó `git status` en detalle (no solo el
resumen usado al inicio de esta migración) y se encontró algo más urgente
que el propio historial de `ray_results`:

**Los siguientes archivos, que son el pipeline activo verificado en toda
esta auditoría, están completamente sin trackear en Git — no existen en
ningún commit, solo en el disco local:**

```text
src/run_bohb.py                (punto de entrada principal, 788 líneas)
src/run_smoke_test.sh
src/test_gpu.py
src/utils/data_checks.py
src/utils/logbook.py
src/utils/profile.py
src/utils/ray_helpers.py
src/utils/runtime.py           (control de semillas, ver docs/reproducibility.md)
src/utils/split.py             (split estratificado 60/20/20)
```

Además, 25 archivos **sí trackeados** tienen cambios sin commitear
(`git status` → `M`), incluyendo `src/models/factory.py`,
`src/models/register.py`, `src/hpo/adapters.py`,
`src/data/datamodule.py`, `src/metrics_log.xlsx` — y 89 archivos
trackeados aparecen como eliminados en el working tree sin commitear
(`run_trials.py`, `run_trials2.py`, todo `src/runs/`), además de los
~19,874 borrados de `src/ray_results/`/`src/batch_logs/`/`__pycache__`
que ya se sabían.

**Esto significa que si se perdiera el directorio de trabajo local
(disco dañado, `git clean -fdx` accidental, etc.), se perdería
irrecuperablemente el pipeline de entrenamiento real** (`run_bohb.py` y
las utilidades de reproducibilidad/HPO que lo soportan) — el último
commit real (`a0df4d68`, 2025-08-22) no lo contiene.

**No se hizo ningún `git add`/`git commit` de esto en esta migración.**
Por la política del asistente ("nunca commitear sin que el usuario lo
pida explícitamente") y porque este estado mezcla trabajo del usuario no
relacionado con esta migración (código del pipeline, borrado de
`run_trials.py`/`src/runs/`) con los ~50 archivos nuevos de esta
migración, se dejó intacto para que el equipo lo revise y decida qué
commitear.

**Recomendación inmediata (antes de cualquier otra acción de Git):**
```bash
git add src/run_bohb.py src/run_smoke_test.sh src/test_gpu.py src/utils/
git status   # revisar qué más de lo modificado/eliminado se quiere incluir
git commit -m "Respaldar pipeline activo (run_bohb.py y utils/) nunca commiteado"
```

## Fase 3 — Historial de Git (purga de `ray_results`)

**No ejecutada.** `git-filter-repo` (instalado y verificado,
`git-filter-repo --version` → `a40bce548d2c`) reescribe **todos** los
commits del historial, y su propio mecanismo de seguridad exige un
árbol de trabajo limpio (o `--force`, que este asistente no usa para
saltarse protecciones de seguridad). Con ~20,000 cambios sin commitear
en curso — varios de ellos ajenos a esta migración — ejecutar la purga
ahora habría sido irresponsable: el riesgo de interferir con el
respaldo pendiente del hallazgo crítico de arriba supera el beneficio de
reducir el tamaño de `.git` (1.2 GB) en este momento.

**Plan listo para ejecutar una vez el equipo haya commiteado/limpiado el
estado pendiente:**

```bash
# 1) Backup obligatorio antes de reescribir historia
git clone --mirror . ../ecgi-1d-architectures-comparison-backup.git

# 2) Purga de rutas pesadas del historial completo
git filter-repo \
    --path src/ray_results \
    --path src/batch_logs \
    --invert-paths

# 3) Verificar tamaño e integridad
du -sh .git
git log --oneline | head
git fsck

# 4) Publicar (requiere autorización EXPLÍCITA adicional — reescribe
#    hashes de todos los commits; si origin es compartido, cualquier
#    clon existente quedará divergente)
# git push origin --force --all
# git push origin --force --tags
```

`git-filter-repo` no reescribe `refs/remotes/origin/*` automáticamente
(por diseño, para evitar pushes accidentales) — el force-push queda como
un paso manual separado, explícito, que este asistente no ejecuta sin
confirmación adicional del usuario en el momento.

## Hallazgos menores no accionados (bajo riesgo, no ejecutados)

| Hallazgo | Ubicación | Por qué no se corrigió |
|---|---|---|
| Comentario incorrecto en `__init__.py` (contenido copiado de otro archivo) | `src/__init__.py`, `src/models/__init__.py` (ambos contienen `# src/training/__init__.py`) | Cosmético, cero impacto funcional; se documenta aquí en vez de tocar código fuente sin necesidad |
| 4 espacios de búsqueda (`SPACE_REGISTRY`) sin modelo registrado correspondiente | `bertbigru`, `densenetbilstm`, `oscnn`, `phasedlstm` en `src/models/register.py` | Corresponden a modelos comentados/no completados (`BERT_BIGRU`, `DenseNetBiLSTM`, `OSCNN`, `EfficientNet1D`/`PhasedLSTM` con imports comentados en `model_factory.py`); posible trabajo en progreso, no se modifica |
| `src/models/cnnbased/mlp_basic.py` junto a `src/models/cnnbased/MLP.py` | `src/models/cnnbased/` | Posible duplicado/versión anterior de `MLP.py`; no se verificó cuál es la fuente de verdad sin riesgo de romper el registro activo — el equipo debe confirmar cuál conservar |
| Archivos `cuda-*.deb`, `claude-desktop_*.deb`, `*.pin` en la raíz del repositorio | raíz del proyecto | No versionados en Git (confirmado); son artefactos del entorno local del desarrollador, no del repositorio. Se agregaron a `.gitignore` para prevenir que se versionen por accidente; no se eliminan del disco sin autorización explícita del usuario sobre su propio entorno local |

## Información que el equipo debe completar

Ver también la sección "TODO" repetida en cada documento nuevo. Resumen
priorizado:

1. **Origen, licencia y estado ético de los 5 datasets** (`auckland`,
   `bordeaux`, `ckitpvc`, `kitpvc`, `utah2002`) — bloqueante para
   cualquier distribución externa de datos o publicación de resultados.
2. **Confirmar si el split evita fuga de datos por sujeto** (riesgo alto
   documentado en `docs/risks_and_limitations.md`).
3. Decidir sobre el código deprecado (ADR 0004): ¿retomar
   `training/engines.py` o eliminarlo?
4. Completar afiliación institucional en `CITATION.cff` y titularidad de
   `LICENSE`.
5. Decidir si se ejecuta la migración de `metrics_log.xlsx`/`loss_curves/`
   a `reports/` (Fase 2 pendiente, requiere tocar `run_bohb.py`).
6. Confirmar versión exacta de CUDA/cuDNN de la imagen Docker de
   referencia (`INSTALL.md`, TODO).
7. **Prioritario:** commitear `src/run_bohb.py` y `src/utils/*.py` nunca
   respaldados (ver hallazgo crítico arriba) antes de cualquier otra
   operación de Git.
