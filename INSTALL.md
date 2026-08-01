# Guía de instalación

Sigue los principios de documentación de instalación de
ISO/IEC/IEEE 26514. Todos los comandos son ejecutables desde la raíz del
repositorio salvo que se indique lo contrario.

## Sistemas operativos compatibles

- Linux (validado en el entorno original: Ubuntu, con imagen base
  `nvcr.io/nvidia/pytorch:22.05-py3` para entrenamiento con GPU).
- macOS / Windows: viable para desarrollo, lint y pruebas sin GPU
  (`pytest tests/unit`), no validado para entrenamiento real.

```text
TODO: el equipo debe confirmar en qué distribución/versión exacta de Linux
se ejecutaron los entrenamientos de referencia reportados en
src/metrics_log.xlsx.
```

## Versión de Python

Python **≥ 3.10** (fijado en `pyproject.toml` como `requires-python`). Los
artefactos `__pycache__/*.cpython-310.pyc` encontrados originalmente en el
repositorio sugieren que el entorno de entrenamiento de referencia usó
**Python 3.10**.

## Requisitos de hardware

| Recurso | Mínimo (pruebas/CPU) | Recomendado (entrenamiento real) |
|---|---|---|
| CPU | 2 núcleos | ≥ 6 núcleos (`ray.init(num_cpus=6)` por defecto en `run_bohb.py`) |
| RAM | 4 GB | ≥ 16 GB (datasets `.mat` cargados en memoria + Ray object store) |
| GPU | No requerida | 1× GPU NVIDIA con soporte CUDA (`ray.init(num_gpus=1)` por defecto) |
| Almacenamiento | 2 GB (código + entorno) | +Datasets (`.mat`, tamaño variable por dataset) + checkpoints (~745 MB observados históricamente para 4 datasets × 36 modelos) |

## Framework y versiones

- PyTorch ≥ 2.0 (`requirements.txt`). El contenedor Docker de referencia
  usa la imagen NGC `nvcr.io/nvidia/pytorch:22.05-py3` — TODO: el equipo
  debe confirmar la versión exacta de PyTorch/CUDA/cuDNN incluida en esa
  imagen si necesita reproducir bit-a-bit el entorno original de
  entrenamiento.
- Ray `[data,train,tune]` ≥ 2.9.
- PyTorch Lightning ≥ 2.0 (usado por `ECGDataModule`).
- HpBandSter + ConfigSpace (backend de búsqueda BOHB usado por Ray Tune).

## Compatibilidad CUDA / cuDNN

```text
TODO: no se encontró en el repositorio original una versión de CUDA/cuDNN
fijada explícitamente fuera de la imagen Docker (nvcr.io/nvidia/pytorch:
22.05-py3, que corresponde aproximadamente a CUDA 11.7 / cuDNN 8.4 según
las notas de versión públicas de NVIDIA NGC — el equipo debe verificarlo
contra la imagen real usada, no asumir esta cifra como definitiva).
```

Para entrenar sin Docker, instala el driver NVIDIA y el toolkit CUDA
compatibles con la versión de PyTorch instalada
(ver <https://pytorch.org/get-started/locally/>).

## Instalación con entorno virtual

```bash
git clone git@github.com:dmayorcatorres/cgan-ecgi.git
cd ecgi-1d-architectures-comparison

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
```

## Instalación de dependencias

```bash
# Producción / ejecución del pipeline
pip install -r requirements.txt

# Desarrollo (incluye lo anterior + pytest, ruff, mypy, pip-audit, SBOM)
pip install -r requirements-dev.txt
```

Si tu plataforma no tiene GPU, instala PyTorch CPU-only explícitamente
antes de `requirements.txt` para evitar descargar binarios CUDA
innecesarios:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Configuración de variables de entorno

```bash
cp .env.example .env
# edita .env con tus rutas/recursos locales (no se versiona)
```

## Descarga de datos

```text
TODO: el repositorio original no incluye un script de descarga
automatizada (scripts/download_data.py no existe aún) ni documenta el
origen de los 5 datasets (auckland, bordeaux, ckitpvc, kitpvc, utah2002).
Ver data/README.md. Mientras tanto, coloca manualmente los archivos .mat
en las rutas descritas en la sección "Preparación de datos" del
README.md.
```

## Descarga o ubicación de pesos

Los checkpoints entrenados (`*.pt` + `*_config.json`) se generan
localmente en `src/checkpoints/` al ejecutar `run_bohb.py`; no se
distribuyen pesos preentrenados de terceros en este proyecto (ver
`SECURITY.md`, sección de procedencia de pesos).

## Ejecución de pruebas

```bash
pytest tests/unit tests/data tests/model -q
```

Salida esperada: todas las pruebas deben pasar sin requerir GPU ni datos
reales (usan tensores sintéticos). Las pruebas marcadas `@pytest.mark.gpu`
se omiten automáticamente si `torch.cuda.is_available()` es `False`.

## Prueba rápida de entrenamiento

Requiere al menos un dataset real ubicado según la sección "Preparación de
datos". Con 2 trials y 2 épocas se ejecuta en minutos incluso en CPU:

```bash
cd src
python run_bohb.py --dataset kitpvc --method mlp \
    --num-samples 2 --max-epochs 2
```

O usa el smoke test empaquetado (36 modelos, 2 trials / 5 épocas cada
uno):

```bash
cd src
bash run_smoke_test.sh
```

## Prueba rápida de inferencia

```text
TODO: ver README.md, sección 14 (Inferencia) — no existe actualmente un
comando de inferencia independiente; queda documentado como pendiente en
MIGRATION_REPORT.md.
```

## Solución de errores comunes

| Error | Causa probable | Solución |
|---|---|---|
| `ModuleNotFoundError: No module named 'models'` | `run_bohb.py` no se ejecutó desde `src/`, o el import root no está en `sys.path` | Ejecutar siempre `python run_bohb.py ...` desde dentro de `src/` (el propio script inserta su directorio en `sys.path`), o instalar con `pythonpath` configurado como en `pyproject.toml` para pruebas |
| `FileNotFoundError: Dataset '<x>' no encontrado en ...` | El `.mat` no está en la ruta esperada | Ver sección "Preparación de datos" del `README.md` |
| `RuntimeError: CUDA out of memory` | `batch_size`/`num_gpus` de Ray excede la GPU disponible | Reducir `batch_size` en `TrainingConfig` o ajustar `ray.init(num_gpus=...)` en `run_bohb.py` |
| Ray no libera memoria entre trials | Comportamiento conocido de Ray Tune con muchos trials | Ver `utils/ray_helpers.py::spilling_config()`, ya usado por defecto en `run_bohb.py` |
| `pip install -r requirements.txt` intenta instalar una build CUDA de `torch` en una máquina sin GPU | El índice por defecto de PyPI resuelve la build CUDA | Instalar primero `torch` desde el índice CPU (ver arriba) antes de `requirements.txt` |

## Actualización

```bash
git pull
pip install -r requirements.txt --upgrade
pip install -r requirements-dev.txt --upgrade
```

Revisa `CHANGELOG.md` tras cada actualización para cambios que puedan
afectar reproducibilidad (splits, semillas, versiones de dependencias).

## Desinstalación

```bash
deactivate            # si el entorno virtual está activo
rm -rf .venv
```

Esto no elimina datos (`src/data/`), checkpoints (`src/checkpoints/`) ni
resultados (`reports/`, `src/metrics_log.xlsx`) — elimínalos manualmente
si ya no los necesitas, verificando antes que no sean el único respaldo
existente.
