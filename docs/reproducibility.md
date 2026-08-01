# Reproducibilidad

Alineado con los principios de FAIR (Findable, Accessible, Interoperable,
Reusable) y MLOps aplicados a experimentación reproducible.

## Control de semillas

Centralizado en `src/utils/runtime.py::set_seed(seed)`:

```python
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

Usado con `GLOBAL_SEED = 42` (`src/run_bohb.py`), aplicado:

- Al split de datos (`stratified_split_indices_approx(labels, seed=GLOBAL_SEED)`).
- Antes de cada trial de HPO (`set_seed(GLOBAL_SEED)`, comentario en
  código: "reproducibilidad de cada trial").
- Por worker de `DataLoader`, con semilla derivada
  (`torch.initial_seed() + worker_id`) para evitar que todos los workers
  generen la misma secuencia aleatoria — patrón correcto verificado en
  código.

## Determinismo en GPU: limitaciones conocidas

`src/utils/runtime.py::setup_torch_runtime()` expone un flag
`STRICT_DETERMINISM` (por defecto `False`):

```python
STRICT_DETERMINISM = False
torch.backends.cudnn.deterministic = strict
torch.backends.cudnn.benchmark     = not strict
torch.use_deterministic_algorithms(strict, warn_only=True)
```

Con la configuración por defecto (`strict=False`):

- `cudnn.benchmark=True` selecciona el algoritmo de convolución más
  rápido por forma de entrada, lo que **no garantiza el mismo resultado
  numérico entre ejecuciones**, incluso con la misma semilla.
- TF32 está habilitado en matmul/cuDNN (`allow_tf32=True`) cuando hay GPU,
  lo que introduce pérdida de precisión respecto a FP32 completo a cambio
  de velocidad — afecta reproducibilidad bit-a-bit, no la validez
  estadística de los resultados.

**Implicación práctica:** dos ejecuciones con la misma semilla y GPU
deberían producir métricas *muy similares* pero no necesariamente
*idénticas* bit-a-bit. Para reproducibilidad estricta (a costa de
velocidad), el equipo puede invocar `setup_torch_runtime(strict=True)`
desde el punto de entrada — esto no está activado por defecto en el
código actual.

## Identificación de una ejecución reproducible

Cada ejecución de `run_bohb.py` queda determinada por:

| Factor | Dónde se fija |
|---|---|
| Commit de Git del código | `git rev-parse HEAD` (recomendado registrar manualmente, ver `docs/experiments.md`) |
| Dataset y su versión | `--dataset`, hash del `.pt` cacheado (ver `docs/data_documentation.md`) |
| Partición de datos | Determinística dado `(labels, GLOBAL_SEED)` |
| Semilla | `GLOBAL_SEED=42` (fija en código; no es un parámetro de CLI actualmente) |
| Arquitectura e hiperparámetros | `--method`, espacio de `SPACE_REGISTRY`, mejor trial de BOHB |
| Framework y versión | Ver `requirements.txt` / imagen Docker (`Dockerfile`) |
| Hardware | GPU/CPU usados (no registrado automáticamente hoy) |

```text
TODO: GLOBAL_SEED está fijo en el código fuente (src/run_bohb.py), no es
parametrizable por CLI. Si se requiere ejecutar múltiples réplicas con
semillas distintas para análisis de varianza, esto requeriría un cambio
de código (agregar --seed a argparse) — no se implementó en esta
migración por ser un cambio funcional al pipeline activo, se deja
documentado como recomendación en MIGRATION_REPORT.md.
```

## Ejemplo mínimo de experimento reproducible

```bash
cd src
python run_bohb.py \
    --dataset kitpvc \
    --method mlp \
    --direction ecg2emg \
    --num-samples 2 \
    --max-epochs 2
```

Con `GLOBAL_SEED=42` fijo en código, el split de datos y el orden de
exploración de BOHB deberían ser reproducibles entre ejecuciones en la
misma máquina/versión de dependencias. Registrar la ejecución en
`experiments/experiment_registry.csv` (ver `docs/experiments.md`) para
trazabilidad completa.

## Qué NO está garantizado

- Reproducibilidad bit-a-bit en GPU (ver arriba).
- Reproducibilidad entre versiones distintas de PyTorch/CUDA/Ray (no
  fijadas exactamente, ver `requirements.txt`).
- Reproducibilidad si `SPACE_REGISTRY`/`MODEL_REGISTRY` cambian entre
  ejecuciones (nuevas arquitecturas, cambios de espacio de búsqueda).
