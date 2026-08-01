# Manual de usuario

Alineado con los principios de ISO/IEC/IEEE 26514 (documentación de
usuario, instalación y operación).

## Audiencia

Investigadores/as que quieran entrenar, comparar o extender arquitecturas
1D sobre los datasets ECG/EMG soportados. Se asume familiaridad básica con
Python y línea de comandos.

## Instalación

Ver `INSTALL.md`.

## Flujo de uso típico

### 1. Entrenar una arquitectura sobre un dataset

```bash
cd src
python run_bohb.py --dataset <kitpvc|ckitpvc|auckland|bordeaux|utah2002> \
    --method <clave_en_MODEL_REGISTRY> \
    --direction <ecg2emg|emg2ecg> \
    --num-samples 15 \
    --max-epochs 50 \
    [--save-curves]
```

`<clave_en_MODEL_REGISTRY>`: cualquiera de las 36 claves registradas en
`src/models/register.py` (p. ej. `mlp`, `resnet`, `cnngru`, `patchtst`).
Ver `configs/model/default.yaml` para el listado agrupado por familia.

### 2. Ejecutar un barrido completo

```bash
cd src
DATASETS="ckitpvc auckland" NUM_SAMPLES=15 MAX_EPOCHS=50 bash run_experiments.sh
```

Variables de entorno soportadas: `DATASETS`, `METHODS`, `DIRECTION`,
`NUM_SAMPLES`, `MAX_EPOCHS`, `SAVE_CURVES`, `SKIP_DONE` (ver comentarios
en `src/run_experiments.sh`).

### 3. Verificar que todo funciona (smoke test)

```bash
cd src
DATASET=auckland bash run_smoke_test.sh
```

Ejecuta las 36 arquitecturas con 2 trials/5 épocas sobre un dataset
pequeño — detecta errores de carga/forward sin entrenar en serio.

### 4. Revisar resultados

- `src/metrics_log.xlsx`: métricas por ejecución.
- `src/checkpoints/<dataset>_<method>_best.pt` +
  `<dataset>_<method>_config.json`: mejor modelo y su configuración.
- `src/loss_curves/`: curvas de entrenamiento (si se usó `--save-curves`).

### 5. Registrar el experimento para trazabilidad

Ver `docs/experiments.md` y añadir una fila en
`experiments/experiment_registry.csv`.

## Añadir una arquitectura nueva

Ver `CONTRIBUTING.md`, sección "Añadir una nueva arquitectura de modelo".

## Preguntas frecuentes

**¿Por qué mi comando falla con `ModuleNotFoundError: No module named
'models'`?** Ejecuta `run_bohb.py` desde dentro de `src/`, no desde la
raíz del repositorio (ver `INSTALL.md`, tabla de errores comunes).

**¿Cómo cambio la semilla?** Actualmente `GLOBAL_SEED=42` está fijo en
`src/run_bohb.py`; no es un parámetro de CLI (ver
`docs/reproducibility.md`, TODO correspondiente).

**¿Cómo hago inferencia con un checkpoint ya entrenado?** No hay un
comando dedicado todavía (ver `README.md`, sección 14, y
`MIGRATION_REPORT.md`).
