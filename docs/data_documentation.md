# Documentación de datos

Alineado con los principios de ISO/IEC 5259 (calidad de datos para
analítica/ML) y la práctica de *Datasheets for Datasets*. No implica
certificación. Ver también `data/README.md` para la ficha resumida.

## Origen

```text
TODO: el repositorio no documenta el origen exacto (institución,
protocolo de adquisición, equipo de registro) de los 5 datasets. Nombres
verificados en código: auckland, bordeaux, ckitpvc, kitpvc, utah2002. Los
nombres sugieren procedencia geográfica/institucional (Auckland, Burdeos,
Utah) pero esto NO debe tomarse como confirmación sin que el equipo lo
valide explícitamente.
```

## Propietario

TODO: completar (institución/laboratorio propietario de cada dataset;
pueden ser distintos entre sí dado que provienen de fuentes con nombres
geográficos distintos).

## Licencia

TODO: completar. No se encontró archivo de licencia de datos ni nota de
autorización de uso en el repositorio original.

## Propósito

Entrenar y evaluar modelos de regresión secuencia-a-secuencia entre señal
ECG y señal EMG (o su inversa), en el contexto de electrocardiografía por
imágenes (ECGi).

## Descripción de variables

Verificado en `src/data/datamodule.py` (`ECGDataModule`): cada archivo
`.mat` contiene, al menos, las claves:

| Clave `.mat` | Uso | Rol según `direction` |
|---|---|---|
| `ecg` (`key_bspm`) | Señal ECG (o BSPM — *body surface potential mapping*) | `x` si `emg2ecg`, `y` si `ecg2emg` |
| `emg` (`key_x`) | Señal EMG | `x` si `ecg2emg`, `y` si `emg2ecg` |
| `tag` (`key_y`) | Etiqueta de clase, usada para el split estratificado | — |

```text
TODO: completar unidades físicas de ecg/emg (mV, normalizado, etc.),
frecuencia de muestreo original y significado de las clases en `tag` — no
verificables desde el código sin acceso a los archivos .mat reales ni a
documentación asociada.
```

## Formatos

MATLAB `.mat`, cargados con la librería de lectura de `.mat` de SciPy/h5py
(según versión del archivo) desde `src/data/preprocessing.py`, y
cacheados como tensores PyTorch (`.pt`, `torch.save`/`torch.load` con
`weights_only=True`) para acelerar cargas repetidas
(`src/data/auckland/raw/auckland.T256.labels.<hash>.pt`, etc., visibles en
el árbol de trabajo).

## Número de muestras

```text
TODO: completar por dataset (N por dataset se imprime en tiempo de
ejecución por run_bohb.py: "Dataset cargado: {N} muestras", pero no está
documentado de forma estática en el repositorio).
```

## Procedimiento de adquisición

TODO: completar (protocolo clínico/experimental de registro de ECG/EMG).

## Criterios de inclusión y exclusión

TODO: completar.

## Etiquetado

La clave `tag` se usa como variable de estratificación del split
(`utils/split.py`). TODO: documentar el esquema de etiquetas (número de
clases, significado clínico) y el proceso de etiquetado (manual/automático,
quién etiquetó, acuerdo entre anotadores si aplica).

## Preprocesamiento

- Recorte/ajuste a `time_steps=256` (ver `ECGDataModule(time_steps=256)`).
- Cacheo en `.pt` vía `load_mat_cached()` para evitar re-parsear el `.mat`
  en cada ejecución.
- Verificación de NaN/Inf por batch (`utils/data_checks.py::
  assert_no_nan_batch`), usada especialmente para el dataset `utah2002`
  según comentario en el propio código (`dataset_quick_scan`, docstring:
  "Útil para datasets problemáticos (p.ej., utah2002)").

## Particiones de entrenamiento, validación y prueba

Split **60/20/20**, estratificado por `tag`, con
`stratified_split_indices_approx()` (`src/utils/split.py`), semilla fija
`GLOBAL_SEED=42`. Ver `configs/data/default.yaml`.

## Prevención de fuga de datos

```text
TODO — RIESGO IDENTIFICADO: el split estratificado opera sobre el índice
de muestra/ventana (labels por muestra), no se verificó en el código un
mecanismo de agrupación por sujeto/paciente. Si un mismo sujeto aporta
múltiples ventanas/muestras, podría haber fuga de información entre
train/val/test. El equipo debe confirmar si esto aplica a estos datasets
y, si es así, documentar la mitigación o migrar a un split por sujeto. Ver
docs/risks_and_limitations.md.
```

## Valores faltantes

Se detectan (no se imputan automáticamente) vía `assert_no_nan_batch()`,
que **lanza una excepción** si encuentra NaN/Inf en un batch — es decir,
el pipeline actual falla explícitamente ante datos faltantes/corruptos en
lugar de imputarlos silenciosamente (comportamiento verificado, no
inferido).

## Sesgos conocidos

TODO: completar (representatividad demográfica/clínica de cada dataset,
balance de clases en `tag`).

## Limitaciones

TODO: completar junto con el equipo clínico/de dominio.

## Privacidad y anonimización

```text
TODO: sin evidencia en el repositorio de un proceso de anonimización
documentado. Dado que son señales electrofisiológicas humanas, el equipo
debe confirmar el estado de consentimiento informado, aprobación de
comité de ética y anonimización antes de cualquier distribución externa
de estos datos. No asumir cumplimiento sin verificación explícita.
```

## Versión

TODO: no hay versionado explícito de los datasets más allá del hash en el
nombre de los archivos `.pt` cacheados (p. ej.
`auckland.T256.labels.776eff7c.pt`), que corresponde a un hash de
contenido/parámetros de cacheo, no a una versión semántica del dataset en
sí.

## Hash o mecanismo de integridad

Los archivos cacheados (`<dataset>.T256.labels.<hash8>.pt`) incluyen un
hash corto en el nombre — verificado en el árbol de archivos del
repositorio — que actúa como mecanismo de invalidación de caché ante
cambios de `time_steps`/parámetros, no como checksum de integridad
publicado. TODO: si se requiere verificación de integridad formal, añadir
un `SHA256SUMS` junto a los `.mat` originales.

## Instrucciones de descarga o generación

Ver `data/README.md` y `INSTALL.md`, sección "Descarga de datos" (marcada
como TODO — no había un mecanismo de descarga automatizado en el
repositorio original).
