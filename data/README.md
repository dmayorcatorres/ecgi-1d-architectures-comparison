# Datos

Este directorio **no contiene datos** (ni debe contenerlos). Los datasets
reales que consume el pipeline viven en `src/data/<dataset>/raw/*.mat` (o
`src/data/kitpvc/kitpvc.mat` para `kitpvc`, excepción sin subcarpeta
`raw/`), y **no están versionados en Git** (`.gitignore`).

Este archivo actúa como ficha resumida (*datasheet*). Ver
`docs/data_documentation.md` para el detalle completo alineado con
ISO/IEC 5259.

## Datasets soportados

| Dataset | Ruta esperada | ¿Verificado en código? |
|---|---|---|
| `kitpvc` | `src/data/kitpvc/kitpvc.mat` | Sí (`run_bohb.py::_datasets`) |
| `ckitpvc` | `src/data/ckitpvc/raw/ckitpvc.mat` | Sí |
| `auckland` | `src/data/auckland/raw/auckland.mat` | Sí |
| `bordeaux` | `src/data/bordeaux/raw/bordeaux.mat` | Sí |
| `utah2002` | `src/data/utah2002/raw/utah2002.mat` | Sí |

## Formato esperado

Cada `.mat` debe contener las claves `ecg`, `emg`, `tag` (ver
`src/data/datamodule.py::ECGDataModule`). Al cargarse, se cachea
automáticamente como `.pt` (`<dataset>.T256.labels.<hash>.pt`) para
acelerar cargas posteriores.

## Cómo obtener los datos

```text
TODO: no existe en este repositorio un mecanismo de descarga automatizada
ni documentación del origen/licencia de estos 5 datasets. El equipo debe
completar esta sección (fuente, contacto, licencia, procedimiento de
solicitud de acceso) — ver docs/data_documentation.md para el detalle de
lo que falta.
```

## Por qué no están en Git

- Tamaño: cientos de MB por dataset.
- Naturaleza clínica/sensible: señales electrofisiológicas humanas (ver
  `docs/data_documentation.md`, sección "Privacidad y anonimización").

## Recomendación de almacenamiento (no implementada en esta migración)

Ver `docs/adr/0005-almacenamiento-datos-y-modelos.md`. Alternativas
razonables si el equipo decide versionar metadatos/muestras pequeñas:
Git LFS, DVC apuntando a almacenamiento institucional, o un bucket
privado con control de acceso. No se introduce ninguna de estas
herramientas sin decisión explícita del equipo.
