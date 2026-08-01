# SBOM (Software Bill of Materials)

Alineado con la práctica de SPDX/CycloneDX para inventario de
dependencias de la cadena de suministro de software.

## Generación

```bash
pip install cyclonedx-bom
cyclonedx-py requirements -i requirements.txt -o sbom/sbom.json
```

Se regenera automáticamente en cada push a `main` y en cada release
(`.github/workflows/sbom.yml`) y se publica como artefacto de CI.

## `sbom.json`

Formato **CycloneDX 1.x**, generado a partir de `requirements.txt`
(15 componentes de nivel superior verificados: torch, numpy, scipy,
pandas, scikit-learn, pyarrow, pydantic, ray[data,train,tune],
pytorch-lightning, hpbandster, ConfigSpace, openpyxl, matplotlib, thop,
PyYAML — las dos últimas agregadas tras verificar con una ejecución real
de `run_smoke_test.sh` que `thop` faltaba y rompía el arranque de
`run_bohb.py`, ver `MIGRATION_REPORT.md`). No
incluye transitivas resueltas (el generador trabaja a nivel de
`requirements.txt`, no de entorno instalado) — para un SBOM con el árbol
de dependencias completo, generar desde un entorno ya instalado:

```bash
cyclonedx-py environment -o sbom/sbom-full.json
```

TODO: el equipo debe decidir si `sbom-full.json` (con transitivas) se
versiona en el repo o se genera solo bajo demanda/en release (puede
quedar desactualizado rápido si no se automatiza su regeneración).
