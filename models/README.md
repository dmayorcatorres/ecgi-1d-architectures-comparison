# Modelos

Este directorio documenta dónde y cómo se generan los checkpoints
entrenados. Los pesos reales **no se versionan en Git**.

## Ubicación real de los checkpoints

Los checkpoints entrenados por `src/run_bohb.py` se guardan en
`src/checkpoints/`, con el patrón:

```text
src/checkpoints/<dataset>_<variante>_best.pt
src/checkpoints/<dataset>_<variante>_config.json
```

Por ejemplo: `src/checkpoints/auckland_resnetcnn_best.pt` +
`src/checkpoints/auckland_resnetcnn_config.json`.

`models/checkpoints/` y `models/exported/` (estructura objetivo sugerida)
**no se crean como carpetas vacías** en esta migración — se documenta
aquí la ubicación real (`src/checkpoints/`) en vez de mover ~745 MB de
binarios sin necesidad funcional inmediata. Ver
`docs/adr/0005-almacenamiento-datos-y-modelos.md`.

## Carga segura de un checkpoint

```python
import torch
ckpt = torch.load("src/checkpoints/auckland_resnetcnn_best.pt",
                   map_location="cpu", weights_only=True)
```

`weights_only=True` es obligatorio por política de seguridad (ver
`SECURITY.md`) — nunca cargar checkpoints de origen no confiable sin este
flag.

## Ficha por modelo

Ver `docs/model_card.md` para la tarjeta de modelo a nivel de sistema
(36 arquitecturas). Los `*_config.json` junto a cada `.pt` documentan los
hiperparámetros específicos de esa ejecución (arquitectura + dataset).

## Procedencia

Todos los checkpoints se generan localmente por este mismo repositorio
(`run_bohb.py`); no se consumen pesos preentrenados de terceros (ver
`SECURITY.md`, sección "Confianza y procedencia de pesos preentrenados").
