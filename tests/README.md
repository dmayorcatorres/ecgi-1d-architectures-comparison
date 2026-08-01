# Pruebas

Organización:

- `unit/`: config, semillas, split, adaptadores de espacio de búsqueda —
  sin GPU, sin datos reales.
- `data/`: chequeos de calidad de datos (`assert_no_nan_batch`) con
  tensores sintéticos.
- `model/`: registro de modelos (`MODEL_REGISTRY`/`SPACE_REGISTRY`) y
  *forward pass* con tensores sintéticos, uno por familia (CNN, RNN,
  híbrida, atención).
- `integration/`: interacción entre `ModelFactory` + `MODEL_REGISTRY` +
  `SPACE_REGISTRY` + `hpo/adapters.py`.
- `smoke/`: ciclo mínimo end-to-end marcado `@pytest.mark.smoke`, más
  lento, no se ejecuta en cada `pytest` por defecto.

Todas las pruebas usan **tensores sintéticos**, nunca los datasets reales
(`.mat`), para poder ejecutarse en CI sin GPU ni datos clínicos.

## Ejecutar

```bash
pytest tests/unit tests/data tests/model tests/integration -q
pytest tests/smoke -q -m smoke
```

## Convenciones

- Pruebas que requieren GPU: marcarlas `@pytest.mark.gpu`; se omiten
  automáticamente si `torch.cuda.is_available()` es `False`
  (`tests/conftest.py`).
- No se agregan pruebas que "siempre pasan" sin aserciones reales sobre
  comportamiento (p. ej. nunca `assert True`).
- Los umbrales numéricos (tolerancias) se justifican en el propio test
  cuando no son triviales.
