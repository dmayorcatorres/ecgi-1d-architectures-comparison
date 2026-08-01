# ADR 0003 — Estrategia de configuración

## Estado

Parcialmente ejecutado (ver "Actualización" al final). `base_config.py`
ya lee `variants` desde YAML; `run_bohb.py` (CLI) sigue sin un mecanismo
`--config` — ese trabajo futuro descrito abajo sigue pendiente.

## Contexto

La configuración del pipeline está repartida en: `src/config.py`
(`DataConfig`), `src/configs/base_config.py` (`TrainingConfig`, con
variantes de modelos seleccionadas comentando/descomentando tuplas en el
código fuente), y argumentos de CLI en `src/run_bohb.py`
(`--dataset`, `--method`, `--direction`, `--num-samples`, `--max-epochs`,
`--save-curves`). No existe un mecanismo de configuración externa (YAML)
cargado en runtime, pese a que `src/training/engines.py` importa
`hydra`/`omegaconf` de forma condicional (`HYDRA_AVAILABLE`) sin que ese
módulo esté conectado al pipeline activo (ver ADR 0004).

## Problema

¿Cómo externalizar la configuración (especialmente la selección de
variantes de modelo, hoy editada a mano en `base_config.py`) sin alterar
el comportamiento del pipeline en producción durante esta migración?

## Alternativas

1. Migrar inmediatamente a Hydra/OmegaConf, reescribiendo
   `run_bohb.py` para cargar configuración desde `configs/*.yaml`.
2. Crear `configs/{data,model,training,evaluation}/*.yaml` como
   **documentación ejecutable** de los valores reales por defecto (sin que
   el pipeline los lea todavía), dejando la migración de carga en runtime
   como trabajo futuro explícito.

## Decisión

Se adopta la alternativa 2 para esta migración.

## Justificación

La alternativa 1 requiere modificar `run_bohb.py` (788 líneas, punto de
entrada real usado en producción de resultados) y `base_config.py`,
arriesgando alterar valores por defecto que afectan directamente
resultados científicos ya reportados en `metrics_log.xlsx`. Se prioriza no
tocar la lógica científica sin validación explícita del equipo (ver
`MIGRATION_REPORT.md`).

## Consecuencias

- Positivas: los YAML en `configs/` documentan de forma legible y
  versionada los valores reales usados (extraídos y verificados contra el
  código, no inventados), sirviendo de base para la migración futura y de
  referencia para nuevos colaboradores.
- Negativas: existe riesgo de que `configs/*.yaml` diverja del código si
  alguien cambia `base_config.py`/`run_bohb.py` sin actualizar el YAML —
  se documenta explícitamente en cada archivo YAML como advertencia.
- Trabajo futuro: añadir `--config` a `run_bohb.py` para sobrescribir
  defaults vía YAML/CLI (patrón Hydra) — `run_bohb.py` en sí sigue
  usando su propio `argparse`, no se tocó en esta migración.

## Actualización (2026-08-01, refactor autorizado explícitamente)

Se ejecutó la primera mitad del "trabajo futuro" descrito arriba:
`TrainingConfig.variants` (`src/configs/base_config.py`) ya no tiene
tuplas comentadas a mano — su valor por defecto se calcula con
`field(default_factory=_load_default_variants)`, que lee
`configs/training/default.yaml::active_default_variants` y cae de
vuelta a una constante `_FALLBACK_VARIANTS` (idéntica al valor
hardcodeado anterior) si el YAML no existe o no es legible.

Se verificó explícitamente que el valor por defecto resultante es
**exactamente igual** al que `TrainingConfig()` devolvía antes del
cambio (`test_training_config_variants_come_from_yaml_and_match_fallback`
en `tests/unit/test_config.py`), y que el fallback funciona si el YAML
no está disponible (`test_training_config_falls_back_when_yaml_unreadable`).

`run_bohb.py` (el punto de entrada real, con su propio `argparse`)
**no fue modificado** — sigue sin leer `configs/*.yaml` directamente;
solo `TrainingConfig` (usado hoy únicamente por `tests/` y por quien
importe `configs.base_config` directamente, no por `run_bohb.py`) quedó
externalizado. Añadir `--config` a `run_bohb.py` sigue como trabajo
futuro.
