# ADR 0002 — Estructura del repositorio y estrategia de import

## Estado

Aceptado para esta migración (Fase 1). Revisión futura recomendada (ver
"Consecuencias").

## Contexto

El código bajo `src/` usa imports "planos": `from models.factory import
ModelFactory`, `from config import DataConfig`, `from utils.split import
...`, etc. — es decir, trata `src/` como la raíz de import, no como un
paquete `mi_paquete.submódulo`. `run_bohb.py` refuerza esto insertando su
propio directorio en `sys.path` en tiempo de ejecución
(`sys.path.insert(0, str(PROJECT_ROOT))`, `PROJECT_ROOT = Path(__file__).
resolve().parent`).

Esta auditoría propuso originalmente una estructura estándar
`src/<project_name>/...` (paquete instalable). Verificar el código mostró
que **todos** los ~90 módulos Python bajo `src/` dependen de la
convención de import plano.

## Problema

¿Convertir `src/` en un paquete Python instalable (`src/ecgi_1d/...`,
imports `from ecgi_1d.models.factory import ...`) como parte de esta
migración, o mantener la convención de import existente?

## Alternativas

1. **Reescribir los ~90 imports** para adoptar `src/<project_name>/` como
   paquete real, instalable con `pip install -e .`.
2. **Mantener `src/` como raíz de import plana**, y resolver la necesidad
   de "paquete reproducible" con mecanismos no invasivos: `sys.path`
   (como ya hace `run_bohb.py`) y `pythonpath = ["src"]` en
   `pyproject.toml` para pytest.

## Decisión

Se adopta la **alternativa 2** para esta migración.

## Justificación

La alternativa 1 toca el 100% del código fuente activo (~90 archivos) sin
que ningún test automatizado preexistente pudiera validar que el
comportamiento científico no cambió — violaría la restricción explícita
de "conservar la funcionalidad existente" y "no cambiar la lógica
científica sin justificarlo" bajo la que se ejecuta esta migración. El
riesgo (romper silenciosamente un import en 1 de 36 arquitecturas) supera
el beneficio (empaquetado más "estándar") en esta fase.

## Consecuencias

- Positivas: cero riesgo de romper el pipeline de entrenamiento existente;
  las pruebas nuevas (`tests/`) funcionan con `pythonpath = ["src"]` sin
  tocar el código de `src/`.
- Negativas: el proyecto no es instalable vía `pip install
  ecgi-1d-architectures-comparison` como librería reutilizable; `Dockerfile`
  y `run_bohb.py` deben seguir ejecutándose con `src/` como directorio de
  trabajo o con `sys.path` ajustado manualmente.
- Recomendación futura: si el proyecto evoluciona a librería reutilizable
  fuera de este pipeline de investigación, revisar esta decisión con una
  suite de tests de regresión ya consolidada (ver `docs/reproducibility.md`)
  que permita validar la reescritura de imports sin alterar resultados.
