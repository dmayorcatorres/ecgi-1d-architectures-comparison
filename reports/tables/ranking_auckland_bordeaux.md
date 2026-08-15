# Ranking combinado — auckland + bordeaux

Exportado de `src/metrics_log.xlsx` en el commit `9b9daca1` (2026-08-15).
Cubre los 37 métodos con al menos una corrida HPO real registrada en
ambos datasets (60/20/20, test held-out ciego, `direction=ecg2emg`).

## Metodología

Por cada método y dataset se toma la corrida con mejor `PCC_mean` (mejor
trial HPO registrado para ese método en ese dataset). El ranking
combinado se calcula de dos formas, que aquí coinciden en el top:

- **Por PCC promedio**: `(PCC_auckland + PCC_bordeaux) / 2`, ordenado
  descendente.
- **Por rank promedio**: `(rank_auckland + rank_bordeaux) / 2`, ordenado
  ascendente (penaliza más ser irregular entre datasets que el promedio
  de PCC crudo).

No es un benchmark controlado de "mismo presupuesto de HPO para todos"
-- cada método tiene su propio historial de corridas (ver
`docs/experiments.md` y el hallazgo de auckland en la nota al final).

## Top 10 combinado (por PCC promedio)

| # | método | PCC auckland | rank auckland | PCC bordeaux | rank bordeaux | PCC promedio |
|---|---|---|---|---|---|---|
| 1 | `tfrcnn` | 0.9741 | 3 | 0.8990 | 2 | 0.9365 |
| 2 | `resnet` | 0.9816 | 1 | 0.8905 | 5 | 0.9361 |
| 3 | `fcn` | 0.9796 | 2 | 0.8897 | 7 | 0.9346 |
| 4 | `tcn` | 0.9659 | 6 | 0.8933 | 3 | 0.9296 |
| 5 | `disjointlstm` | 0.9667 | 5 | 0.8746 | 9 | 0.9206 |
| 6 | `inceptiontime` | 0.9668 | 4 | 0.8700 | 10 | 0.9184 |
| 7 | `cnnbilstmbigrudil` | 0.9634 | 7 | 0.8459 | 13 | 0.9047 |
| **8** | **`physformer`** | **0.9059** | **23** | **0.9026** | **1** | **0.9042** |
| 9 | `dilatedcnn` | 0.9152 | 19 | 0.8926 | 4 | 0.9039 |
| 10 | `cnnlstmgru` | 0.9598 | 8 | 0.8233 | 16 | 0.8915 |

Por rank promedio (en vez de PCC crudo) el top 7 es idéntico; `physformer`
cae al puesto 9 (empatado con `cnnlstmgru`, rank promedio 12.0) porque su
23º lugar en auckland pesa más bajo este criterio que su 1º en bordeaux.

## Ganador: `tfrcnn`

El modelo mejor rankeado considerando **ambos** datasets es **`tfrcnn`**
(1º combinado, tanto por PCC promedio como por rank promedio) — 3º en
auckland (PCC 0.9741) y 2º en bordeaux (PCC 0.8990), el más consistente
del top del ranking. `resnet` y `fcn` le siguen muy de cerca, con mejor
techo individual en auckland pero peor consistencia relativa en bordeaux.

## `physformer` (PhysMixFormer1D)

Único método que es **1º de 37 en bordeaux**, pero 23º en auckland — el
resultado combinado (8º/9º de 37) refleja esa asimetría, no una debilidad
general del método. Auckland tiene solo 18 muestras totales (10 train /
4 val / 4 test) -- con un test de 4 muestras el ranking ahí tiene mucho
ruido de muestreo; los modelos con menos "piezas móviles" (CNNs
residuales simples) tienden a sobreajustar bien a ese ruido específico,
mientras que un modelo con más parámetros/mecanismos (atención + física)
tiene más superficie para que ese ruido lo perjudique. Ver
`git log --oneline 879d9f6f 9b9daca1` para el detalle completo de la
integración de la física al pipeline y el diagnóstico de por qué el HPO
inicial (sin forzar física) daba peor resultado.
