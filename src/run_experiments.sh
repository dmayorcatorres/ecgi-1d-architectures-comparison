#!/usr/bin/env bash
# run_experiments.sh — Ejecución completa: todos los datasets y modelos
#
# Variables de entorno opcionales:
#   DATASETS     lista de datasets  (default: los 4)
#   METHODS      lista de modelos   (default: los 36)
#   DIRECTION    ecg2emg | emg2ecg  (default: ecg2emg)
#   NUM_SAMPLES  trials HPO BOHB    (default: 15)
#   MAX_EPOCHS   épocas máximas     (default: 50)
#   SAVE_CURVES  1 guarda PNG/CSV   (default: 0)
#   SKIP_DONE    1 salta completados (default: 1)
#
# Uso:
#   bash run_experiments.sh
#   NUM_SAMPLES=30 bash run_experiments.sh
#   DATASETS="ckitpvc auckland" bash run_experiments.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

PYTHON="${PYTHON:-python3}"
SCRIPT="run_bohb.py"
DIRECTION="${DIRECTION:-ecg2emg}"
NUM_SAMPLES="${NUM_SAMPLES:-15}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
SAVE_CURVES="${SAVE_CURVES:-0}"
SKIP_DONE="${SKIP_DONE:-1}"

# ── Datasets ──────────────────────────────────────────────────────────────────
read -r -a DATASETS <<< "${DATASETS:-ckitpvc auckland bordeaux utah2002}"

# ── Modelos (37 en total) ─────────────────────────────────────────────────────
read -r -a METHODS <<< "${METHODS:-\
  dilatedcnn disjointcnn disjointresnet fcn inceptiontime lite mlp \
  multichanneldcnn multiscalecnn resnet resnetcnn tcn tfrcnn zhaocnn \
  esn indrnn peepholelstm qrnn rnn_gru rnn_lstm tlgru vsgrunet \
  cnnbilstmbigrudil cnnlstm cnngru cnnlstmgru deepresidualbilstm disjointlstm lstmfcn \
  alstm autoformer1d informer1d patchtst physformer timenet1d transformer1d vit1d}"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR="batch_logs/${DIRECTION}"
mkdir -p "${LOG_DIR}"

EXTRA_ARGS=("--direction" "${DIRECTION}" "--num-samples" "${NUM_SAMPLES}" "--max-epochs" "${MAX_EPOCHS}")
[[ "${SAVE_CURVES}" == "1" ]] && EXTRA_ARGS+=("--save-curves")

TOTAL=$(( ${#DATASETS[@]} * ${#METHODS[@]} ))
N_DONE=0
N_SKIP=0
FAILED=()

echo "════════════════════════════════════════════════════"
echo "  run_bohb.py — batch completo"
printf "  Datasets    : %d  → %s\n" "${#DATASETS[@]}" "${DATASETS[*]}"
printf "  Modelos     : %d\n" "${#METHODS[@]}"
printf "  Total runs  : %d\n" "${TOTAL}"
printf "  Direction   : %s\n" "${DIRECTION}"
printf "  Num samples : %s\n" "${NUM_SAMPLES}"
printf "  Max epochs  : %s\n" "${MAX_EPOCHS}"
printf "  Save curves : %s\n" "${SAVE_CURVES}"
printf "  Skip done   : %s\n" "${SKIP_DONE}"
echo "════════════════════════════════════════════════════"

for ds in "${DATASETS[@]}"; do
  for m in "${METHODS[@]}"; do
    LOG_FILE="${LOG_DIR}/${ds}_${m}.log"

    if [[ "${SKIP_DONE}" == "1" && -f "${LOG_FILE}" ]] \
        && grep -q "Métricas finales" "${LOG_FILE}" 2>/dev/null; then
      printf "⏭  %-22s / %-26s (ya completado)\n" "${ds}" "${m}"
      N_SKIP=$(( N_SKIP + 1 ))
      continue
    fi

    echo ""
    echo "──────────────────────────────────────────────────────"
    printf "  Dataset: %-12s  Modelo: %-22s  Dir: %s\n" "${ds}" "${m}" "${DIRECTION}"
    echo "──────────────────────────────────────────────────────"

    if "${PYTHON}" "${SCRIPT}" \
        --dataset "${ds}" \
        --method  "${m}" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "${LOG_FILE}"; then
      printf "✅  Completado: %s / %s\n" "${ds}" "${m}"
      N_DONE=$(( N_DONE + 1 ))
    else
      printf "❌  Error en:   %s / %s  → %s\n" "${ds}" "${m}" "${LOG_FILE}"
      FAILED+=("${ds}/${m}")
    fi
  done
done

echo ""
echo "════════════════════════════════════════════════════"
echo "  Resumen final"
printf "  Completados : %d\n" "${N_DONE}"
printf "  Saltados    : %d\n" "${N_SKIP}"
printf "  Fallidos    : %d\n" "${#FAILED[@]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "  Detalle fallidos:"
  for f in "${FAILED[@]}"; do
    printf "    ❌ %s\n" "${f}"
  done
fi
echo "════════════════════════════════════════════════════"
