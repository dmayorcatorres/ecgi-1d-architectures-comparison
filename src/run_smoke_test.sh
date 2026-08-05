#!/usr/bin/env bash
# run_smoke_test.sh — Verifica que todos los modelos cargan y corren sin errores
#
# Corre 2 trials HPO con 5 épocas sobre un dataset pequeño.
# NO guarda métricas ni checkpoints reales — solo detecta crashes.
#
# Variables de entorno opcionales:
#   DATASET    dataset a usar      (default: ckitpvc)
#   METHODS    subset de modelos   (default: los 36)
#   DIRECTION  ecg2emg | emg2ecg  (default: ecg2emg)
#
# Uso:
#   bash run_smoke_test.sh
#   DATASET=auckland bash run_smoke_test.sh
#   METHODS="resnet vit1d cnnlstm" bash run_smoke_test.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

PYTHON="${PYTHON:-python3}"
SCRIPT="run_bohb.py"
DATASET="${DATASET:-auckland}"
DIRECTION="${DIRECTION:-ecg2emg}"
NUM_SAMPLES=2
MAX_EPOCHS=5

# ── Modelos (37 en total) ─────────────────────────────────────────────────────
read -r -a METHODS <<< "${METHODS:-\
  dilatedcnn disjointcnn disjointresnet fcn inceptiontime lite mlp \
  multichanneldcnn multiscalecnn resnet resnetcnn tcn tfrcnn zhaocnn \
  esn indrnn peepholelstm qrnn rnn_gru rnn_lstm tlgru vsgrunet \
  cnnbilstmbigrudil cnnlstm cnngru cnnlstmgru deepresidualbilstm disjointlstm lstmfcn \
  alstm autoformer1d informer1d patchtst physformer timenet1d transformer1d vit1d}"

LOG_DIR="batch_logs/smoke_${DIRECTION}"
mkdir -p "${LOG_DIR}"

N_OK=0
N_FAIL=0
FAILED=()

echo "════════════════════════════════════════════════════"
echo "  SMOKE TEST — verificación rápida de modelos"
printf "  Dataset     : %s\n" "${DATASET}"
printf "  Modelos     : %d\n" "${#METHODS[@]}"
printf "  Trials HPO  : %d\n" "${NUM_SAMPLES}"
printf "  Max épocas  : %d\n" "${MAX_EPOCHS}"
printf "  Direction   : %s\n" "${DIRECTION}"
echo "════════════════════════════════════════════════════"

for m in "${METHODS[@]}"; do
  LOG_FILE="${LOG_DIR}/${DATASET}_${m}.log"
  printf "  %-26s ... " "${m}"

  if "${PYTHON}" "${SCRIPT}" \
      --dataset    "${DATASET}" \
      --method     "${m}" \
      --direction  "${DIRECTION}" \
      --num-samples "${NUM_SAMPLES}" \
      --max-epochs  "${MAX_EPOCHS}" \
      > "${LOG_FILE}" 2>&1; then
    echo "✅ OK"
    N_OK=$(( N_OK + 1 ))
  else
    echo "❌ FAIL  → ${LOG_FILE}"
    N_FAIL=$(( N_FAIL + 1 ))
    FAILED+=("${m}")
  fi
done

echo ""
echo "════════════════════════════════════════════════════"
printf "  Pasaron : %d / %d\n" "${N_OK}" "${#METHODS[@]}"
printf "  Fallaron: %d / %d\n" "${N_FAIL}" "${#METHODS[@]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "  Modelos con error:"
  for f in "${FAILED[@]}"; do
    printf "    ❌ %s\n" "${f}"
  done
fi
echo "════════════════════════════════════════════════════"
