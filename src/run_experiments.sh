#!/usr/bin/env bash
# run_all.sh — Ejecuta run_trials2.py para múltiples datasets y métodos
set -euo pipefail

# Configuración
PYTHON_CMD="python run_trials2.py"
DATASETS=(kitpvc auckland bordeux utah2002)
METHODS=(
  dilatedcnn disjointcnn disjointresnet fcn inceptiontime lite mlp
  multichanneldcnn multiscalecnn resnet resnetcnn tcn tfrcnn zhaocnn
  tlgru vsgrunet rnn_gru rnn_lstm peepholelstm indrnn esn qrnn
  deepresidualbilstm cnnlstm cnngru cnnlstmgru lstmfcn disjointlstm cnnbilstmbigrudil
  alstm transformer1d informer1d autoformer1d patchtst vit1d timenet1d
)

LOG_DIR="batch_logs"
mkdir -p "${LOG_DIR}"

for ds in "${DATASETS[@]}"; do
  for m in "${METHODS[@]}"; do
    echo "=============================="
    echo "Dataset: $ds   Method: $m"
    echo "=============================="
    LOG_FILE="${LOG_DIR}/${ds}_${m}.log"

    # Ejecuta, muestra en pantalla y escribe al log
    if ${PYTHON_CMD} --dataset "$ds" --method "$m" 2>&1 | tee "${LOG_FILE}"; then
      echo "✅ Completado: $ds / $m"
    else
      echo "❌ Error en:     $ds / $m  (ver ${LOG_FILE})"
    fi

  done
done
