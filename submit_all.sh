#!/bin/bash

#BSUB -J submit_all
#BSUB -q hpc
#BSUB -W 00:10
#BSUB -o results/submit_all_%J.out

# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------
POOL_VALUES=(mean cls max)
EMB_VALUES=(256)
HID_VALUES=(128)
FF_VALUES=(1024)
BS_VALUES=(64)
EPOCH_VALUES=(10)

RUN_ID="$(date +%Y-%m-%d_%H%M%S)"
BATCH_DIR="results/${RUN_ID}"
mkdir -p "$BATCH_DIR"

echo "Experiment batch: $RUN_ID"
TOTAL=0

for pool in "${POOL_VALUES[@]}"; do
for emb in "${EMB_VALUES[@]}"; do
for hid in "${HID_VALUES[@]}"; do
for ff in "${FF_VALUES[@]}"; do
for bs in "${BS_VALUES[@]}"; do
for ep in "${EPOCH_VALUES[@]}"; do

    NAME="pool${pool}_emb${emb}_hid${hid}_ff${ff}_bs${bs}_ep${ep}"
    RUN_DIR="${BATCH_DIR}/${NAME}"
    mkdir -p "$RUN_DIR"
    echo "Submitting: $NAME"

    bsub \
        -J "dpr_${pool}" \
        -oo "${RUN_DIR}/stdout.out" \
        -eo "${RUN_DIR}/stderr.err" \
        -env "all,POOL=$pool,EMB=$emb,HID=$hid,FF=$ff,BS=$bs,EPOCHS=$ep,RUN_ID=$RUN_ID" \
        < job.sh

    [ $? -eq 0 ] && TOTAL=$((TOTAL + 1)) || echo "ERROR: failed to submit $NAME"

done; done; done; done; done; done

echo ""
echo "Submitted $TOTAL jobs into $BATCH_DIR/  (check status with: bstat)"
echo "When done, summarize with: python summarize_results.py $BATCH_DIR"
