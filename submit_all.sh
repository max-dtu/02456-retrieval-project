#!/bin/bash
# Submit one GPU job per hyperparameter combination. Run from the project root:
#   bash submit_all.sh

# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------
POOL_VALUES=(mean cls max)
EMB_VALUES=(256)
HID_VALUES=(128)
FF_VALUES=(1024)
BS_VALUES=(64)
EPOCH_VALUES=(10)

RUN_ID="$(date +%Y-%m-%d_%H%M)"
LOG_DIR="results/${RUN_ID}/logs"
mkdir -p "$LOG_DIR"

echo "Experiment batch: $RUN_ID"
TOTAL=0

for pool in "${POOL_VALUES[@]}"; do
for emb in "${EMB_VALUES[@]}"; do
for hid in "${HID_VALUES[@]}"; do
for ff in "${FF_VALUES[@]}"; do
for bs in "${BS_VALUES[@]}"; do
for ep in "${EPOCH_VALUES[@]}"; do

    NAME="pool${pool}_emb${emb}_hid${hid}_ff${ff}_bs${bs}_ep${ep}"
    echo "Submitting: $NAME"

    bsub \
        -J "dpr_${pool}" \
        -oo "${LOG_DIR}/${NAME}.out" \
        -eo "${LOG_DIR}/${NAME}.err" \
        -env "all,POOL=$pool,EMB=$emb,HID=$hid,FF=$ff,BS=$bs,EPOCHS=$ep,RUN_ID=$RUN_ID" \
        < job.sh

    [ $? -eq 0 ] && TOTAL=$((TOTAL + 1)) || echo "ERROR: failed to submit $NAME"

done; done; done; done; done; done

echo ""
echo "Submitted $TOTAL jobs. Logs: $LOG_DIR/  (check status with: bstat)"
