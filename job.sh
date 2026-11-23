#!/bin/bash

# --------------------------------------------------
# LSF CONFIGURATION  (one DPR training + evaluation run on one GPU)
# --------------------------------------------------
#BSUB -q gpuv100
#BSUB -W 02:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"

# --------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------
module load python3/3.12.11          # check name with: module avail python3
source ~/venvs/dpr/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export HF_HUB_OFFLINE=1              

# --------------------------------------------------
# OUTPUT CONFIGURATION
# --------------------------------------------------
RUN_NAME="pool${POOL}_emb${EMB}_hid${HID}_ff${FF}_bs${BS}_ep${EPOCHS}"
RUN_DIR="results/${RUN_ID}/${RUN_NAME}"
mkdir -p "$RUN_DIR"

echo "Run:          $RUN_NAME"
echo "Batch ID:     $RUN_ID"
echo "Output dir:   $RUN_DIR"
nvidia-smi

# --------------------------------------------------
# RUN
# --------------------------------------------------
python main.py \
    device=cuda \
    model.pooling_type="$POOL" \
    model.emb_dim="$EMB" \
    model.hidden_dim="$HID" \
    model.feed_forward_dim="$FF" \
    model.batch_size="$BS" \
    model.epochs="$EPOCHS" \
    +model.force_train=true \
    +model.query_encoder_weights_path="${RUN_DIR}/query_encoder.pt" \
    +model.document_encoder_weights_path="${RUN_DIR}/document_encoder.pt" \
    hydra.run.dir="$RUN_DIR"

STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: Run failed with exit code $STATUS"
    exit "$STATUS"
fi
echo "Run completed successfully."
