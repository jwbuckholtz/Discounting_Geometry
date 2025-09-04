#!/bin/bash
#
# This script submits a SLURM job for each subject and each target variable.
#
# Usage:
# ./slurm/submit_all_decoding.sh
# --------------------------------------------------------------------------------

# --- Configuration ---
BIDS_DIR="data"
CONFIG_FILE="config/project_config.yaml"
ENV="hpc"
SBATCH_TEMPLATE="slurm/submit_decoding_analysis.sbatch"

# Define the list of target variables you want to decode
TARGETS=(
    "choice"
    "delay_to_reward"
    "SVchosen"
    "SVunchosen"
    "SVsum"
    "SVdiff"
)

# --- Validation ---
if [ ! -d "$BIDS_DIR" ]; then
    echo "Error: BIDS directory not found at '$BIDS_DIR'"
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at '$CONFIG_FILE'"
    exit 1
fi
if [ ! -f "$SBATCH_TEMPLATE" ]; then
    echo "Error: SBATCH template not found at '$SBATCH_TEMPLATE'"
    exit 1
fi

# --- Job Submission Loop ---
for subject_dir in "$BIDS_DIR"/sub-*/; do
    if [ -d "$subject_dir" ]; then
        subject_id=$(basename "$subject_dir")
        
        for target in "${TARGETS[@]}"; do
            echo "Submitting decoding job for subject: $subject_id, target: $target"
            
            sbatch --export=ALL,SUBJECT_ID="$subject_id",TARGET="$target",CONFIG_FILE="$CONFIG_FILE",ENV="$ENV" \
                   "$SBATCH_TEMPLATE"
        done
    fi
done

echo "All decoding jobs submitted."
