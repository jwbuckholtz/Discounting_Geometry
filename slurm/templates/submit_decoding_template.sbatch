#!/bin/bash
#SBATCH --job-name=decoding
#SBATCH --output=logs/decoding_%j.out
#SBATCH --error=logs/decoding_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --account=russpold
#SBATCH --partition=russpold

# --- Print job submission info ---
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Submitted from: $(pwd)"
echo "Subject ID: $SUBJECT_ID"
echo "Target: $TARGET"
echo "Config File: $CONFIG_FILE"
echo "Environment: $ENV"

# --- Setup ---
source /oak/stanford/groups/russpold/users/buckholtz/Decoding_DD/setup.sh
# --- Command ---
python scripts/mvpa/run_decoding_analysis.py \
    $SUBJECT_ID \
    $DERIVATIVES_DIR \
    --target $TARGET \
    --config $CONFIG_FILE \
    --env $ENV
