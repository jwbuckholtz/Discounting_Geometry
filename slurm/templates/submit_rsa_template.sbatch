#!/bin/bash
set -e  # Exit immediately if any command fails
#SBATCH --job-name=rsa
#SBATCH --output=logs/rsa_%j.out
#SBATCH --error=logs/rsa_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --account=russpold
#SBATCH --partition=russpold

# --- Validate Required Environment Variables ---
: ${SUBJECT_ID:?ERROR: SUBJECT_ID not set - required for subject processing}
: ${ANALYSIS_TYPE:?ERROR: ANALYSIS_TYPE not set - required for RSA analysis type}
: ${CONFIG_FILE:?ERROR: CONFIG_FILE not set - required for configuration}
: ${ENV:?ERROR: ENV not set - required for environment selection}

# --- Print job submission info ---
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Submitted from: $(pwd)"
echo "Subject ID: $SUBJECT_ID"
echo "Analysis Type: $ANALYSIS_TYPE"
echo "Config File: $CONFIG_FILE"
echo "Environment: $ENV"

# --- Shell Setup ---
# Change to the directory where you are submitting the script from
cd "$SLURM_SUBMIT_DIR"

# Ensure logs directory exists for output files
mkdir -p logs

# --- Setup ---
ml python/3.9
source .venv/bin/activate

# --- Command ---
# All variables quoted for safety
python scripts/rsa/run_rsa_analysis.py \
    --subject "$SUBJECT_ID" \
    --analysis-type "$ANALYSIS_TYPE" \
    --config "$CONFIG_FILE" \
    --env "$ENV"
