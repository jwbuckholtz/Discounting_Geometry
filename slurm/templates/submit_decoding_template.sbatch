#!/bin/bash
#SBATCH --job-name=decoding
#SBATCH --output=logs/decoding_%j.out
#SBATCH --error=logs/decoding_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
# Note: Set SLURM_ACCOUNT and SLURM_PARTITION environment variables for your cluster
# Example: export SLURM_ACCOUNT=your_account SLURM_PARTITION=your_partition
# Uncomment and modify these lines for cluster-specific settings:
# #SBATCH --account=${SLURM_ACCOUNT}
# #SBATCH --partition=${SLURM_PARTITION}

# --- Script Initialization ---
set -e  # Exit immediately if any command fails

# --- Validate Required Environment Variables ---
: ${SUBJECT_ID:?ERROR: SUBJECT_ID not set - required for subject processing}
: ${TARGET:?ERROR: TARGET not set - required for decoding target}
: ${CONFIG_FILE:?ERROR: CONFIG_FILE not set - required for configuration}
: ${ENV:?ERROR: ENV not set - required for environment selection}
: ${PROJECT_ROOT:?ERROR: PROJECT_ROOT not set - export PROJECT_ROOT=/path/to/project}

# --- Print job submission info ---
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Submitted from: $(pwd)"
echo "Subject ID: $SUBJECT_ID"
echo "Target: $TARGET"
echo "Config File: $CONFIG_FILE"
echo "Environment: $ENV"

# --- Shell Setup ---
# Change to project directory for consistent execution context
cd "$PROJECT_ROOT"

# Ensure logs directory exists for output files
mkdir -p logs

# --- Setup ---
source slurm/load_python_module.sh
source .venv/bin/activate

# --- Command ---
# Arguments: subject_id (positional), then flags (all quoted for safety)
python scripts/mvpa/run_decoding_analysis.py \
    "$SUBJECT_ID" \
    --target "$TARGET" \
    --config "$CONFIG_FILE" \
    --env "$ENV"
