#!/bin/bash
set -e  # Exit immediately if any command fails
#
#SBATCH --job-name=standard_glm
#SBATCH --output=logs/standard_glm_%j_%x.out
#SBATCH --error=logs/standard_glm_%j_%x.err
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00

# --- Shell Setup ---
# Change to the directory where you are submitting the script from
cd "$SLURM_SUBMIT_DIR"

# Ensure logs directory exists for output files
mkdir -p logs

# --- Load Modules ---
ml python/3.9

# --- Environment Setup ---
source .venv/bin/activate

# --- Job Execution ---
echo "Running standard GLM for subject: $SUBJECT_ID"
# Arguments: config_path env subject_id (positional)
./.venv/bin/python scripts/modeling/run_standard_glm.py \
    "$CONFIG_FILE" \
    "$ENV" \
    "$SUBJECT_ID"
