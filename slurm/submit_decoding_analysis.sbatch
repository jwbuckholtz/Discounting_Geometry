#!/bin/bash
#
#SBATCH --job-name=decoding
#SBATCH --output=slurm/logs/decoding_%j.out
#SBATCH --error=slurm/logs/decoding_%j.err
#
# --------------------------------------------------------------------------------
# Resource Requests
# --------------------------------------------------------------------------------
# Decoding can be computationally intensive.
#
#SBATCH --time=01:00:00  # Wall time limit (hh:mm:ss)
#SBATCH --nodes=1        # Number of nodes
#SBATCH --ntasks=1       # Number of tasks
#SBATCH --cpus-per-task=4 # Number of CPUs per task
#SBATCH --mem=8G         # Memory per node
#
# --------------------------------------------------------------------------------
# Job Script
# --------------------------------------------------------------------------------
# Variables expected from the wrapper: SUBJECT_ID, TARGET, CONFIG_FILE, ENV
# --------------------------------------------------------------------------------

# Set up the environment
set -e # Exit immediately if a command exits with a non-zero status.

# --- Environment Activation ---
: "${CONFIG_FILE:=config/project_config.yaml}"
: "${ENV:=hpc}"

# Activate your project's virtual environment
source .venv/bin/activate

# --- Run the Decoding Analysis ---
echo "Starting decoding for subject: $SUBJECT_ID, target: $TARGET"
echo "Using config file: $CONFIG_FILE"
echo "Using environment: $ENV"
echo "Job started at: $(date)"

python scripts/mvpa/run_decoding_analysis.py \
    --subject "$SUBJECT_ID" \
    --target "$TARGET" \
    --config "$CONFIG_FILE" \
    --env "$ENV"

echo "Job finished at: $(date)"
