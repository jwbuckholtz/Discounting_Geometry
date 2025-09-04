#!/bin/bash
#
#SBATCH --job-name=lss_modeling
#SBATCH --output=slurm/logs/lss_modeling_%j.out
#SBATCH --error=slurm/logs/lss_modeling_%j.err
#
# --------------------------------------------------------------------------------
# Resource Requests
# --------------------------------------------------------------------------------
# LSS is computationally intensive. Adjust these resources as needed.
#
#SBATCH --time=02:00:00  # Wall time limit (hh:mm:ss)
#SBATCH --nodes=1        # Number of nodes
#SBATCH --ntasks=1       # Number of tasks
#SBATCH --cpus-per-task=8 # Request more CPUs for parallel processing
#SBATCH --mem=16G        # Request more memory
#
# --------------------------------------------------------------------------------
# Job Script
# --------------------------------------------------------------------------------
# Variables expected from the wrapper: SUBJECT_ID, CONFIG_FILE, ENV
# --------------------------------------------------------------------------------

# Set up the environment
set -e # Exit immediately if a command exits with a non-zero status.

# --- Environment Activation ---
# Use default values for variables if they are not set.
: "${CONFIG_FILE:=config/project_config.yaml}"
: "${ENV:=hpc}"

# Activate your project's virtual environment
source .venv/bin/activate

# --- Run the LSS Modeling ---
echo "Starting LSS modeling for subject: $SUBJECT_ID"
echo "Using config file: $CONFIG_FILE"
echo "Using environment: $ENV"
echo "Job started at: $(date)"

python scripts/modeling/run_lss_modeling.py \
    --subject "$SUBJECT_ID" \
    --config "$CONFIG_FILE" \
    --env "$ENV"

echo "Job finished at: $(date)"
