#!/bin/bash
#
#SBATCH --job-name=rsa
#SBATCH --output=slurm/logs/rsa_%j.out
#SBATCH --error=slurm/logs/rsa_%j.err
#
# --------------------------------------------------------------------------------
# Resource Requests
# --------------------------------------------------------------------------------
# Searchlight is very intensive. ROI/whole-brain are less so.
# We will request resources for the most demanding case (searchlight).
#
#SBATCH --time=04:00:00  # Wall time limit (hh:mm:ss)
#SBATCH --nodes=1        # Number of nodes
#SBATCH --ntasks=1       # Number of tasks
#SBATCH --cpus-per-task=8 # CPUs for searchlight
#SBATCH --mem=24G        # Memory for searchlight
#
# --------------------------------------------------------------------------------
# Job Script
# --------------------------------------------------------------------------------
# Variables from wrapper: SUBJECT_ID, ANALYSIS_TYPE, ROI_PATH, CONFIG_FILE, ENV
# --------------------------------------------------------------------------------

# --- Environment Setup ---
set -e # Exit immediately if a command exits with a non-zero status.

# Set default values for environment variables
: "${SUBJECT_ID:?Need to set SUBJECT_ID}"
: "${ANALYSIS_TYPE:?Need to set ANALYSIS_TYPE}"
: "${CONFIG_FILE:="config/project_config.yaml"}"
: "${ENV:="hpc"}"
: "${ROI_PATH:=""}" # Default ROI_PATH to an empty string if not set

# Load the Python module to make libraries available on the compute node
ml python/3.9

# Activate Conda/Venv Environment
source .venv/bin/activate

# --- Construct the Python Command ---
# Start with the base command
CMD="python scripts/rsa/run_rsa_analysis.py \
    --subject '$SUBJECT_ID' \
    --config '$CONFIG_FILE' \
    --env '$ENV' \
    --analysis-type '$ANALYSIS_TYPE'"

if [ -n "$ROI_PATH" ]; then
    CMD="$CMD --roi-path '$ROI_PATH'"
fi

# --- Run the analysis ---
echo "Executing command: $CMD"
eval $CMD

echo "Job finished at: $(date)"
