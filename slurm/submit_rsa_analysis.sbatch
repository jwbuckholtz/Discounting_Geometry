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

# Set up the environment
set -e # Exit immediately if a command exits with a non-zero status.

# --- Environment Activation ---
: "${CONFIG_FILE:=config/project_config.yaml}"
: "${ENV:=hpc}"

# Activate your project's virtual environment
source .venv/bin/activate

# --- Construct the Python Command ---
# Start with the base command
CMD="python scripts/rsa/run_rsa_analysis.py \
    --subject \"$SUBJECT_ID\" \
    --analysis-type \"$ANALYSIS_TYPE\" \
    --config \"$CONFIG_FILE\" \
    --env \"$ENV\""

# Add the --roi-path argument ONLY if the analysis type is 'roi'
if [ "$ANALYSIS_TYPE" = "roi" ]; then
    CMD="$CMD --roi-path \"$ROI_PATH\""
fi

# --- Run the RSA ---
echo "Starting RSA for subject: $SUBJECT_ID"
echo "Analysis Type: $ANALYSIS_TYPE"
if [ "$ANALYSIS_TYPE" = "roi" ]; then
    echo "ROI Path: $ROI_PATH"
fi
echo "Job started at: $(date)"

# Execute the constructed command
eval $CMD

echo "Job finished at: $(date)"
