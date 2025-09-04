#!/bin/bash
#
#SBATCH --job-name=dd_visualization
#SBATCH --output=slurm/logs/slurm-visualization-%j.out
#SBATCH --error=slurm/logs/slurm-visualization-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# --- Environment Setup ---
set -e # Exit immediately if a command exits with a non-zero status.

# Set default values for environment variables
: "${SUBJECT_ID:?Need to set SUBJECT_ID}"
: "${ROI_DIR:?Need to set ROI_DIR}"
: "${CONFIG_FILE:="config/project_config.yaml"}"
: "${ENV:="hpc"}"

echo "--- Job Details ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Subject ID: $SUBJECT_ID"
echo "ROI Directory: $ROI_DIR"
echo "Config File: $CONFIG_FILE"
echo "Environment: $ENV"
echo "--------------------"

# --- Activate Conda/Venv Environment ---
# Modify this line to activate your specific Python environment
# e.g., source /path/to/your/venv/bin/activate
source .venv/bin/activate

# --- Execute the Script ---
echo "Running visualization script..."
./scripts/visualization/run_all_visualizations_for_subject.sh "$SUBJECT_ID" "$ROI_DIR"

echo "Job finished with exit code $?"
