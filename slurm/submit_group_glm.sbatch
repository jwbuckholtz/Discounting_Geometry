#!/bin/bash
#
#SBATCH --job-name=group_glm
#SBATCH --output=slurm/logs/group_glm_%j.out
#SBATCH --error=slurm/logs/group_glm_%j.err
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00

# --- Load Modules ---
ml python/3.9

# --- Environment Setup ---
# Activate the virtual environment
source .venv/bin/activate
# Install project in editable mode if not already done
uv pip install -e .

# --- Job Execution ---
echo "========================================="
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Contrast: $CONTRAST"
echo "Config File: $CONFIG_FILE"
echo "Environment: $ENV"
echo "========================================="

# Run the group-level GLM script
./.venv/bin/python scripts/group_level/run_group_glm.py \
    --config "$CONFIG_FILE" \
    --env "$ENV" \
    --contrast "$CONTRAST"

# --- Job Completion ---
echo "========================================="
echo "Job finished at $(date)"
echo "========================================="
