#!/bin/bash
#
#SBATCH --job-name=standard_glm
#SBATCH --output=slurm/logs/standard_glm_%j_%x.out
#SBATCH --error=slurm/logs/standard_glm_%j_%x.err
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00

# --- Load Modules ---
ml python/3.9

# --- Environment Setup ---
source .venv/bin/activate

# --- Job Execution ---
echo "Running standard GLM for subject: $SUBJECT_ID"
./.venv/bin/python scripts/modeling/run_standard_glm.py \
    --config "$CONFIG_FILE" \
    --env "$ENV" \
    --subject "$SUBJECT_ID"
