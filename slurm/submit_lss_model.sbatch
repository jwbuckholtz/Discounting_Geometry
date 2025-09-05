#!/bin/bash
#
#SBATCH --job-name=lss_model
#SBATCH --output=slurm/logs/lss_model_%j_%x.out
#SBATCH --error=slurm/logs/lss_model_%j_%x.err
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00 # LSS is more time-intensive

# --- Load Modules ---
ml python/3.9

# --- Environment Setup ---
source .venv/bin/activate

# --- Job Execution ---
echo "Running LSS Model for subject: $SUBJECT_ID"
./.venv/bin/python scripts/modeling/run_lss_model.py \
    --config "$CONFIG_FILE" \
    --env "$ENV" \
    --subject "$SUBJECT_ID"
