#!/bin/bash
#SBATCH --job-name=modeling
#SBATCH --output=logs/modeling_%j.out
#SBATCH --error=logs/modeling_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --account=russpold
#SBATCH --partition=russpold

# --- Print job submission info ---
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Submitted from: $(pwd)"
echo "Subject ID: $SUBJECT_ID"
echo "Config File: $CONFIG_FILE"
echo "Environment: $ENV"

# --- Setup ---
ml python/3.9
source .venv/bin/activate

# --- Commands ---
echo "Running LSS Model..."
python scripts/modeling/run_lss_model.py \
    --subject $SUBJECT_ID \
    --config $CONFIG_FILE \
    --env $ENV

echo "Running Standard GLM..."
python scripts/modeling/run_standard_glm.py \
    --subject $SUBJECT_ID \
    --config $CONFIG_FILE \
    --env $ENV

echo "Modeling complete for $SUBJECT_ID."
