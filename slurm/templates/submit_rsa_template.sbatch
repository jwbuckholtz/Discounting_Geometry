#!/bin/bash
#SBATCH --job-name=rsa
#SBATCH --output=logs/rsa_%j.out
#SBATCH --error=logs/rsa_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --account=russpold
#SBATCH --partition=russpold

# --- Print job submission info ---
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Submitted from: $(pwd)"
echo "Subject ID: $SUBJECT_ID"
echo "Analysis Type: $ANALYSIS_TYPE"
echo "Config File: $CONFIG_FILE"
echo "Environment: $ENV"

# --- Setup ---
source /oak/stanford/groups/russpold/users/buckholtz/Decoding_DD/setup.sh
# --- Command ---
python scripts/rsa/run_rsa_analysis.py \
    --subject $SUBJECT_ID \
    --analysis-type $ANALYSIS_TYPE \
    --config $CONFIG_FILE \
    --env $ENV
