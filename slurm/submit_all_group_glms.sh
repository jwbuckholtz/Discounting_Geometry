#!/bin/bash
#
# This script submits a group-level GLM analysis job for each specified contrast.
#
# --------------------------------------------------------------------------------

# --- Configuration ---
CONFIG_FILE="config/project_config.yaml"
ENV="hpc"
SBATCH_TEMPLATE="slurm/submit_group_glm.sbatch"

# An array of the contrast names to run group analyses for
CONTRASTS=(
    "choice"
    "SVchosen"
    "SVunchosen"
    "SVsum"
    "SVdiff"
)

# --- Validation ---
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at '$CONFIG_FILE'"
    exit 1
fi
if [ ! -f "$SBATCH_TEMPLATE" ]; then
    echo "Error: SBATCH template not found at '$SBATCH_TEMPLATE'"
    exit 1
fi

# --- Job Submission Loop ---
for contrast in "${CONTRASTS[@]}"; do
    echo "Submitting group GLM job for contrast: $contrast"
    
    sbatch --export=ALL,CONFIG_FILE="$CONFIG_FILE",ENV="$ENV",CONTRAST="$contrast" \
           "$SBATCH_TEMPLATE"
done

echo "All group GLM jobs submitted."
