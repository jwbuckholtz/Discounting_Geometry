#!/bin/bash
set -e  # Exit immediately if any command fails
#
# This script submits a group-level GLM analysis job for each specified contrast.
#
# --------------------------------------------------------------------------------

# --- Environment Validation ---
: ${PROJECT_ROOT:?ERROR: PROJECT_ROOT not set - export PROJECT_ROOT=/path/to/project}

# Verify PROJECT_ROOT exists
if [[ ! -d "$PROJECT_ROOT" ]]; then
    echo "ERROR: PROJECT_ROOT directory does not exist: $PROJECT_ROOT"
    exit 1
fi

# Change to project directory for absolute path resolution
cd "$PROJECT_ROOT"

# Ensure logs directory exists before submission
mkdir -p logs

# --- Configuration ---
CONFIG_FILE="$PROJECT_ROOT/config/project_config.yaml"
ENV="hpc"
SBATCH_TEMPLATE="$PROJECT_ROOT/slurm/submit_group_glm.sbatch"

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
