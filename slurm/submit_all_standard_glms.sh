#!/bin/bash
#
# This script submits a SLURM job for each subject's standard GLM.
# --------------------------------------------------------------------------------

# --- Default Configuration ---
BIDS_DIR="" # This will be read from the config file
CONFIG_FILE="config/project_config.yaml"
ENV="hpc"
SBATCH_TEMPLATE="slurm/submit_standard_glm.sbatch"

# --- Load Environment ---
# Load the python module to ensure the config can be read correctly.
ml python/3.9

# --- Argument Parsing ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --bids-dir) BIDS_DIR="$2"; shift ;;
        --config) CONFIG_FILE="$2"; shift ;;
        --env) ENV="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Read BIDS directory from config if not provided ---
if [ -z "$BIDS_DIR" ]; then
    BIDS_DIR=$(python -c "import yaml; f = open('$CONFIG_FILE'); config = yaml.safe_load(f); print(config['$ENV']['bids_dir'])")
fi

# --- Validation ---
if [ ! -d "$BIDS_DIR" ]; then
    echo "Error: BIDS directory not found at '$BIDS_DIR'"
    exit 1
fi

# --- Job Submission Loop ---
for subject_dir in "$BIDS_DIR"/sub-*/; do
    if [ -d "$subject_dir" ]; then
        subject_id=$(basename "$subject_dir")
        echo "Submitting standard GLM job for subject: $subject_id"
        
        sbatch --export=ALL,SUBJECT_ID="$subject_id",CONFIG_FILE="$CONFIG_FILE",ENV="$ENV" \
               "$SBATCH_TEMPLATE"
    fi
done

echo "All standard GLM jobs submitted."
