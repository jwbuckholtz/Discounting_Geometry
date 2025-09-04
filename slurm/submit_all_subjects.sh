#!/bin/bash
#
# This script is a wrapper that submits a SLURM job for each subject found
# in a BIDS data directory.
#
# Usage:
# ./slurm/submit_all_subjects.sh --bids-dir path/to/data --config path/to/config.yaml --env hpc
# --------------------------------------------------------------------------------

# --- Default Configuration ---
BIDS_DIR="data"
CONFIG_FILE="config/project_config.yaml"
ENV="hpc"
SBATCH_TEMPLATE="slurm/submit_behavioral_analysis.sbatch"

# --- Argument Parsing ---
# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --bids-dir) BIDS_DIR="$2"; shift ;;
        --config) CONFIG_FILE="$2"; shift ;;
        --env) ENV="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Validation ---
if [ ! -d "$BIDS_DIR" ]; then
    echo "Error: BIDS directory not found at '$BIDS_DIR'"
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at '$CONFIG_FILE'"
    exit 1
fi
if [ ! -f "$SBATCH_TEMPLATE" ]; then
    echo "Error: SBATCH template not found at '$SBATCH_TEMPLATE'"
    exit 1
fi

# --- Job Submission Loop ---
for subject_dir in "$BIDS_DIR"/sub-*/; do
    if [ -d "$subject_dir" ]; then
        subject_id=$(basename "$subject_dir")
        echo "Submitting job for subject: $subject_id"
        
        # Submit the job to SLURM, passing the configuration as environment variables
        sbatch --export=ALL,SUBJECT_ID="$subject_id",CONFIG_FILE="$CONFIG_FILE",ENV="$ENV" \
               "$SBATCH_TEMPLATE"
    fi
done

echo "All jobs submitted."
