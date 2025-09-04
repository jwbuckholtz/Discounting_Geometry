#!/bin/bash
#
# This script is a wrapper that submits a SLURM job for each subject found
# in a BIDS data directory.
#
# Usage:
# ./slurm/submit_all_subjects.sh --bids-dir path/to/data --config path/to/config.yaml --env hpc
# --------------------------------------------------------------------------------

set -e # Exit immediately on error

# --- Configuration ---
CONFIG_FILE="config/project_config.yaml"
ENV="hpc" # Explicitly set the environment for this script

# --- Argument Parsing ---
# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --bids-dir) BIDS_DIR_ARG="$2"; shift ;;
        --config) CONFIG_FILE="$2"; shift ;;
        --env) ENV="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Path Resolution ---
# If BIDS_DIR is not provided via command line, extract it from the config file
if [ -z "$BIDS_DIR_ARG" ]; then
    # This command uses Python's YAML parser to safely extract the path from the 'hpc' section
    BIDS_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['$ENV']['bids_dir'])")
else
    BIDS_DIR=$BIDS_DIR_ARG
fi

# --- Validation ---
if [ ! -d "$BIDS_DIR" ]; then
    echo "Error: BIDS directory not found at '$BIDS_DIR'"
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at '$CONFIG_FILE'"
    exit 1
fi

# --- Job Submission Loop ---
echo "Submitting behavioral analysis jobs for all subjects in: $BIDS_DIR"

for subj_dir in "$BIDS_DIR"/sub-*; do
    if [ -d "$subj_dir" ]; then
        subject_id=$(basename "$subj_dir")
        echo "Submitting job for subject: $subject_id"
        
        sbatch \
            --export=SUBJECT_ID="$subject_id",CONFIG_FILE="$CONFIG_FILE",ENV="$ENV" \
            slurm/submit_behavioral_analysis.sbatch
    fi
done

echo "All jobs submitted."
