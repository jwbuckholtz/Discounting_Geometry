#!/bin/bash
#
# This wrapper script submits SLURM jobs to generate t-SNE visualizations
# for all subjects found in a BIDS directory.
#
# Usage:
# ./slurm/submit_all_visualizations.sh [--bids-dir /path/to/bids] [--roi-dir /path/to/rois]
#
# Arguments:
#   --bids-dir : Optional. Path to the BIDS data directory. If not provided,
#                it will be read from the project_config.yaml file.
#   --roi-dir  : Optional. Path to the directory containing ROI masks.
#                Defaults to 'Masks/'.
#
set -e # Exit immediately on error

# --- Configuration ---
CONFIG_FILE="config/project_config.yaml"
ENV="hpc"
ROI_DIR_DEFAULT="Masks/"

# --- Argument Parsing ---
# Use a while loop to parse arguments
BIDS_DIR_ARG=""
ROI_DIR_ARG=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --bids-dir) BIDS_DIR_ARG="$2"; shift ;;
        --roi-dir) ROI_DIR_ARG="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Path Resolution ---
# If BIDS_DIR is not provided via command line, extract it from the config file
if [ -z "$BIDS_DIR_ARG" ]; then
    # This command uses Python's YAML parser to safely extract the path
    BIDS_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['$ENV']['bids_dir'])")
else
    BIDS_DIR=$BIDS_DIR_ARG
fi

ROI_DIR=${ROI_DIR_ARG:-$ROI_DIR_DEFAULT}

# --- Validation ---
if [ ! -d "$BIDS_DIR" ]; then
    echo "Error: BIDS directory not found at '$BIDS_DIR'"
    exit 1
fi
if [ ! -d "$ROI_DIR" ]; then
    echo "Error: ROI directory not found at '$ROI_DIR'"
    exit 1
fi

# --- Job Submission Loop ---
echo "Submitting visualization jobs for all subjects in: $BIDS_DIR"
echo "Using ROIs from: $ROI_DIR"

for subj_dir in "$BIDS_DIR"/sub-*; do
    if [ -d "$subj_dir" ]; then
        subject_id=$(basename "$subj_dir")
        echo "Submitting job for subject: $subject_id"
        
        sbatch \
            --export=SUBJECT_ID="$subject_id",ROI_DIR="$ROI_DIR",CONFIG_FILE="$CONFIG_FILE",ENV="$ENV" \
            slurm/submit_visualization.sbatch
    fi
done

echo "All jobs submitted."
