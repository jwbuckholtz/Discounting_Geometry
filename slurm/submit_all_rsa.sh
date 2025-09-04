#!/bin/bash
#
# This script submits a SLURM job for each subject's RSA.
#
# Usage:
# ./slurm/submit_all_rsa.sh --analysis-type <type> [--roi-path <path>]
#
# Examples:
# ./slurm/submit_all_rsa.sh --analysis-type whole_brain
# ./slurm/submit_all_rsa.sh --analysis-type searchlight
# ./slurm/submit_all_rsa.sh --analysis-type roi --roi-path /path/to/rois
# --------------------------------------------------------------------------------

# --- Default Configuration ---
BIDS_DIR="data"
CONFIG_FILE="config/project_config.yaml"
ENV="hpc"
SBATCH_TEMPLATE="slurm/submit_rsa_analysis.sbatch"
ANALYSIS_TYPE=""
ROI_PATH=""

# --- Argument Parsing ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --analysis-type) ANALYSIS_TYPE="$2"; shift ;;
        --roi-path) ROI_PATH="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Validation ---
if [ -z "$ANALYSIS_TYPE" ]; then
    echo "Error: --analysis-type is a required argument."
    exit 1
fi
if [ "$ANALYSIS_TYPE" = "roi" ] && [ -z "$ROI_PATH" ]; then
    echo "Error: --roi-path must be provided for 'roi' analysis type."
    exit 1
fi
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
        echo "Submitting RSA job for subject: $subject_id, type: $ANALYSIS_TYPE"
        
        sbatch --export=ALL,SUBJECT_ID="$subject_id",ANALYSIS_TYPE="$ANALYSIS_TYPE",ROI_PATH="$ROI_PATH",CONFIG_FILE="$CONFIG_FILE",ENV="$ENV" \
               "$SBATCH_TEMPLATE"
    fi
done

echo "All RSA jobs submitted."
