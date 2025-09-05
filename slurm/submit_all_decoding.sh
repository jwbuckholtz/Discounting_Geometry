#!/bin/bash
#
# This script submits a SLURM job for each subject and each target variable.
#
# Usage:
# ./slurm/submit_all_decoding.sh
# --------------------------------------------------------------------------------

# --- Default Configuration ---
BIDS_DIR="" # This will be read from the config file
CONFIG_FILE="config/project_config.yaml"
ENV="hpc"
SBATCH_TEMPLATE="slurm/submit_decoding_analysis.sbatch"

# Define the list of target variables you want to decode
TARGETS=(
    "choice"
    "delay_to_reward"
    "SVchosen"
    "SVunchosen"
    "SVsum"
    "SVdiff"
)

# --- Argument Parsing ---
while getopts ":d:" opt; do
    case $opt in
        d)
            BIDS_DIR="$OPTARG"
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
    shift
done

# --- Read BIDS directory from config if not provided ---
if [ -z "$BIDS_DIR" ]; then
    # Use python to parse the yaml file to avoid adding a shell-based yaml parser dependency
    BIDS_DIR=$(python -c "import yaml; f = open('$CONFIG_FILE'); config = yaml.safe_load(f); print(config['$ENV']['bids_dir'])")
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
if [ ! -f "$SBATCH_TEMPLATE" ]; then
    echo "Error: SBATCH template not found at '$SBATCH_TEMPLATE'"
    exit 1
fi

# --- Job Submission Loop ---
for subject_dir in "$BIDS_DIR"/sub-*/; do
    if [ -d "$subject_dir" ]; then
        subject_id=$(basename "$subject_dir")
        
        for target in "${TARGETS[@]}"; do
            echo "Submitting decoding job for subject: $subject_id, target: $target"
            
            sbatch --export=ALL,SUBJECT_ID="$subject_id",TARGET="$target",CONFIG_FILE="$CONFIG_FILE",ENV="$ENV" \
                   "$SBATCH_TEMPLATE"
        done
    fi
done

echo "All decoding jobs submitted."
