#!/bin/bash
set -e  # Exit immediately if any command fails
#
# Dynamic SLURM array submission wrapper for LSS batch processing
# Automatically calculates correct array bounds based on subject count
#

# --- Configuration ---
: ${PROJECT_ROOT:?ERROR: PROJECT_ROOT not set - export PROJECT_ROOT=/path/to/project}

# Verify directories exist
if [[ ! -d "$PROJECT_ROOT" ]]; then
    echo "ERROR: PROJECT_ROOT directory does not exist: $PROJECT_ROOT"
    exit 1
fi

# --- Calculate Subject Count ---
cd "$PROJECT_ROOT"

# Ensure logs directory exists before submission
mkdir -p logs

# Read behavioral data path from config file
CONFIG_FILE="$PROJECT_ROOT/config/project_config.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Extract behavioral data path from config (assuming 'hpc' environment)
# This reads the onsets_dir from the hpc section of the config
BEHAVIORAL_DIR=$(python3 -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
print(config['hpc']['onsets_dir'])
")

if [[ ! -d "$BEHAVIORAL_DIR" ]]; then
    echo "ERROR: Behavioral data directory from config does not exist: $BEHAVIORAL_DIR"
    echo "Please check the 'onsets_dir' setting in your config file under the 'hpc' section"
    exit 1
fi

# Count subjects dynamically from config-specified behavioral directory
SUBJECT_COUNT=$(find "$BEHAVIORAL_DIR" -maxdepth 1 -type d -name "sub-*" | wc -l)

if [ "$SUBJECT_COUNT" -eq 0 ]; then
    echo "ERROR: No subjects found in $BEHAVIORAL_DIR"
    echo "Please check that your behavioral data is in the correct location"
    exit 1
fi

# Calculate max array index (0-based indexing)
MAX_INDEX=$((SUBJECT_COUNT - 1))

echo "Found $SUBJECT_COUNT subjects in $BEHAVIORAL_DIR"
echo "Setting SLURM array bounds to: 0-$MAX_INDEX"

# --- Submit Job with Dynamic Array Bounds ---
echo "Submitting LSS batch job with dynamic array bounds..."

# Submit directly with array bounds parameter (more robust than sed editing)
export PROJECT_ROOT
sbatch --array=0-"$MAX_INDEX" slurm/submit_lss_batch.sbatch

echo "LSS batch job submitted successfully with array bounds 0-$MAX_INDEX"
