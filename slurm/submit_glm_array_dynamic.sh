#!/bin/bash
set -e  # Exit immediately if any command fails
#
# Dynamic SLURM array submission wrapper for GLM batch processing
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

# Read derivatives directory path from config file
CONFIG_FILE="$PROJECT_ROOT/config/project_config.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Extract derivatives directory path from config (assuming 'hpc' environment)
# Count subjects from derivatives/behavioral directory (where Python scripts look)
DERIVATIVES_DIR=$(python3 -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
print(config['hpc']['derivatives_dir'])
")

BEHAVIORAL_DIR="$DERIVATIVES_DIR/behavioral"

if [[ ! -d "$BEHAVIORAL_DIR" ]]; then
    echo "ERROR: Behavioral data directory does not exist: $BEHAVIORAL_DIR"
    echo "Please check the 'derivatives_dir' setting in your config file under the 'hpc' section"
    echo "Expected to find processed behavioral data in derivatives/behavioral/"
    exit 1
fi

# Count subjects dynamically from derivatives/behavioral directory
# Look for sub-* directories (where Python scripts expect processed data)
SUBJECT_COUNT=$(find "$BEHAVIORAL_DIR" -maxdepth 1 -type d -name "sub-*" | wc -l)

if [ "$SUBJECT_COUNT" -eq 0 ]; then
    echo "ERROR: No subject directories found in $BEHAVIORAL_DIR"
    echo "Looking for directories matching pattern: sub-*"
    echo "Please check that your processed behavioral data is in the correct location"
    echo "Expected structure: derivatives/behavioral/sub-sXXX/sub-sXXX_discounting_with_sv.tsv"
    exit 1
fi

# Calculate max array index (0-based indexing)
MAX_INDEX=$((SUBJECT_COUNT - 1))

echo "Found $SUBJECT_COUNT subjects in $BEHAVIORAL_DIR"
echo "Setting SLURM array bounds to: 0-$MAX_INDEX"

# --- Submit Job with Dynamic Array Bounds ---
echo "Submitting GLM batch job with dynamic array bounds..."

# Submit directly with array bounds parameter (more robust than sed editing)
export PROJECT_ROOT
sbatch --array=0-"$MAX_INDEX" slurm/submit_glm_batch.sbatch

echo "GLM batch job submitted successfully with array bounds 0-$MAX_INDEX"
