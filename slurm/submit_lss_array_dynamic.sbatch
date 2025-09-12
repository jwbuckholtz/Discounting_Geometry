#!/bin/bash
#SBATCH --job-name=lss_array_dynamic
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/lss_array_dynamic_%j.out
#SBATCH --error=logs/lss_array_dynamic_%j.err
#SBATCH --account=russpold
#SBATCH --partition=russpold

# Dynamic SLURM array submission wrapper for LSS batch processing
# This sbatch script automatically calculates correct array bounds and submits LSS jobs
# Usage: sbatch slurm/submit_lss_array_dynamic.sbatch

set -e  # Exit immediately if any command fails

echo "Starting dynamic LSS array submission at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"

# --- Environment Validation ---
: ${PROJECT_ROOT:?ERROR: PROJECT_ROOT not set - export PROJECT_ROOT=/path/to/project}

# Verify PROJECT_ROOT directory exists
if [[ ! -d "$PROJECT_ROOT" ]]; then
    echo "ERROR: PROJECT_ROOT directory does not exist: $PROJECT_ROOT"
    exit 1
fi

echo "PROJECT_ROOT: $PROJECT_ROOT"

# Change to project directory
cd "$PROJECT_ROOT"

# Verify config file exists
CONFIG_FILE="$PROJECT_ROOT/config/project_config.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Config file: $CONFIG_FILE"

# --- Calculate Subject Count ---
echo "Calculating subject count..."

# Read derivatives directory path from config file
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
    echo "You may need to run behavioral analysis first:"
    echo "  sbatch slurm/submit_behavioral_analysis.sbatch"
    exit 1
fi

# Count subjects dynamically from derivatives/behavioral directory
SUBJECT_COUNT=$(find "$BEHAVIORAL_DIR" -maxdepth 1 -type d -name "sub-*" | wc -l)

if [ "$SUBJECT_COUNT" -eq 0 ]; then
    echo "ERROR: No subject directories found in $BEHAVIORAL_DIR"
    echo "Looking for directories matching pattern: sub-*"
    echo "Please check that your processed behavioral data is in the correct location"
    echo "Expected structure: derivatives/behavioral/sub-sXXX/sub-sXXX_discounting_with_sv.tsv"
    echo ""
    echo "You may need to run behavioral analysis first:"
    echo "  sbatch slurm/submit_behavioral_analysis.sbatch"
    exit 1
fi

# Calculate max array index (0-based indexing)
MAX_INDEX=$((SUBJECT_COUNT - 1))

echo "Found $SUBJECT_COUNT subjects in $BEHAVIORAL_DIR"
echo "Setting SLURM array bounds to: 0-$MAX_INDEX"

# Ensure logs directory exists for the array jobs
mkdir -p logs

# --- Submit LSS Array Job ---
echo "Submitting LSS batch job with dynamic array bounds..."

# Submit directly with array bounds parameter
export PROJECT_ROOT
ARRAY_JOB_ID=$(sbatch --array=0-"$MAX_INDEX" --parsable slurm/submit_lss_batch.sbatch)

echo "LSS batch job submitted successfully!"
echo "Array job ID: $ARRAY_JOB_ID"
echo "Array bounds: 0-$MAX_INDEX"
echo "Total subjects: $SUBJECT_COUNT"
echo ""
echo "Monitor job progress with:"
echo "  squeue -u \$USER"
echo "  squeue -j $ARRAY_JOB_ID"
echo ""
echo "Check logs with:"
echo "  ls logs/lss_batch_${ARRAY_JOB_ID}_*.out"
echo "  tail -f logs/lss_batch_${ARRAY_JOB_ID}_0.out  # First subject"

echo "Dynamic LSS array submission completed at $(date)"
