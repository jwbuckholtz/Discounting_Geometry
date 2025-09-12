#!/bin/bash
set -e  # Exit immediately if any command fails
#
#SBATCH --job-name=glm_batch
#SBATCH --output=logs/glm_batch_%A_%a.out  # Standard output log, %A=job ID, %a=task ID
#SBATCH --error=logs/glm_batch_%A_%a.err   # Standard error log
#SBATCH --time=01:00:00                   # 1 hour
#SBATCH --mem=32G                         # 32 GB of memory (increased from 16G)
#SBATCH --nodes=1                         # 1 node
#SBATCH --ntasks=1                        # 1 task
#SBATCH --cpus-per-task=2                 # 2 CPUs
# --- SBATCH Array Directive ---
# NOTE: Array bounds should be passed dynamically via submission wrapper:
# Use: ./slurm/submit_glm_array_dynamic.sh (recommended)
# Or:  sbatch --array=0-N slurm/submit_glm_batch.sbatch
# Hard-coded bounds removed to prevent stale values
# #SBATCH --array=0-102  # REMOVED: Use dynamic submission instead

# --- Environment Validation (BEFORE any usage) ---
# Require PROJECT_ROOT via environment variable for portability
# Users must set this before submitting jobs
: ${PROJECT_ROOT:?ERROR: PROJECT_ROOT not set - export PROJECT_ROOT=/path/to/project}

# Verify directories exist before proceeding
if [[ ! -d "$PROJECT_ROOT" ]]; then
    echo "ERROR: PROJECT_ROOT directory does not exist: $PROJECT_ROOT"
    exit 1
fi

# Verify config file exists
CONFIG_FILE="$PROJECT_ROOT/config/project_config.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file does not exist: $CONFIG_FILE"
    exit 1
fi

# --- Shell Setup ---
# Change to the project root directory for consistent execution context
cd "$PROJECT_ROOT"

# Ensure logs directory exists for output files
mkdir -p logs

# Load Python module with compatibility check
source slurm/load_python_module.sh

# Activate your Python environment.
source "${PROJECT_ROOT}/.venv/bin/activate"

# --- Subject List ---
# Read behavioral data path from config file
CONFIG_FILE="$PROJECT_ROOT/config/project_config.yaml"
BEHAVIORAL_DIR=$(python3 -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
print(config['hpc']['onsets_dir'])
")

# Create a list of all subject IDs from the config-specified behavioral directory
# Extract subject IDs from sxx_discountfix_events.tsv filenames
ALL_SUBJECTS=($(find "${BEHAVIORAL_DIR}" -maxdepth 1 -name "s*_discountfix_events.tsv" -exec basename {} \; | sed 's/_discountfix_events\.tsv$//' | sort))
NUM_SUBJECTS=${#ALL_SUBJECTS[@]}

# --- Job Logic ---
# Check if the current task ID is valid for the number of subjects we have.
if [ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_SUBJECTS" ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID is out of bounds for subject list of size $NUM_SUBJECTS. Exiting."
    exit 0
fi

# Use the SLURM task ID to select the correct subject from the list.
CURRENT_SUBJECT="${ALL_SUBJECTS[$SLURM_ARRAY_TASK_ID]}"

echo "--- Starting Standard GLM ---"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing Subject: $CURRENT_SUBJECT"

# Run the Python script for the current subject.
# Arguments: config_path env subject_id
python scripts/modeling/run_standard_glm.py "$CONFIG_FILE" hpc "$CURRENT_SUBJECT"

echo "--- Finished GLM for $CURRENT_SUBJECT ---"
