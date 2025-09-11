#!/bin/bash
set -e  # Exit immediately if any command fails
#
#SBATCH --job-name=lss_batch
#SBATCH --output=logs/lss_batch_%A_%a.out  # Standard output log
#SBATCH --error=logs/lss_batch_%A_%a.err   # Standard error log
#SBATCH --time=10:00:00                   # 10 hours (LSS is very slow)
#SBATCH --mem=64G                         # 64 GB of memory (increased from 32G)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

# --- Setup ---
# Set array bounds to match the actual number of subjects (0-102 for 103 subjects)  
# To determine correct bounds, run: ./slurm/get_subject_count.sh
# This prevents unnecessary out-of-bounds task submissions and wasted compute resources
#SBATCH --array=0-102

# This command is crucial for SLURM job arrays. It makes sure that if the job is
# requeued, it starts from the first task again.
#SBATCH --requeue

# --- Shell Setup ---
# Change to the project root directory for consistent execution context
cd "$PROJECT_ROOT"

# Ensure logs directory exists for output files
mkdir -p logs

# --- Environment Setup ---
# Require all paths via environment variables for portability
# Users must set these before submitting jobs
: ${PROJECT_ROOT:?ERROR: PROJECT_ROOT not set - export PROJECT_ROOT=/path/to/project}
: ${BEHAVIORAL_DIR:?ERROR: BEHAVIORAL_DIR not set - export BEHAVIORAL_DIR=/path/to/behavioral/data}

# Verify directories exist before proceeding
if [[ ! -d "$PROJECT_ROOT" ]]; then
    echo "ERROR: PROJECT_ROOT directory does not exist: $PROJECT_ROOT"
    exit 1
fi

if [[ ! -d "$BEHAVIORAL_DIR" ]]; then
    echo "ERROR: BEHAVIORAL_DIR directory does not exist: $BEHAVIORAL_DIR"
    exit 1
fi

# Load the base Python module. This is necessary for the virtual environment to work.
module load python/3.9.0

# Activate your Python environment.
source "${PROJECT_ROOT}/.venv/bin/activate"

# --- Subject List ---
# Create a list of all subject IDs from the behavioral directory (keep find command on one line)
ALL_SUBJECTS=($(find "${BEHAVIORAL_DIR}" -maxdepth 1 -type d -name "sub-*" -exec basename {} \; | sort))
NUM_SUBJECTS=${#ALL_SUBJECTS[@]}

# --- Job Logic ---
# Check if the current task ID is valid for the number of subjects we have.
if [ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_SUBJECTS" ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID is out of bounds for subject list of size $NUM_SUBJECTS. Exiting."
    exit 0
fi

# Use the SLURM task ID to select the correct subject from the full list
CURRENT_SUBJECT="${ALL_SUBJECTS[$SLURM_ARRAY_TASK_ID]}"

echo "--- Starting LSS Model ---"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing Subject: $CURRENT_SUBJECT"

# Run the Python script for the current subject
# Arguments: --config config_path --env environment --subject subject_id
python scripts/modeling/run_lss_model.py --config config/project_config.yaml --env hpc --subject "$CURRENT_SUBJECT"

echo "--- Finished LSS for $CURRENT_SUBJECT ---"
