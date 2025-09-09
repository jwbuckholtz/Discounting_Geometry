#!/bin/bash
#
#SBATCH --job-name=lss_batch
#SBATCH --output=logs/lss_batch_%A_%a.out  # Standard output log
#SBATCH --error=logs/lss_batch_%A_%a.err   # Standard error log
#SBATCH --time=10:00:00                   # 10 hours (LSS is very slow)
#SBATCH --mem=16G                         # 16 GB of memory
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

# --- Setup ---
# Set a high-level upper bound for the number of subjects.
# The script will exit gracefully if the task ID is out of bounds.
#SBATCH --array=0-199

# This command is crucial for SLURM job arrays. It makes sure that if the job is
# requeued, it starts from the first task again.
#SBATCH --requeue

# --- Shell Setup ---
# Change to the directory where you are submitting the script from
cd $SLURM_SUBMIT_DIR

# --- Environment Setup ---
# Load the base Python module. This is necessary for the virtual environment to work.
module load python/3.9.0

# Activate your Python environment.
source /home/users/joshuawb/Code/Discounting_Geometry/.venv/bin/activate

# --- Subject List ---
# Create a list of all subject IDs from the behavioral directory
ALL_SUBJECTS=($(find /oak/stanford/groups/russpold/users/buckholtz/Decoding_DD/output/behavioral/ -maxdepth 1 -type d -name "sub-*" -exec basename {} \; | sort))
NUM_SUBJECTS=${#ALL_SUBJECTS[@]}

# --- Job Logic ---
# Check if the current task ID is valid for the number of subjects we have.
if [ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_SUBJECTS" ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID is out of bounds for subject list of size $NUM_SUBJECTS. Exiting."
    exit 0
fi

# Use the SLURM task ID to select the correct subject from the full list
CURRENT_SUBJECT=${ALL_SUBJECTS[$SLURM_ARRAY_TASK_ID]}

echo "--- Starting LSS Model ---"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing Subject: $CURRENT_SUBJECT"

# Run the Python script for the current subject
python scripts/modeling/run_lss_model.py --subject $CURRENT_SUBJECT --env hpc

echo "--- Finished LSS for $CURRENT_SUBJECT ---"
