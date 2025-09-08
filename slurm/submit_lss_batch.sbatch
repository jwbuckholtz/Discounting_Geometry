#!/bin/bash
#
#SBATCH --job-name=lss_batch
#SBATCH --output=logs/lss_batch_%A_%a.out  # Standard output log
#SBATCH --error=logs/lss_batch_%A_%a.err   # Standard error log
#SBATCH --time=06:00:00                   # 6 hours (LSS is slow)
#SBATCH --mem=16G                         # 16 GB of memory
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

# --- Setup ---
# This command is crucial for SLURM job arrays. It makes sure that if the job is
# requeued, it starts from the first task again.
#SBATCH --requeue

# Get the total number of subjects from the behavioral directory
NUM_SUBJECTS=$(find derivatives/behavioral -maxdepth 1 -type d -name "sub-*" | wc -l)
# Set the array size to be the number of subjects, minus 1 (for 0-based indexing)
#SBATCH --array=0-$(($NUM_SUBJECTS - 1))

# --- Environment Setup ---
# Load the base Python module. This is necessary for the virtual environment to work.
module load python/3.9.0

# Activate your Python environment.
source /home/users/joshuawb/Code/Discounting_Geometry/.venv/bin/activate

# --- Subject List ---
# Create a list of all subject IDs from the behavioral directory
ALL_SUBJECTS=($(find derivatives/behavioral -maxdepth 1 -type d -name "sub-*" -exec basename {} \; | sort))

# --- Job Logic ---
# Use the SLURM task ID to select the correct subject from the full list
CURRENT_SUBJECT=${ALL_SUBJECTS[$SLURM_ARRAY_TASK_ID]}

echo "--- Starting LSS Model ---"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing Subject: $CURRENT_SUBJECT"

# Run the Python script for the current subject
python scripts/modeling/run_lss_model.py --subject $CURRENT_SUBJECT --env hpc

echo "--- Finished LSS for $CURRENT_SUBJECT ---"
