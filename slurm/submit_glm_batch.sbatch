#!/bin/bash
#
#SBATCH --job-name=glm_batch
#SBATCH --output=logs/glm_batch_%A_%a.out  # Standard output log, %A=job ID, %a=task ID
#SBATCH --error=logs/glm_batch_%A_%a.err   # Standard error log
#SBATCH --time=01:00:00                   # 1 hour
#SBATCH --mem=16G                         # 16 GB of memory (increased from 8G)
#SBATCH --nodes=1                         # 1 node
#SBATCH --ntasks=1                        # 1 task
#SBATCH --cpus-per-task=2                 # 2 CPUs
#SBATCH --array=0-$(($(find /oak/stanford/groups/russpold/users/buckholtz/Decoding_DD/output/behavioral/ -maxdepth 1 -type d -name "sub-*" -exec basename {} \; | sort | wc -l) - 1))

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

# --- SBATCH Array Directive ---
# We dynamically set the array size based on the number of subjects found.
# The last index is NUM_SUBJECTS - 1.
#SBATCH --array=0-$(($NUM_SUBJECTS - 1))

# --- Job Logic ---
# Use the SLURM task ID to select the correct subject from the list.
CURRENT_SUBJECT=${ALL_SUBJECTS[$SLURM_ARRAY_TASK_ID]}

echo "--- Starting Standard GLM ---"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing Subject: $CURRENT_SUBJECT"

# Run the Python script for the current subject.
python scripts/modeling/run_standard_glm.py --subject $CURRENT_SUBJECT --env hpc

echo "--- Finished GLM for $CURRENT_SUBJECT ---"
