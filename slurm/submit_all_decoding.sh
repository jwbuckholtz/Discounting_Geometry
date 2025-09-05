#!/bin/bash
# Submits a SLURM job for each subject to run the MVPA decoding analysis.
# This script now uses a simplified sbatch template and CLI.

# --- Configuration ---
CONFIG_FILE="config/project_config.yaml"
ENV="hpc"
SBATCH_TEMPLATE="slurm/templates/submit_decoding_template.sbatch"

# --- Read paths from config ---
BIDS_DIR=$(python -c "import yaml; f=open('$CONFIG_FILE'); config=yaml.safe_load(f); print(config['$ENV']['bids_dir'])")
DERIVATIVES_DIR=$(python -c "import yaml; f=open('$CONFIG_FILE'); config=yaml.safe_load(f); print(config['$ENV']['derivatives_dir'])")

# --- Validation ---
if [ ! -d "$BIDS_DIR" ]; then echo "Error: BIDS_DIR not found at $BIDS_DIR"; exit 1; fi
if [ ! -f "$SBATCH_TEMPLATE" ]; then echo "Error: SBATCH template not found at $SBATCH_TEMPLATE"; exit 1; fi

# --- Job Submission Loop ---
for subject_dir in "$BIDS_DIR"/sub-*/; do
    if [ -d "$subject_dir" ]; then
        subject_id=$(basename "$subject_dir")
        
        # Define path to the subject's brain mask
        # This is a simplified example; a more robust solution would be needed if paths vary.
        MASK_PATH="$DERIVATIVES_DIR/fmriprep/$subject_id/ses-1/func/${subject_id}_ses-1_task-discountFix_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"

        echo "Submitting decoding job for subject: $subject_id"
        sbatch --export=ALL,SUBJECT_ID="$subject_id",CONFIG_FILE="$CONFIG_FILE",ENV="$ENV",BIDS_DIR="$BIDS_DIR",DERIVATIVES_DIR="$DERIVATIVES_DIR",MASK_PATH="$MASK_PATH" \
               "$SBATCH_TEMPLATE"
    fi
done

echo "All decoding jobs submitted."
