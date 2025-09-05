#!/bin/bash
# Submits a SLURM job for each subject to run the first-level models (LSS and Standard).

# --- Configuration ---
CONFIG_FILE="config/project_config.yaml"
ENV="hpc"
SBATCH_TEMPLATE="slurm/templates/submit_modeling_template.sbatch"

# --- Read paths from config ---
BIDS_DIR=$(python -c "import yaml; f=open('$CONFIG_FILE'); config=yaml.safe_load(f); print(config['$ENV']['bids_dir'])")

# --- Validation ---
if [ ! -d "$BIDS_DIR" ]; then echo "Error: BIDS_DIR not found at $BIDS_DIR"; exit 1; fi
if [ ! -f "$SBATCH_TEMPLATE" ]; then echo "Error: SBATCH template not found at $SBATCH_TEMPLATE"; exit 1; fi

# --- Job Submission Loop ---
for subject_dir in "$BIDS_DIR"/sub-*/; do
    if [ -d "$subject_dir" ]; then
        subject_id=$(basename "$subject_dir")
        
        echo "Submitting modeling job for subject: $subject_id"
        sbatch --export=ALL,SUBJECT_ID="$subject_id",CONFIG_FILE="$CONFIG_FILE",ENV="$ENV" \
               "$SBATCH_TEMPLATE"
    fi
done

echo "All modeling jobs submitted."
