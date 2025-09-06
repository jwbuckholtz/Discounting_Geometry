#!/bin/bash
# Submits a SLURM job for each subject and each target variable for MVPA.

# --- Configuration ---
CONFIG_FILE="config/project_config.yaml"
ENV="hpc"
SBATCH_TEMPLATE="slurm/templates/submit_decoding_template.sbatch"

# --- Load Environment ---
ml python/3.9

# --- Read paths and targets from config ---
BIDS_DIR=$(python -c "import yaml; f=open('$CONFIG_FILE'); config=yaml.safe_load(f); print(config['$ENV']['bids_dir'])")
DERIVATIVES_DIR=$(python -c "import yaml; f=open('$CONFIG_FILE'); config=yaml.safe_load(f); print(config['$ENV']['derivatives_dir'])")
TARGETS=$(python -c "import yaml; f=open('$CONFIG_FILE'); config=yaml.safe_load(f); print(' '.join(config['analysis_params']['mvpa']['targets']))")

# --- Validation ---
if [ ! -d "$BIDS_DIR" ]; then echo "Error: BIDS_DIR not found at $BIDS_DIR"; exit 1; fi
if [ ! -f "$SBATCH_TEMPLATE" ]; then echo "Error: SBATCH template not found at $SBATCH_TEMPLATE"; exit 1; fi

# --- Job Submission Loop ---
for subject_dir in "$BIDS_DIR"/sub-*/; do
    if [ -d "$subject_dir" ]; then
        subject_id=$(basename "$subject_dir")
        
        for target in $TARGETS; do
            echo "Submitting decoding job for subject: $subject_id, target: $target"
            sbatch --export=ALL,SUBJECT_ID="$subject_id",DERIVATIVES_DIR="$DERIVATIVES_DIR",TARGET="$target",CONFIG_FILE="$CONFIG_FILE",ENV="$ENV" \
                   "$SBATCH_TEMPLATE"
        done
    fi
done

echo "All decoding jobs submitted."
