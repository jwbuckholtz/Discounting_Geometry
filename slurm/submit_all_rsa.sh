#!/bin/bash
# Submits a SLURM job for each subject to run the RSA (whole-brain and searchlight).

# --- Configuration ---
CONFIG_FILE="config/project_config.yaml"
ENV="hpc"
SBATCH_TEMPLATE="slurm/templates/submit_rsa_template.sbatch"
ANALYSIS_TYPES=("whole_brain" "searchlight")

# --- Load Environment ---
ml python/3.9

# --- Read paths from config ---
BIDS_DIR=$(python -c "import yaml; f=open('$CONFIG_FILE'); config=yaml.safe_load(f); print(config['$ENV']['bids_dir'])")

# --- Validation ---
if [ ! -d "$BIDS_DIR" ]; then echo "Error: BIDS_DIR not found at $BIDS_DIR"; exit 1; fi
if [ ! -f "$SBATCH_TEMPLATE" ]; then echo "Error: SBATCH template not found at $SBATCH_TEMPLATE"; exit 1; fi

# --- Job Submission Loop ---
for subject_dir in "$BIDS_DIR"/sub-*/; do
    if [ -d "$subject_dir" ]; then
        subject_id=$(basename "$subject_dir")
        
        for analysis_type in "${ANALYSIS_TYPES[@]}"; do
            echo "Submitting RSA job for subject: $subject_id, analysis: $analysis_type"
            sbatch --export=ALL,SUBJECT_ID="$subject_id",ANALYSIS_TYPE="$analysis_type",CONFIG_FILE="$CONFIG_FILE",ENV="$ENV" \
                   "$SBATCH_TEMPLATE"
        done
    fi
done

echo "All RSA jobs submitted."
