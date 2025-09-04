#!/bin/bash
#
# This script runs the t-SNE visualization for a single subject across all 
# specified behavioral variables and all ROI masks found in a given directory.
#
# Usage:
# ./scripts/visualization/run_all_visualizations_for_subject.sh <subject_id> [roi_dir]
#
# Example:
# ./scripts/visualization/run_all_visualizations_for_subject.sh sub-s061 Masks/
#
# Arguments:
#   subject_id : The BIDS subject ID (e.g., 'sub-s061').
#   roi_dir    : Optional. The directory containing ROI mask files (.nii.gz).
#                Defaults to 'Masks/'.
#
# The script will generate a .png plot for each combination of ROI and variable
# in the derivatives/visualization/<subject_id>/ directory.

# --- Configuration ---
set -e # Exit immediately if a command exits with a non-zero status.

# Check for subject ID argument
if [ -z "$1" ]; then
    echo "Error: Subject ID is required."
    echo "Usage: $0 <subject_id> [roi_dir]"
    exit 1
fi

SUBJECT_ID=$1
ROI_DIR=${2:-"Masks/"} # Use the second argument or default to "Masks/"

# List of behavioral variables to color the plots by
COLOR_BY_VARS=(
    "choice"
    "later_delay"
    "SVchosen"
    "SVunchosen"
    "SVsum"
    "SVdiff"
)

# --- Validation ---
# Check if ROI directory exists
if [ ! -d "$ROI_DIR" ]; then
    echo "Error: ROI directory not found at '$ROI_DIR'"
    exit 1
fi

# Check if the python script exists
PYTHON_SCRIPT="scripts/visualization/plot_embeddings.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at '$PYTHON_SCRIPT'"
    exit 1
fi

# Check if the virtual environment exists
VENV_PYTHON="./.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Python virtual environment not found. Please run setup first."
    exit 1
fi


# --- Main Loop ---
echo "Starting visualization generation for subject: $SUBJECT_ID"
echo "Using ROIs from: $ROI_DIR"
echo "Coloring by variables: ${COLOR_BY_VARS[*]}"
echo "-----------------------------------------------------"

# Find all NIfTI files in the ROI directory
for roi_file in "$ROI_DIR"/*.nii.gz; do
    if [ -f "$roi_file" ]; then
        roi_name=$(basename "$roi_file" .nii.gz)
        echo ""
        echo "Processing ROI: $roi_name"
        
        # Loop through each behavioral variable
        for variable in "${COLOR_BY_VARS[@]}"; do
            echo "  - Generating plot for variable: $variable"
            
            # Construct and execute the command
            "$VENV_PYTHON" "$PYTHON_SCRIPT" \
                --subject "$SUBJECT_ID" \
                --roi-path "$roi_file" \
                --color-by "$variable" \
                --method "tsne"
        done
    fi
done

echo ""
echo "-----------------------------------------------------"
echo "All visualizations generated successfully for subject $SUBJECT_ID."
echo "Output files are located in: derivatives/visualization/$SUBJECT_ID/"
