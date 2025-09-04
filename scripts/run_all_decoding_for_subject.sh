#!/bin/bash
#
# A wrapper script to run all decoding analyses for a single subject across
# all standard target variables and all ROIs.
#
# Usage:
#   ./scripts/run_all_decoding_for_subject.sh <subject_id>
#
# Example:
#   ./scripts/run_all_decoding_for_subject.sh sub-s061
#

# --- Configuration ---
# Stop the script if any command fails
set -e

# The subject ID is the first argument to the script
SUBJECT_ID=$1
if [ -z "$SUBJECT_ID" ]; then
    echo "Error: You must provide a subject ID as the first argument."
    echo "Usage: ./scripts/run_all_decoding_for_subject.sh <subject_id>"
    exit 1
fi

# Path to the directory containing all ROI masks
ROI_DIR="Masks/"

# List of all target variables to be decoded
TARGETS=(
    "choice"
    "later_delay"
    "SVchosen"
    "SVunchosen"
    "SVsum"
    "SVdiff"
)

# --- Main Loop ---
echo "--- Starting all decoding analyses for subject: ${SUBJECT_ID} ---"

# Loop through each target variable
for TARGET in "${TARGETS[@]}"; do
    echo ""
    echo "--------------------------------------------------------"
    echo "--- Running decoding for target: ${TARGET}"
    echo "--------------------------------------------------------"
    
    # Construct and execute the command
    COMMAND="./.venv/bin/python scripts/mvpa/run_decoding_analysis.py \
        --subject ${SUBJECT_ID} \
        --target ${TARGET} \
        --roi-path ${ROI_DIR}"
    
    echo "Executing: ${COMMAND}"
    eval ${COMMAND}
done

echo ""
echo "--- All decoding analyses for subject ${SUBJECT_ID} are complete. ---"
