#!/bin/bash
#
# Helper script to determine the number of subjects for SLURM array jobs
# This ensures array bounds match the actual subject count
#

# Path to behavioral data (adjust if needed)
BEHAVIORAL_DIR="/oak/stanford/groups/russpold/users/buckholtz/Decoding_DD/output/behavioral/"

# Check if directory exists
if [ ! -d "$BEHAVIORAL_DIR" ]; then
    echo "Error: Behavioral directory not found: $BEHAVIORAL_DIR" >&2
    exit 1
fi

# Count subjects (directories matching "sub-*" pattern)
SUBJECT_COUNT=$(find "$BEHAVIORAL_DIR" -maxdepth 1 -type d -name "sub-*" | wc -l)

# Validate we found subjects
if [ "$SUBJECT_COUNT" -eq 0 ]; then
    echo "Error: No subjects found in $BEHAVIORAL_DIR" >&2
    exit 1
fi

# Calculate max array index (0-based indexing)
MAX_INDEX=$((SUBJECT_COUNT - 1))

echo "Found $SUBJECT_COUNT subjects"
echo "SLURM array bounds should be: 0-$MAX_INDEX"
echo "Use: #SBATCH --array=0-$MAX_INDEX"
