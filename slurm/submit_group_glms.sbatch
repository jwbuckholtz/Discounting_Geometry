#!/bin/bash
#
#SBATCH --job-name=group_glm
#SBATCH --output=logs/group_glm_%j.out
#SBATCH --error=logs/group_glm_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

# --- Setup ---
set -e # Exit if any command fails

# Activate your project's Python environment
ml python/3.9
source .venv/bin/activate

# Define the list of contrasts to run group-level models on
# These should match the contrast names from your project_config.yaml
CONTRASTS=(
    "decision"
    "choice"
    "SVchosen"
    "SVunchosen"
    "SVsum"
    "SVdiff"
)

echo "--- Starting Group-Level GLM Analysis ---"
echo "Found ${#CONTRASTS[@]} contrasts to analyze."

# --- Main Loop ---
# Loop through each contrast and run the group-level script
for contrast in "${CONTRASTS[@]}"; do
    echo ""
    echo "--- Running analysis for contrast: $contrast ---"
    
    python scripts/group_level/run_group_glm.py \
        --config "config/project_config.yaml" \
        --env "hpc" \
        --contrast "$contrast"
        
    echo "--- Finished analysis for contrast: $contrast ---"
done

echo ""
echo "--- All Group-Level GLM Analyses Complete ---"
