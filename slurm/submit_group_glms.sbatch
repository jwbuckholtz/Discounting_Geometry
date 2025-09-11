#!/bin/bash
#
#SBATCH --job-name=group_glm
#SBATCH --output=logs/group_glm_%j.out
#SBATCH --error=logs/group_glm_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

# --- Environment Validation (BEFORE any usage) ---
set -e # Exit if any command fails

# Require PROJECT_ROOT for consistent execution context
: ${PROJECT_ROOT:?ERROR: PROJECT_ROOT not set - export PROJECT_ROOT=/path/to/project}

# Verify PROJECT_ROOT exists
if [[ ! -d "$PROJECT_ROOT" ]]; then
    echo "ERROR: PROJECT_ROOT directory does not exist: $PROJECT_ROOT"
    exit 1
fi

# --- Shell Setup ---
# Change to the project root directory for consistent execution context
cd "$PROJECT_ROOT"

# Ensure logs directory exists for output files
mkdir -p logs

# Verify config file exists
if [[ ! -f "config/project_config.yaml" ]]; then
    echo "ERROR: Config file not found: config/project_config.yaml"
    echo "Make sure you're running from the project root directory"
    exit 1
fi

# Activate your project's Python environment
source slurm/load_python_module.sh
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
