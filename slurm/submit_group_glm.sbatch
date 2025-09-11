#!/bin/bash
#
#SBATCH --job-name=group_glm
#SBATCH --output=logs/group_glm_%j.out
#SBATCH --error=logs/group_glm_%j.err
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00

# --- Script Initialization ---
set -e  # Exit immediately if any command fails

# --- Environment Setup ---
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

# --- Load Modules ---
source slurm/load_python_module.sh

# --- Environment Setup ---
# Activate the virtual environment
source .venv/bin/activate

# Note: Package installation should be done once during environment setup,
# not on every job run. If needed, install dependencies manually:
# uv pip install -e . or pip install -e .

# --- Validate Required Variables ---
: ${CONTRAST:?ERROR: CONTRAST not set - required for group GLM analysis}
: ${CONFIG_FILE:?ERROR: CONFIG_FILE not set - required for configuration}
: ${ENV:?ERROR: ENV not set - required for environment selection}

# Verify config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file does not exist: $CONFIG_FILE"
    exit 1
fi

# --- Job Execution ---
echo "========================================="
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Contrast: $CONTRAST"
echo "Config File: $CONFIG_FILE"
echo "Environment: $ENV"
echo "========================================="

# Run the group-level GLM script
./.venv/bin/python scripts/group_level/run_group_glm.py \
    --config "$CONFIG_FILE" \
    --env "$ENV" \
    --contrast "$CONTRAST"

# --- Job Completion ---
echo "========================================="
echo "Job finished at $(date)"
echo "========================================="
