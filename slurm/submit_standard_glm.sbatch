#!/bin/bash
#
#SBATCH --job-name=standard_glm
#SBATCH --output=logs/standard_glm_%j_%x.out
#SBATCH --error=logs/standard_glm_%j_%x.err
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00

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
source .venv/bin/activate

# --- Validate Required Variables ---
: ${SUBJECT_ID:?ERROR: SUBJECT_ID not set - required for subject processing}
: ${CONFIG_FILE:?ERROR: CONFIG_FILE not set - required for configuration}
: ${ENV:?ERROR: ENV not set - required for environment selection}

# Verify config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file does not exist: $CONFIG_FILE"
    exit 1
fi

# --- Job Execution ---
echo "Running standard GLM for subject: $SUBJECT_ID"
echo "Config file: $CONFIG_FILE"
echo "Environment: $ENV"
# Arguments: config_path env subject_id (positional)
./.venv/bin/python scripts/modeling/run_standard_glm.py \
    "$CONFIG_FILE" \
    "$ENV" \
    "$SUBJECT_ID"
