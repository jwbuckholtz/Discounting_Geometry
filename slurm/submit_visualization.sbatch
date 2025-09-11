#!/bin/bash
#
#SBATCH --job-name=dd_visualization
#SBATCH --output=logs/visualization_%j.out
#SBATCH --error=logs/visualization_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# --- Environment Setup ---
set -e # Exit immediately if a command exits with a non-zero status.

# Validate required environment variables
: "${SUBJECT_ID:?ERROR: SUBJECT_ID not set - required for subject processing}"
: "${ROI_DIR:?ERROR: ROI_DIR not set - required for ROI directory path}"

# Verify ROI directory exists
if [[ ! -d "$ROI_DIR" ]]; then
    echo "ERROR: ROI directory does not exist: $ROI_DIR"
    exit 1
fi

# Note: CONFIG_FILE and ENV are not used by the visualization script
# The visualization script only requires SUBJECT_ID and ROI_DIR

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

# Load the Python module to make libraries available on the compute node
source slurm/load_python_module.sh

echo "--- Job Details ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Subject ID: $SUBJECT_ID"
echo "ROI Directory: $ROI_DIR"
echo "--------------------"

# --- Activate Conda/Venv Environment ---
# Modify this line to activate your specific Python environment
# e.g., source /path/to/your/venv/bin/activate
source .venv/bin/activate

# --- Execute the Script ---
echo "Running visualization script..."

# Capture exit code before set -e can interfere
set +e  # Temporarily disable exit-on-error
./scripts/visualization/run_all_visualizations_for_subject.sh "$SUBJECT_ID" "$ROI_DIR"
EXIT_CODE=$?
set -e  # Re-enable exit-on-error

echo "Job finished with exit code $EXIT_CODE"

# Exit with the actual script's exit code
exit $EXIT_CODE
