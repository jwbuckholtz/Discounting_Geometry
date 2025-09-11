#!/bin/bash
set -e  # Exit immediately if any command fails
#
# Prepare SLURM environment for job submission
# Ensures all required directories and environment variables are set up correctly
#

echo "=== SLURM Environment Preparation ==="

# --- Validate Required Environment Variables ---
: ${PROJECT_ROOT:?ERROR: PROJECT_ROOT not set - export PROJECT_ROOT=/path/to/project}

# Verify PROJECT_ROOT exists
if [[ ! -d "$PROJECT_ROOT" ]]; then
    echo "ERROR: PROJECT_ROOT directory does not exist: $PROJECT_ROOT"
    exit 1
fi

echo "✓ PROJECT_ROOT validated: $PROJECT_ROOT"

# --- Change to Project Directory ---
cd "$PROJECT_ROOT"
echo "✓ Changed to project directory: $(pwd)"

# --- Create Required Directories ---
echo "Creating required directories..."

# Main logs directory
mkdir -p logs
echo "✓ Created logs/ directory"

# SLURM-specific logs directory (if needed)
mkdir -p slurm/logs
echo "✓ Created slurm/logs/ directory"

# Derivatives directories
mkdir -p derivatives
mkdir -p derivatives/standard_glm
mkdir -p derivatives/lss
mkdir -p derivatives/visualization
echo "✓ Created derivatives directories"

# --- Validate Common Tools ---
echo "Validating environment..."

# Check for Python
if command -v python &> /dev/null; then
    echo "✓ Python available: $(python --version)"
else
    echo "⚠ Python not found - may need to load module"
fi

# Check for sbatch
if command -v sbatch &> /dev/null; then
    echo "✓ SLURM available: $(sbatch --version | head -1)"
else
    echo "⚠ SLURM not available - not on HPC system"
fi

# --- Check Virtual Environment ---
if [[ -d ".venv" ]]; then
    echo "✓ Virtual environment found: .venv/"
else
    echo "⚠ Virtual environment not found - may need to create .venv/"
fi

# --- Check Configuration ---
if [[ -f "config/project_config.yaml" ]]; then
    echo "✓ Configuration file found: config/project_config.yaml"
else
    echo "⚠ Configuration file not found: config/project_config.yaml"
fi

echo ""
echo "=== Environment Preparation Complete ==="
echo "You can now submit SLURM jobs safely."
echo ""
echo "Usage examples:"
echo "  ./slurm/submit_glm_array_dynamic.sh"
echo "  ./slurm/submit_lss_array_dynamic.sh"
echo "  sbatch --array=0-N slurm/submit_glm_batch.sbatch"
echo ""
