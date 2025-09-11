#!/bin/bash
#SBATCH --job-name=behavioral
#SBATCH --output=logs/behavioral_%j.out
#SBATCH --error=logs/behavioral_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
# Note: Set SLURM_ACCOUNT and SLURM_PARTITION environment variables for your cluster
# Example: export SLURM_ACCOUNT=your_account SLURM_PARTITION=your_partition
# Uncomment and modify these lines for cluster-specific settings:
# #SBATCH --account=${SLURM_ACCOUNT}
# #SBATCH --partition=${SLURM_PARTITION}

# --- Script Initialization ---
set -e  # Exit immediately if any command fails

# --- Print job submission info ---
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Submitted from: $(pwd)"

# --- Shell Setup ---
# Require PROJECT_ROOT for consistent execution context
: ${PROJECT_ROOT:?ERROR: PROJECT_ROOT not set - export PROJECT_ROOT=/path/to/project}

# Change to project directory for consistent execution
cd "$PROJECT_ROOT"

# Ensure logs directory exists for output files
mkdir -p logs

# --- Setup ---
# Load the python module and activate your virtual environment
source slurm/load_python_module.sh
source .venv/bin/activate

# --- Command ---
# This script processes all subjects found in the BIDS directory.
python scripts/behavioral/calculate_discount_rates.py --env hpc

echo "Behavioral analysis complete."
