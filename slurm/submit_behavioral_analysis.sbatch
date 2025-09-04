#!/bin/bash
#
#SBATCH --job-name=behavioral_analysis
#SBATCH --output=slurm/logs/behavioral_analysis_%j.out
#SBATCH --error=slurm/logs/behavioral_analysis_%j.err
#
# --------------------------------------------------------------------------------
# Resource Requests
# --------------------------------------------------------------------------------
#
#SBATCH --time=00:30:00  # Wall time limit (hh:mm:ss)
#SBATCH --nodes=1        # Number of nodes
#SBATCH --ntasks=1       # Number of tasks (usually 1 unless using MPI)
#SBATCH --cpus-per-task=2 # Number of CPUs per task
#SBATCH --mem=4G         # Memory per node (e.g., 4G, 8G, 16G)
#
# --------------------------------------------------------------------------------
# Job Script
# --------------------------------------------------------------------------------
#
# This script is a template for submitting a single subject's behavioral analysis.
# The following variables are expected to be passed from the wrapper script:
# - SUBJECT_ID: The ID of the subject to process.
# - CONFIG_FILE: The path to the project configuration file.
# - ENV: The environment to use from the config file ('local' or 'hpc').
# --------------------------------------------------------------------------------

# Set up the environment
set -e # Exit immediately if a command exits with a non-zero status.

# --- Module loading and environment activation ---
# Use default values for variables if they are not set.
# This allows the script to be run standalone for testing.
: "${CONFIG_FILE:=config/project_config.yaml}"
: "${ENV:=hpc}"

# TODO: Add any module load commands if necessary (e.g., for Python)
# module load python/3.9

# Activate your project's virtual environment
# Assumes you have created a virtual environment at the project root
source .venv/bin/activate

# --- Run the analysis ---
echo "Starting behavioral analysis for subject: $SUBJECT_ID"
echo "Using config file: $CONFIG_FILE"
echo "Using environment: $ENV"
echo "Job started at: $(date)"

python scripts/behavioral/calculate_discount_rates.py \
    --subjects "$SUBJECT_ID" \
    --config "$CONFIG_FILE" \
    --env "$ENV"

echo "Job finished at: $(date)"
