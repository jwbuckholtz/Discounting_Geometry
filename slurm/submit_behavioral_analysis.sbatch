#!/bin/bash
#SBATCH --job-name=behavioral
#SBATCH --output=logs/behavioral_%j.out
#SBATCH --error=logs/behavioral_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --account=russpold
#SBATCH --partition=russpold

# --- Print job submission info ---
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Submitted from: $(pwd)"

# --- Setup ---
source /oak/stanford/groups/russpold/users/buckholtz/Decoding_DD/setup.sh
# --- Command ---
# This script processes all subjects found in the BIDS directory.
python scripts/behavioral/calculate_discount_rates.py --env hpc

echo "Behavioral analysis complete."
