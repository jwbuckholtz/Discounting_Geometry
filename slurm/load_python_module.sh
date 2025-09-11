#!/bin/bash
# load_python_module.sh
# Compatibility function for loading Python module across different HPC systems
#
# Usage: source slurm/load_python_module.sh

load_python_module() {
    # Try ml first (modern Lmod), then module load (traditional), then warn
    if command -v ml &> /dev/null; then
        echo "Loading Python module using 'ml'..."
        ml python/3.9
    elif command -v module &> /dev/null; then
        echo "Loading Python module using 'module load'..."
        module load python/3.9.0
    else
        echo "WARNING: Neither 'ml' nor 'module load' commands available"
        echo "Attempting to proceed without loading Python module"
        echo "If you encounter Python errors, manually load the Python module"
    fi
}

# Call the function automatically when sourced
load_python_module
