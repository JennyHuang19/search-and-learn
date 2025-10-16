#!/bin/bash
#
#SBATCH --job-name=gen_bo1
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=04:00:00
#SBATCH --output=/tmp/bo1_%j.log
#SBATCH --error=/tmp/bo1_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -e  # Exit on error
set -x  # Print commands

# Print debug info
echo "Starting job on $(hostname) at $(date)"
echo "Working directory: $(pwd)"
echo "User: $(whoami)"
echo "PATH: $PATH"

# Initialize conda with full path
if [ -f "/afs/csail.mit.edu/u/j/jhuang9/miniconda3/etc/profile.d/conda.sh" ]; then
    source /afs/csail.mit.edu/u/j/jhuang9/miniconda3/etc/profile.d/conda.sh
    conda activate sal
else
    echo "Conda not found!"
    exit 1
fi

# Verify Python
which python
python --version

# Change to working directory
cd /afs/csail.mit.edu/u/j/jhuang9/search-and-learn

# Run the generation script
python chat-scripts/generate-only.py --config chat-recipes/bo1.yaml
