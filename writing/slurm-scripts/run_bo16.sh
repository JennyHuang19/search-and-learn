#!/bin/bash
#
#SBATCH --job-name=gen_bo16
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=24:00:00
#SBATCH --output=./writing/logs/bo16_%j.log
#SBATCH --error=./writing/logs/bo16_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Activate your conda environment if needed
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sal

# Run the generation script
python chat-scripts/generate-only.py --config chat-recipes/bo16.yaml
