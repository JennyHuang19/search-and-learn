#!/bin/bash
#
#SBATCH --job-name=gen_bo4
#SBATCH --account=lingo
#SBATCH --partition=vision-shared-h100
#SBATCH --qos=lingo-main
#SBATCH --time=12:00:00
#SBATCH --output=./writing/logs/bo4_%j.log
#SBATCH --error=./writing/logs/bo4_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Activate your conda environment if needed
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sal

# Run the generation script
python chat-scripts/generate-only.py --config chat-recipes/bo4.yaml
