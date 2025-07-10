#!/bin/bash -l

#SBATCH --error=logs/hoomd_%j.err
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@example.com

# Activate conda and load HOOMD
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flowermd-dev  # Replace with your environment name

# Run the Python script
python ej_flakes.py


