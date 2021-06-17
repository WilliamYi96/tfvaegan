#!/usr/bin/env bash
#SBATCH --job-name=SUN
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=4:00:00

module load anaconda3
source activate torch0.3

echo "SUN"
python -u /home/x_jhad/tfvaegan/zero-shot-images/image-scripts/run_sun_tfvaegan.py