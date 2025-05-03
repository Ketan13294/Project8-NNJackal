#!/bin/bash

#SBATCH --job-name=Jackal_trajectory_tracking_grid_v2
#SBATCH --output=training_grid_v2.txt # Output file
#SBATCH --time=48:00:00 # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --partition=l40s_normal_q
#SBATCH -A ece6474drlspring2025
#SBATCH -vv

source ~/.bashrc
export PATH="$HOME/anaconda3/bin:$PATH"
source activate
conda activate deeprl
python train_grid_weights_v2.py
