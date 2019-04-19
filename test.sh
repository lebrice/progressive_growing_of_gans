#!/bin/bash
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --time=1-00:00
cd ~/IFT6085/progressive_growing_of_gans
. ./train.sh
python ./train.py --run-name TEST --blur-schedule LINEAR --train-k-images 10
