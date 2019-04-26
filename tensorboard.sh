#!/bin/bash
#SBATCH --ntasks-per-node=4
#SBATCH --mem=8G
#SBATCH --time=7-00:00

cd ~/IFT6085/progressive_growing_of_gans

source ~/miniconda3/bin/activate
conda activate tensorflow1

git checkout master
git pull

tensorboard --logdir results --port 12354
