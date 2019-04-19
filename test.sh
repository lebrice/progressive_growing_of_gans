#!/bin/bash
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=16G
#SBATCH --time=1-00:00

source ~/miniconda3/bin/activate
conda activate tensorflow1

cd ~/IFT6085/progressive_growing_of_gans
git checkout master
git pull

cp -r --no-clobber datasets /Tmp/pichetre/ -v

python ./train.py --run-name TEST --blur-schedule LINEAR --train-k-images 10

