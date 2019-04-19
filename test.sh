#!/bin/bash
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --time=1-00:00
cd ~/IFT6085/progressive_growing_of_gans
source ./train.sh 
train LINEAR 10
