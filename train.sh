#!/bin/bash

cd ~/IFT6085/progressive_growing_of_gans

source ~/miniconda3/bin/activate
conda activate tensorflow1

git checkout master
git pull

cp -r --no-clobber datasets /Tmp/pichetre/ -v

python ./train.py --run-name TEST --blur-schedule $1 --train-k-images ${$2:1000}

