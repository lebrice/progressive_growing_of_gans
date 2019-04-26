#!/bin/bash
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --time=2-00:00
#SBATCH --exclude=kepler2,kepler3

if [[ -z "$BLUR_SCHEDULE" ]]; then
	echo "Please choose a schedule type."
	exit
fi

cd ~/IFT6085/progressive_growing_of_gans

source ~/miniconda3/bin/activate
conda activate tensorflow1

git checkout master
git pull

cp -r --no-clobber datasets /Tmp/pichetre/ -v

python ./train.py --blur-schedule $BLUR_SCHEDULE --train-k-images 1000
