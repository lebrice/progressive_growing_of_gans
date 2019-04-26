#!/bin/bash
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=5-00:00
#SBATCH --exclude=kepler2,kepler3

# passed in using --export=RUN_ID=x now.
##RUN_ID=$1
HOST=`hostname`
MILA=true
if [[ "$HOST" == "Brigitte" ]]; then
    MILA=false
fi

echo "Running on MILA servers? $MILA"
if [[ -z "$RUN_ID" ]]; then
    echo "Please input a run id to use."
    exit
elif [[ -n "$RUN_ID" ]]; then
    echo "Running metrics for run_id $RUN_ID"
fi
# Change to the directory containing this script
cd "$(dirname "$0")"
echo `pwd`
if $MILA; then
    source ~/miniconda3/bin/activate
    cd ~/IFT6085/progressive_growing_of_gans/
    cp -r --no-clobber datasets /Tmp/pichetre/ -v
fi

conda activate tensorflow1

git checkout master
git pull


python run_metrics.py --run-id $RUN_ID --metric "swd-16k"
python run_metrics.py --run-id $RUN_ID --metric "fid-10k"
python run_metrics.py --run-id $RUN_ID --metric "fid-50k"
python run_metrics.py --run-id $RUN_ID --metric "is-50k"
python run_metrics.py --run-id $RUN_ID --metric "msssim-20k"
