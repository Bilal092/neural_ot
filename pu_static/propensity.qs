#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=100G
#SBATCH --gpus=1
#SBATCH --job-name=Dynamic-SS
#SBATCH --partition=gpu-v100
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --export=NONE



# Add a container to the environment:
vpkg_devrequire anaconda/2024.02:python3
conda activate your_environment
# Ensure dataset is provided
dataset=$1

# Execute the Python script
# ml_collections based argparsing is used
python ./main_propensity.py \
  --config=./config_propensity.py:${dataset} \
  --workdir ./checkpoints/ \
  --mode 'train'
