#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=100G
#SBATCH --gpus=1
#SBATCH --job-name=ffhq_Dynamic-SS
#SBATCH --partition=gpu-v100
#SBATCH --time=7-00:00:00
#SBATCH --mail-user='you_email_id'
#SBATCH --mail-type=ALL
#SBATCH --export=NONE
#SBATCH -D /your_path


# Add a container to the environment:
#
vpkg_devrequire anaconda/2024.02:python3
conda activate you_renvironment

#
# Execute our Python script:

python main_run.py --config /configs/config_ffhq_embeddings_adult_young.py  --workdir /checkpoints/ --c 1.0
python main_run.py --config /configs/config_ffhq_embeddings_adult_young.py  --workdir /checkpoints/ --c 2.0
python main_run.py --config /configs/config_ffhq_embeddings_adult_young.py  --workdir /checkpoints/ --c 3.0
python main_run.py --config /configs/config_ffhq_embeddings_adult_young.py  --workdir /checkpoints/ --c 4.0
