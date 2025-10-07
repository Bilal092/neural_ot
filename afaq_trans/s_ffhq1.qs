#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=100G
#SBATCH --gpus=1
#SBATCH --job-name=ffhq_Dynamic-SS
#SBATCH --partition=gpu-v100
#SBATCH --time=7-00:00:00
#SBATCH --mail-user='your_email'
#SBATCH --mail-type=ALL
#SBATCH --export=NONE
#SBATCH -D /your_path

#UD_QUIET_JOB_SETUP=YES
#export UD_JOB_EXIT_FN_SIGNALS="SIGTERM EXIT"

# Add a container to the environment:
#
vpkg_devrequire anaconda/2024.02:python3
conda activate environment

#
# Execute our Python script:
python ./static_ffhq_run.py --config ./configs/static_config_ffhq1.py  --workdir ./checkpoints --c 1.0 
python ./static_ffhq_run.py --config ./configs/static_config_ffhq1.py  --workdir ./checkpoints --c 2.0 
python ./static_ffhq_run.py --config ./configs/static_config_ffhq1.py  --workdir ./checkpoints --c 3.0 
python ./static_ffhq_run.py --config ./configs/static_config_ffhq1.py  --workdir ./checkpoints --c 4.0 