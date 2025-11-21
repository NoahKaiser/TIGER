#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=Train-TIGER
#SBATCH --output=job_name%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=7-00:00:00
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --qos=batch

# Activate everything you need
module load cuda/11.2
#pyenv activate venv
# Run your python code
#export WANDB_API_KEY=
uv run --extra=cu118 audio_train.py --conf_dir configs/tiger.yml
#uv run --extra=cu118 DataPreProcess/process_echoset.py --in_dir /data/public/EchoSet --out_dir out
