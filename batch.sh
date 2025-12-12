#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=Test-TIGER
#SBATCH --output=job_name%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=7-00:00:00
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --nodelist=linse20

# Activate everything you need
#module load cuda/11.2
#pyenv activate venv
# Run your python code

#uv run --extra=cu118 DataPreProcess/process_echoset.py --in_dir /data/public/EchoSet --out_dir out
#uv run --extra=cu128 audio_train.py --conf_dir configs/tiger.yml

uv run --extra=cu128 audio_test.py --conf_dir /misc/usrhomes/s1495/TIGER/Experiments/checkpoint/TIGER-small-on-EchoSet/conf.yml