#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=Inference_Speech_with_pretrained_model
#SBATCH --output=job_name%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --qos=batch

# Activate everything you need
module load cuda/11.2
#pyenv activate venv
# Run your python code
export WANDB_API_KEY=3df4b50850a84840d28b55305ca54391ebaaf209

uv run audio_train.py --conf_dir configs/tiger.yml
