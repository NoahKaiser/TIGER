#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=Simple_Self_Attention
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
#verwende cu128 wegen dr pytorch module
#module load cuda/12.8
#module load cudnn/9
#module list
#uv run --extra=cu118 audio_test_noah.py --conf_dir /misc/usrhomes/s1495/TIGER/Experiments/checkpoint/TIGER-small-on-EchoSet/conf.yml
#-> verwende cu118 wegen der GPU GTX 1080 Ti

#uv run visualize_reports_matplotlib.py --metrics_csv \
                                        #/usrhomes/s1495/TIGER/Experiments/checkpoint/TIGER-small-on-EchoSet/results/metrics.csv \
                                       #--metrics sdr_i sdr
#uv run --extra=cu118 stft_Pytorch_Test.py --in /data/public/EchoSet/train/1LXtFkjw3qL/1_6_kitchen/2817_5322/spk2_reverb.wav \
                                          #--out /no_backups/s1495/TIGER/iSTFT_Output/iSTFT_spk2_reverb_kausal.wav \
                                          #--n_fft 2048 --hop_length 512
uv run --extra=cu118 Simple_Self_Attention_Experiment.py