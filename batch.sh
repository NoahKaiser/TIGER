#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=CausalTIGER3_on_EchosSet
#SBATCH --output=%x_%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=7-00:00:00
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --nodelist=linse19



#uv run --extra=cu118 DataPreProcess/process_echoset.py --in_dir /data/public/EchoSet --out_dir out
#uv run --extra=cu118 audio_train.py --conf_dir configs/tiger_test.yml
#uv run --extra=cu118 audio_train.py --conf_dir configs/tiger_on_ECHI.yml

#uv run --extra=cu118 DataPreProcess/process_echi_old.py --in_dir /data/public/CHiME9 --out_dir out

#uv run --extra=cu128 audio_train.py --conf_dir configs/causal_tiger.yml
uv run --extra=cu128 audio_train.py --conf_dir configs/causal_tiger3.yml
#uv run --extra=cu118 audio_test_noah.py --conf_dir /misc/usrhomes/s1495/TIGER/Experiments/checkpoint/CausalTIGER-on-EchoSet/conf.yml --save_dir /no_backups/s1495/TIGER/CausalTIGER_on_EchoSet
#-> verwende cu118 wegen der GPU GTX 1080 Ti

#uv run visualize_reports_matplotlib.py --metrics_csv \
                                        #/usrhomes/s1495/TIGER/Experiments/checkpoint/CausalTIGER-on-EchoSet/results/metrics.csv \
                                      # --metrics sdr_i sdr
#uv run --extra=cu118 stft_Pytorch_Test.py --in /data/public/EchoSet/train/1LXtFkjw3qL/1_6_kitchen/2817_5322/spk2_reverb.wav \
                                          #--out /no_backups/s1495/TIGER/iSTFT_Output/iSTFT_spk2_reverb_kausal.wav \
                                          #--n_fft 2048 --hop_length 512




# Vergleiche zwei .wav files, auf 16 kHz resamplen und nur die ersten 20 s vergleichen:
#uv run --extra=cpu compare_two_wav.py /data/public/CHiME9/ref/dev/dev_02.ha.pos1.wav /no_backups/s1495/PreProcessed_ECHI/Processed_ECHI/ha/dev/dev_02.ha.16kHz.mono.wav 16000 10


#uv run --extra=cu118 Listen_to_ECHIDataset.py --json_dir /no_backups/s1495/Processed_ECHI/ha/train \
    #--out_dir  /no_backups/s1495/Listen_to_ECHIDataset/batch01 \
    #--n 100 --n_src 4 --sr 16000 --segment 3.0 --start_idx 0
