#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=tse_tigerFiLM2_on_Subset_TSE_ECHI
#SBATCH --output=slurm_logs/%x_%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=7-00:00:00
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --nodelist=linse20



#uv run --extra=cu118 DataPreProcess/process_echoset.py --in_dir /data/public/EchoSet --out_dir out
#uv run --extra=cu118 audio_train_tse.py --conf_dir configs/tiger_tse_selfcross.yml
#uv run --extra=cu118 audio_train_tse.py --conf_dir configs/tiger_tse_film2.yml
uv run --extra=cu118 audio_train_on_Subset.py --conf_dir configs/tse_tigerFiLM2_on_Subset_TSE_ECHI.yml
#uv run --extra=cu118 audio_train.py --conf_dir configs/tiger_on_ECHI.yml

#uv run --extra=cpu DataPreProcess/preprocess_tse_echi.py --echi_root /misc/data/public/CHiME9 \
                                                             # --output_root /no_backups/s1495 \
                                                             # --device ha
#uv run --extra=cpu DataPreProcess/verify_tse_spk_id_alignment.py
#uv run --extra=cu118 DataPreProcess/patch_missing_spk_embeddings_from_targets.py --inplace \
 # --device cuda \
  #--max_refs_per_speaker 1 \
  #--max_chunk_batch 32
#uv run --extra=cu128 audio_train.py --conf_dir configs/causal_tiger.yml
#uv run --extra=cu128 audio_train.py --conf_dir configs/causal_tiger2.yml
#uv run --extra=cu118 audio_test_EchoSet.py --conf_dir /misc/usrhomes/s1495/TIGER/Experiments/checkpoint/CausalTIGER-on-EchoSet/conf.yml --save_dir /no_backups/s1495/TIGER/CausalTIGER_on_EchoSet
#-> verwende cu118 wegen der GPU GTX 1080 Ti

#uv run visualize_reports_matplotlib.py --metrics_csv \
                                        #/usrhomes/s1495/TIGER/Experiments/checkpoint/CausalTIGER-on-EchoSet/results/metrics.csv \
                                      # --metrics sdr_i sdr
#uv run --extra=cu118 stft_Pytorch_Test.py --in /data/public/EchoSet/train/1LXtFkjw3qL/1_6_kitchen/2817_5322/spk2_reverb.wav \
                                          #--out /no_backups/s1495/TIGER/iSTFT_Output/iSTFT_spk2_reverb_kausal.wav \
                                          #--n_fft 2048 --hop_length 512




# Vergleiche zwei .wav files, auf 16 kHz resampeln und nur die ersten 20 s vergleichen:
#uv run --extra=cpu compare_two_wav.py /data/public/CHiME9/ref/dev/dev_02.ha.pos1.wav /no_backups/s1495/PreProcessed_ECHI/Processed_ECHI/ha/dev/dev_02.ha.16kHz.mono.wav 16000 10


#uv run --extra=cu118 Listen_to_ECHIDataset.py --json_dir /no_backups/s1495/Processed_ECHI/ha/train \
    #--out_dir  /no_backups/s1495/Listen_to_ECHIDataset/batch01 \
    #--n 100 --n_src 4 --sr 16000 --segment 3.0 --start_idx 0
#uv run --extra=cu118 Listen_to_TSE_ECHIDataset.py \
 # --json_dir /no_backups/s1495/Processed_TSE_ECHI/ha/train \
 # --spk_emb_path /no_backups/s1495/ECHI_spk_embeddings/ECAPA_embeddings/ecapa_embeddings.pt \
  #--out_dir /no_backups/s1495/Listen_to_TSE_ECHIDataset/tse_batch01 \
  #--n 100 --segment 3.0 --start_idx 0 \
  #--only_valid_speech_region \
  #--valid_speech_metadata_root /data/public/CHiME9/metadata/ref

#uv run --extra=cpu inspect_ECAPA_Embeddings.py --emb_pt /no_backups/s1495/ECHI_spk_embeddings/ECAPA_embeddings/ecapa_embeddings.pt --meta_json /no_backups/s1495/ECHI_spk_embeddings/ECAPA_embeddings/ecapa_embeddings_meta.json