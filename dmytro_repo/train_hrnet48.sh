#!/bin/bash
#SBATCH --job-name=dmytro_model_training_hrnet48.job
#SBATCH --output=/storage/ice-shared/vip-vvk/data/AOT/psomu3/dmytro_logs/dmytro_model_training_hrnet48.%A.%a.out
#SBATCH --error=/storage/ice-shared/vip-vvk/data/AOT/psomu3/dmytro_logs/dmytro_model_training_hrnet48.%A.%a.err
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH -G 1
#SBATCH -C "A100-40GB|H100|H200"

module load anaconda3/2023.03
cd dmytro_repo/seg_tracker
nvidia-smi
#four scripts for each model
#python train.py train experiments/120_gernet_m_b2_all.yaml
#python train.py train experiments/120_hrnet32_all.yaml
#python train.py train experiments/120_dla60_256_sgd_all_rerun.yaml
conda run -n dmytro7 --no-capture-output python train.py train experiments/130_hrnet48_all.yaml

#maybe?
#python train.py export_model experiments/120_gernet_m_b2_all.yaml --epoch 2220
#python train.py export_model experiments/120_hrnet32_all.yaml --epoch 2220
#python train.py export_model experiments/120_dla60_256_sgd_all_rerun.yaml --epoch 2220
#python train.py export_model experiments/130_hrnet48_all.yaml --epoch 50