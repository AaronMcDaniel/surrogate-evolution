#!/bin/bash
#SBATCH --job-name=dmytro_model_training_hrnet32.job
#SBATCH --output=dmytro_repo/logs/dmytro_model_training_hrnet32.%A.%a.out
#SBATCH --error=dmytro_repo/logs/dmytro_model_training_hrnet32.%A.%a.err
#SBATCH --time=20-00:00
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH -G 1
#SBATCH -x ice109,ice111,ice161,ice113,ice116,ice114,ice170,ice149,ice158,ice177,ice178,ice120
##BATCH -C "NVIDIAA100-SXM4-80GB"

module load anaconda3/2023.07
cd dmytro_repo/seg_tracker
nvidia-smi
#four scripts for each model
#python train.py train experiments/120_gernet_m_b2_all.yaml
python train.py train experiments/120_hrnet32_all.yaml
#python train.py train experiments/120_dla60_256_sgd_all_rerun.yaml
#python train.py train experiments/130_hrnet48_all.yaml

#maybe?
#python train.py export_model experiments/120_gernet_m_b2_all.yaml --epoch 2220
#python train.py export_model experiments/120_hrnet32_all.yaml --epoch 50
#python train.py export_model experiments/120_dla60_256_sgd_all_rerun.yaml --epoch 2220
#python train.py export_model experiments/130_hrnet48_all.yaml --epoch 2220