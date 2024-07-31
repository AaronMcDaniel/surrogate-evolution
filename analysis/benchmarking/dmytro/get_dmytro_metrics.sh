#!/bin/bash
#SBATCH --job-name=dmytro_get_metrics.job
#SBATCH --output=logs/dmytro_get_metrics.%A.%a.out
#SBATCH --error=logs/dmytro_get_metrics.%A.%a.err
#SBATCH --time=20-00:00
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH -G 1
#SBATCH -x ice109,ice111,ice161,ice113,ice116,ice114,ice170,ice149,ice158,ice177,ice178,ice120
#SBATCH --array=1-11

module load anaconda3/2023.07
cd analysis/benchmarking/dmytro
nvidia-smi
#four scripts for each model
#python train.py train experiments/120_gernet_m_b2_all.yaml
#python train.py train experiments/120_hrnet32_all.yaml
#python dmytro_eval_script.py -g /home/eharpster3/precog-opt-grip/dmytro_metrics/complex -o /gv1/projects/GRIP_Precog_Opt/unseeded_baseline_evolution 
python dmytro_eval_script.py -g dmytro_metrics/epochs/$((SLURM_ARRAY_TASK_ID)) -o /gv1/projects/GRIP_Precog_Opt/unseeded_baseline_evolution --job_id $((SLURM_ARRAY_TASK_ID))
#--job_id $((SLURM_ARRAY_TASK_ID))
#python train.py train experiments/130_hrnet48_all.yaml

#maybe?
#python train.py export_model experiments/120_gernet_m_b2_all.yaml --epoch 2220
#python train.py export_model experiments/120_hrnet32_all.yaml --epoch 2220
#python train.py export_model experiments/120_dla60_256_sgd_all_rerun.yaml --epoch 50
#python train.py export_model experiments/130_hrnet48_all.yaml --epoch 2220