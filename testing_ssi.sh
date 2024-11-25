#!/bin/bash
#SBATCH --job-name=ttest_ssi
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=4               
#SBATCH --time=4:00:00   
#SBATCH --mem=80G                
#SBATCH --output=/storage/ice-shared/vip-vvk/data/AOT/ttest_ssi/output-%j.out  
#SBATCH --error=/storage/ice-shared/vip-vvk/data/AOT/ttest_ssi/error-%j.log            
#SBATCH --mail-type=BEGIN,END,FAIL         
#SBATCH --mail-user=tthakur9@gatech.edu    
#SBATCH --constraint=V100-32GB                
#SBATCH --gres=gpu:V100:2    

cd /home/hice1/tthakur9/scratch/surrogate-evolution
module load anaconda3/2023.03
module load cuda/11.8.0
conda activate pco
python testing_ssi.py