#!/bin/bash
#SBATCH --job-name=codestral
#SBATCH -G 1
#SBATCH -c 16
#SBATCH --mem=80g
#SBATCH --time=16:00:00
#SBATCH --output=/storage/ice-shared/vip-vvk/data/AOT/%u/codestral/codestral.%A.%a.log
#SBATCH --error=/storage/ice-shared/vip-vvk/data/AOT/%u/codestral/codestral_error.%A.%a.log
#SBATCH --constraint="H200|H100"

# Set CUDA debugging environment variables
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# # Set PyTorch memory management
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Load modules
module load anaconda3/2023.03
module load cuda/12.6.1

nvidia-smi

# Run the Codestral dataset creation with unbuffered output
conda run -n nas --no-capture-output python -u -m surrogates.codestral --mode dataset