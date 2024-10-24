#!/bin/bash
#SBATCH --job-name=test_surr_evo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=18:00:00
#SBATCH --output="/storage/ice-shared/vip-vvk/data/AOT/evolution_logs/evolution.%A.%a.log"
#SBATCH --error="/storage/ice-shared/vip-vvk/data/AOT/evolution_logs/evolution_error.%A.%a.log"

output_dir=""
conf_file=""
num_generations=1
force_wipe=""
clear_dir=""
conda_environment="pco"

while getopts ":o:c:n:e:fr" opt; do
  case $opt in
    o) output_dir="$OPTARG"
    ;;
    c) conf_file="$OPTARG"
    ;;
    n) num_generations="$OPTARG"
    ;;
    e) conda_environment="$OPTARG"
    ;;
    f) force_wipe="-f"
    ;;
    r) clear_dir="-r"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
        exit 1
    ;;
  esac
done

module load anaconda3/2023.03
module load cuda/12.1.1

START_TIME=$(date +%s)
TIME_LIMIT=$((18 * 60 * 60))  # 18 hours in seconds
THRESHOLD=$((17 * 60 * 60))   # 17 hours, gives 1-hour buffer

function check_time_remaining() {
    CURRENT_TIME=$(date +%s)
    ELAPSED_TIME=$((CURRENT_TIME - START_TIME))
    REMAINING_TIME=$((TIME_LIMIT - ELAPSED_TIME))
    
    # If remaining time is less than the threshold, exit after the current generation
    if [ "$REMAINING_TIME" -le "$THRESHOLD" ]; then
        echo "Approaching time limit. Launching spin-off script to resume evolution."
        ./resume_evolution.sh -o "$output_dir" -c "$conf_file" -n "$num_generations" -e "$conda_environment" $clear_dir
        exit 0
    fi
}

while :; do
    conda run -n ${conda_environment} --no-capture-output python -u main.py -o "$output_dir" -conf "$conf_file" -n "$num_generations" $force_wipe $clear_dir
    check_time_remaining
done
