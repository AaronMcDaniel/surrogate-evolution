#!/bin/bash
output_dir=""
conf_file=""
num_generations=1
clear_dir=""
conda_environment="pco"

while getopts ":o:c:n:e:r" opt; do
  case $opt in
    o) output_dir="$OPTARG"
    ;;
    c) conf_file="$OPTARG"
    ;;
    n) num_generations="$OPTARG"
    ;;
    e) conda_environment="$OPTARG"
    ;;
    r) clear_dir="-r"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
        exit 1
    ;;
  esac
done

echo "Resubmitting main job to resume evolution..."
sbatch main.job -o "$output_dir" -c "$conf_file" -n "$num_generations" -e
