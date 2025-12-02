#!/usr/bin/env bash
set -euo pipefail

# Script to run baseline_runs.py for multiple channels and speeds.
# Parallelizes jobs in batches on GPUs 0 and 1.

ROOT_DIR="$(pwd)"
LOGDIR="${ROOT_DIR}/runs_logs"
mkdir -p "${LOGDIR}"

channels=("TDL-D" "TDL-A" "CDL-C")
speeds=(10 40 60 100)

# GPUs available (logical device ids)
gpus=(0 1)
num_gpus=${#gpus[@]}

# Build job list (channel,speed pairs)
jobs=()
for ch in "${channels[@]}"; do
  for s in "${speeds[@]}"; do
    jobs+=("${ch} ${s}")
  done
done

# Run jobs in batches of size num_gpus, assigning GPUs round-robin per batch
total=${#jobs[@]}
idx=0
while [ $idx -lt $total ]; do
  echo "Starting batch at job index $idx"
  # Launch up to num_gpus jobs
  for slot in $(seq 0 $((num_gpus - 1))); do
    job_index=$((idx + slot))
    if [ $job_index -ge $total ]; then
      break
    fi
    pair="${jobs[$job_index]}"
    ch="$(echo "$pair" | awk '{print $1}')"
    s="$(echo "$pair" | awk '{print $2}')"
    
    # Parse channel type and model from designation (e.g., "TDL-D" -> type="TDL", model="D")
    ch_type=$(echo "$ch" | cut -d'-' -f1)
    ch_model=$(echo "$ch" | cut -d'-' -f2)
    if [ -z "$ch_model" ]; then
      ch_model="A"
    fi
    
    gpu=${gpus[$slot]}
    stamp=$(date +"%Y%m%d-%H%M%S")
    logfile="${LOGDIR}/run_${ch// /_}_speed${s}_gpu${gpu}_${stamp}.log"
    echo "Launching: channel=${ch} (type=${ch_type}, model=${ch_model}), speed=${s} on GPU ${gpu} -> ${logfile}"
    CUDA_VISIBLE_DEVICES=${gpu} python3 baseline_runs.py --channel "${ch}" --channel-type "${ch_type}" --channel-model "${ch_model}" --speed "${s}" > "${logfile}" 2>&1 &
    sleep 1
  done

  # Wait for this batch to finish before starting next batch
  wait
  idx=$((idx + num_gpus))
done

echo "All runs submitted. Logs in ${LOGDIR}." 
