#!/bin/bash

# Array of speeds to test
SPEEDS=(10)
# 40 60 100)

# Array of channel models
MODELS=(A D C)

# Function to generate matrices for a given GPU
generate_on_gpu() {
    gpu_id=$1
    channel_type=$2
    model=$3
    
    for speed in "${SPEEDS[@]}"; do
        echo "GPU ${gpu_id}: ${channel_type}-${model} at ${speed} m/s"
        python save_R_matrix_TD.py --gpu $gpu_id --channel_type $channel_type --channel_model $model --speed $speed
    done
}

# Run TDL on GPU 0 and GPU 1 in parallel
# generate_on_gpu 0 TDL A 
generate_on_gpu 0 TDL D &

# # Run CDL on GPU 2 and GPU 3 in parallel
generate_on_gpu 1 CDL C
# generate_on_gpu 3 CDL D &

# Wait for all background jobs to complete
wait

echo "All R matrices generated!"