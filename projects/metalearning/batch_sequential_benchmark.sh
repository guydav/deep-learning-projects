#!/usr/bin/env bash

if [[ $# < 3 || $# > 4 ]];  then
    echo "Usage: ./batch_sequential_benchmark.sh <GPU ID> <Number of repetitions> <Dataset Random seed> [optional train coreset size]"
    exit 1
fi

gpu_id=$1
num_reps=$2
initial_dataset_random_seed=$3

if [[ $# == 4 ]]; then
    coreset_size=$4
else
    coreset_size=22500
fi

export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=${gpu_id}

wandb login 9676e3cc95066e4865586082971f2653245f09b4

for i in `seq 1 ${num_reps}`; do
    # TODO: how do I specify boolean args? Integer?
    let "current_random_seed = ${initial_dataset_random_seed} + ${i}"
    python run_sequential_benchmark.py --train_coreset_size ${coreset_size} --dataset_random_seed ${current_random_seed}
    sleep 1s;
done




