#!/usr/bin/env bash

if [[ $# < 3 ]];  then
    echo "Usage: ./batch_sequential_benchmark.sh <GPU ID> <Number of repetitions> <Dataset Random seed> [additional arguments to the internal script]"
    exit 1
fi

gpu_id=$1
num_reps=$2
initial_dataset_random_seed=$3

export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=${gpu_id}

wandb login 9676e3cc95066e4865586082971f2653245f09b4

for i in `seq 1 ${num_reps}`; do
    # TODO: how do I specify boolean args? Integer?
    let "current_random_seed = ${initial_dataset_random_seed} + ${i}"
    python run_sequential_benchmark.py --dataset_random_seed ${current_random_seed} ${@:4}
    sleep 1s;
done




