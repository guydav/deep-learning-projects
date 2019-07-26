#!/usr/bin/env bash

#echo $#
#echo $@
#let "test = $# <= 3";
#echo $test
#
#if [[ $# <= 3 ]];  then
#    echo "Usage: ./batch_sequential_benchmark.sh <GPU ID> <Number of repetitions> <run id> <Dataset Random seed> [additional arguments to the internal script]"
#    exit 1
#fi

gpu_id=$1
num_reps=$2
run_name=$3
initial_dataset_random_seed=$4

export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=${gpu_id}

wandb login 9676e3cc95066e4865586082971f2653245f09b4

for i in `seq 1 ${num_reps}`; do
    let "current_random_seed = ${initial_dataset_random_seed} + ${i}"
    python run_simultaneous_training.py --name ${run_name} --dataset_random_seed ${current_random_seed} ${@:5}
    sleep 1s;
done




