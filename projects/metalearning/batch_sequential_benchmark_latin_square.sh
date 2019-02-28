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
let "num_reps = $2 - 1"
run_name=$3
random_seed=$4

export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=${gpu_id}

wandb login 9676e3cc95066e4865586082971f2653245f09b4

for i in `seq 0 ${num_reps}`; do
    let "current_random_seed = ${random_seed} + ${i}"
    python run_sequential_benchmark.py --name ${run_name} --dataset_random_seed ${current_random_seed} --use_latin_square --latin_square_index ${current_random_seed} --latin_square_random_seed ${random_seed} ${@:5}
    sleep 1s;
done




