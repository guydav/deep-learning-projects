#!/usr/bin/env bash

#echo $#
#echo $@
#let "test = $# <= 3";
#echo $test
#
#if [[ $# <= 3 ]];  then
#    echo "Usage: ./batch_control_sequential_benchmark.sh <GPU ID> <start index> <end index> <run id> <Dataset Random seed> [additional arguments to the internal script]"
#    exit 1
#fi

gpu_id=$1
start_index=$2
let "end_index = $3 - 1"
run_name=$4

export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=${gpu_id}

wandb login 9676e3cc95066e4865586082971f2653245f09b4

for i in `seq ${start_index} ${end_index}`; do
    python run_forgetting_experiment.py --name ${run_name} --run_id_line_number ${i} ${@:5}
    sleep 1s;
done




