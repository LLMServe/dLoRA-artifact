#!/bin/bash

machines=$1
tp=$2
num_groups=$3
num_models=$4
master_ip=$5
model=$6
exec_type=$7
migration_type=$8

if [ $# -eq 9 ]; then
    policy=$9
else
    policy="credit"
fi

if [ $model -eq 70 ]; then
    num_node=4
elif [ $num_groups -eq 32 ]; then
    num_node=4
else 
    num_node=1
fi

srun -s --unbuffered -p llm_s -w $machines --gres=gpu:0 -n$num_node -c 128 bash ./run_server.sh $tp $num_groups $num_models $master_ip $model $exec_type $migration_type $policy