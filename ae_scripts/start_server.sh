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

srun -s --unbuffered -p llm_s -w $machines --gres=gpu:0 -n1 -c 128 bash ./run_server.sh $tp $num_groups $num_models $master_ip $model $exec_type $migration_type $policy