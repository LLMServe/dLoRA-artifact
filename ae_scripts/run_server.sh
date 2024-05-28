#! /bin/bash

# usage: srun -s --unbuffered -p llm_s -w $ips --gres=gpu:0 -n1 -c 128 bash ./eval_scripts/run.sh 1 1 8 $master_ip 7 3 3

tp=$1
num_groups=$2
num_models=$3
master_ip=$4
model=$5
exec_type=$6
migration_type=$7
policy=$8

num_gpus=$((tp * num_groups))

ips=$(ip a)
is_master=false

if [[ $ips =~ $master_ip/24 ]]; then # 
    echo "is master"
    ray start --head
else
    echo "not master"
    sleep 2
    ray start --address="$master_ip":6379
fi

# wait for ray to start
sleep 5

ray status

# python pmon.py --num_gpu $num_gpus &

# llama7b: NousResearch/Llama-2-7b-hf
# llama13b: NousResearch/Llama-2-13b-hf
# llama70b: NousResearch/Llama-2-70b-hf
if [[ $ips =~ $master_ip ]]; then 

    mkdir -p logs

    ts=$(date +%s)
    RAY_DEDUP_LOGS=0 TOKENIZERS_PARALLELISM=true python -m vllm.entrypoints.api_server --model NousResearch/Llama-2-${model}b-hf --tokenizer hf-internal-testing/llama-tokenizer  --swap-space 16 --disable-log-requests --num-models $num_models --num-groups $num_groups -tp $tp --use-dummy-weights  --worker-use-ray --engine-use-ray --exec-type $exec_type --migration-type $migration_type --host $master_ip --policy $policy 2>&1 | tee logs/output_${ts}.log
else
    sleep 10000000
fi
