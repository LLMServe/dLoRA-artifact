#! /bin/bash

# usage: srun -s --unbuffered -p llm_s -w $master_ip --gres=gpu:0 -n1 -c 128 bash ./eval_scripts/run.sh

tp=1
num_groups=8
num_models=32
num_gpus=$((tp * $num_groups))

ips=$(ip a)
master_ip='10.140.60.153'
is_master=false

if [[ $ips =~ $master_ip ]]; then # 
    echo "is master"
    is_master=true
    ray start --head
else
    echo "not master"
    sleep 2
    ray start --address="$master_ip":6379
fi

ray status

python eval_scripts/pmon.py --num_gpu $num_gpus &

python eval_scripts/cpu.py --num_cpu 128 &

# llama7b: NousResearch/Llama-2-7b-hf
# llama13b: NousResearch/Llama-2-13b-hf
# llama70b: NousResearch/Llama-2-70b-hf
if [[ $ips =~ $master_ip ]]; then 
    ts=$(date +%s)
    RAY_DEDUP_LOGS=0 TOKENIZERS_PARALLELISM=true python -m vllm.entrypoints.api_server --model NousResearch/Llama-2-7b-hf --tokenizer hf-internal-testing/llama-tokenizer  --swap-space 16 --disable-log-requests --num-models $num_models --num-groups $num_groups -tp $tp --use-dummy-weights  --worker-use-ray --engine-use-ray --exec-type 3 --migration-type 3 --host $master_ip 2>&1 | tee logs/output_${ts}.log
else
    sleep 10000000
fi
