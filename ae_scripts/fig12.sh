#!/bin/bash

host=$1
exec_type=$2
num_models=$3

# create fig12.csv if not exists

SRC=$(pwd)
output=${SRC}/fig12.csv

if [ ! -f $output ]; then
    echo "exec_type,num_models,throughput" > $output
fi

cd ..

python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 24 --num-models $num_models --num-prompts 3200 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 2