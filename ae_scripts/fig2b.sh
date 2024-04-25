#!/bin/bash

host=$1
exec_type=$2

# create fig2b.csv if not exists

SRC=$(pwd)
output=${SRC}/fig2b.csv

if [ ! -f $output ]; then
    echo "replica_id,queueing_ratio" > $output
fi

cd ..

python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 3 --num-models 32 --num-prompts 600 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $host --output $output --exec_type 1 --output_style 5
