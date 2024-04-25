#!/bin/bash

host=$1
policy=$2

# create fig10.csv if not exists

SRC=$(pwd)
output=${SRC}/fig10.csv

if [ ! -f $output ]; then
    echo "policy,stride,avg_lat,p90_lat" > $output
fi

cd ..

python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 4 --num-models 8 --num-prompts 400 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $1 --policy $2 --output $output --need_sort --map_stride 4

python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 4 --num-models 8 --num-prompts 400 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $1 --policy $2 --output $output --need_sort --map_stride 8

python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 4 --num-models 8 --num-prompts 400 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $1 --policy $2 --output $output --need_sort --map_stride 16

python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 4 --num-models 8 --num-prompts 400 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $1 --policy $2 --output $output --need_sort --map_stride 32