#!/bin/bash

host=$1
migration_type=$2

# create fig11a.csv if not exists

SRC=$(pwd)
output=${SRC}/fig11b.csv

if [ ! -f $output ]; then
    echo "migration_type,ratio,avg_lat" > $output
fi

cd ..

python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 8 --num-models 32 --num-prompts 1000 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --ratio 1 --max_len 4096 --host $host --policy credit --output $output --migration_type $migration_type --output_style 1

python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 8 --num-models 32 --num-prompts 1000 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --ratio 2 --max_len 4096 --host $host --policy credit --output $output --migration_type $migration_type --output_style 1

python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 8 --num-models 32 --num-prompts 1000 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --ratio 4 --max_len 4096 --host $host --policy credit --output $output --migration_type $migration_type --output_style 1

python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 8 --num-models 32 --num-prompts 1000 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --ratio 6 --max_len 4096 --host $host --policy credit --output $output --migration_type $migration_type --output_style 1

python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 8 --num-models 32 --num-prompts 1000 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --ratio 8 --max_len 4096 --host $host --policy credit --output $output --migration_type $migration_type --output_style 1

