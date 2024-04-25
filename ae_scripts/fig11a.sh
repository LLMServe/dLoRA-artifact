#!/bin/bash

host=$1
migration_type=$2

# create fig11a.csv if not exists

SRC=$(pwd)
output=${SRC}/fig11a.csv

if [ ! -f $output ]; then
    echo "migration_type,rate,queueing_delay,adjust_time,solver_time,tot_time" > $output
fi

cd ..

if [ $migration_type -eq 1 ]; then

    # python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 2 --num-models 32 --num-prompts 300 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $host --policy credit --output $output --need_sort --migration_type $migration_type --output_style 3

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 3 --num-models 32 --num-prompts 400 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $host --policy credit --output $output --need_sort --migration_type $migration_type --output_style 3

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 4 --num-models 32 --num-prompts 500 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $host --policy credit --output $output --need_sort --migration_type $migration_type --output_style 3

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 5 --num-models 32 --num-prompts 600 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $host --policy credit --output $output --need_sort --migration_type $migration_type --output_style 3

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 6 --num-models 32 --num-prompts 700 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $host --policy credit --output $output --need_sort --migration_type $migration_type --output_style 3

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 7 --num-models 32 --num-prompts 800 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $host --policy credit --output $output --need_sort --migration_type $migration_type --output_style 3

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 8 --num-models 32 --num-prompts 1000 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $host --policy credit --output $output --need_sort --migration_type $migration_type --output_style 3

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 12 --num-models 32 --num-prompts 1600 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $host --policy credit --output $output --need_sort --migration_type $migration_type --output_style 3

elif [ $migration_type -eq 2 ]; then

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 6 --num-models 32 --num-prompts 700 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $host --policy credit --output $output --need_sort --migration_type $migration_type --output_style 3

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 8 --num-models 32 --num-prompts 1000 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $host --policy credit --output $output --need_sort --migration_type $migration_type --output_style 3

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 12 --num-models 32 --num-prompts 1600 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $host --policy credit --output $output --need_sort --migration_type $migration_type --output_style 3

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 16 --num-models 32 --num-prompts 2100 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $host --policy credit --output $output --need_sort --migration_type $migration_type --output_style 3

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 20 --num-models 32 --num-prompts 2700 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $host --policy credit --output $output --need_sort --migration_type $migration_type --output_style 3

else

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 6 --num-models 32 --num-prompts 700 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $host --policy credit --output $output --need_sort --migration_type $migration_type --output_style 3

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 8 --num-models 32 --num-prompts 1000 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $host --policy credit --output $output --need_sort --migration_type $migration_type --output_style 3

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 12 --num-models 32 --num-prompts 1600 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $host --policy credit --output $output --need_sort --migration_type $migration_type --output_style 3

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 16 --num-models 32 --num-prompts 2100 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $host --policy credit --output $output --need_sort --migration_type $migration_type --output_style 3

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 20 --num-models 32 --num-prompts 2700 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $host --policy credit --output $output --need_sort --migration_type $migration_type --output_style 3

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 24 --num-models 32 --num-prompts 3200 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $host --policy credit --output $output --need_sort --migration_type $migration_type --output_style 3

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 28 --num-models 32 --num-prompts 3800 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $host --policy credit --output $output --need_sort --migration_type $migration_type --output_style 3

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 32 --num-models 32 --num-prompts 4300 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $host --policy credit --output $output --need_sort --migration_type $migration_type --output_style 3

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 36 --num-models 32 --num-prompts 4800 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.6.0 --interval 60 --host $host --policy credit --output $output --need_sort --migration_type $migration_type --output_style 3
    

fi
