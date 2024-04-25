#!/bin/bash

host=$1
exec_type=$2

SRC=$(pwd)
output=${SRC}/fig9_maf1_13b.csv

if [ ! -f $output ]; then
    echo "exec_type,rate,avg_lat" > $output
fi

cd ..

if [ $exec_type -eq 1 ]; then

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 0.25 --num-models 128 --num-prompts 25 --trace_name azure_v1 --trace_path ~/maf1/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 0.5 --num-models 128 --num-prompts 50 --trace_name azure_v1 --trace_path ~/maf1/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 1 --num-models 128 --num-prompts 100 --trace_name azure_v1 --trace_path ~/maf1/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 1.5 --num-models 128 --num-prompts 150 --trace_name azure_v1 --trace_path ~/maf1/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 2 --num-models 128 --num-prompts 200 --trace_name azure_v1 --trace_path ~/maf1/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 3 --num-models 128 --num-prompts 300 --trace_name azure_v1 --trace_path ~/maf1/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 4 --num-models 128 --num-prompts 400 --trace_name azure_v1 --trace_path ~/maf1/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

elif [ $exec_type -eq 2 ]; then

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 0.5 --num-models 128 --num-prompts 50 --trace_name azure_v1 --trace_path ~/maf1/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 1 --num-models 128 --num-prompts 100 --trace_name azure_v1 --trace_path ~/maf1/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 2 --num-models 128 --num-prompts 200 --trace_name azure_v1 --trace_path ~/maf1/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 4 --num-models 128 --num-prompts 400 --trace_name azure_v1 --trace_path ~/maf1/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 6 --num-models 128 --num-prompts 600 --trace_name azure_v1 --trace_path ~/maf1/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 8 --num-models 128 --num-prompts 800 --trace_name azure_v1 --trace_path ~/maf1/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

else

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 8 --num-models 128 --num-prompts 800 --trace_name azure_v1 --trace_path ~/maf1/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 16 --num-models 128 --num-prompts 1600 --trace_name azure_v1 --trace_path ~/maf1/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 24 --num-models 128 --num-prompts 2400 --trace_name azure_v1 --trace_path ~/maf1/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 32 --num-models 128 --num-prompts 3200 --trace_name azure_v1 --trace_path ~/maf1/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 40 --num-models 128 --num-prompts 4000 --trace_name azure_v1 --trace_path ~/maf1/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 48 --num-models 128 --num-prompts 4800 --trace_name azure_v1 --trace_path ~/maf1/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 56 --num-models 128 --num-prompts 5600 --trace_name azure_v1 --trace_path ~/maf1/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 64 --num-models 128 --num-prompts 6400 --trace_name azure_v1 --trace_path ~/maf1/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 72 --num-models 128 --num-prompts 7200 --trace_name azure_v1 --trace_path ~/maf1/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 80 --num-models 128 --num-prompts 8000 --trace_name azure_v1 --trace_path ~/maf1/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

fi