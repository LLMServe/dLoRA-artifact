#!/bin/bash

host=$1
exec_type=$2

SRC=$(pwd)
output=${SRC}/fig9_maf2_70b.csv

if [ ! -f $output ]; then
    echo "exec_type,rate,avg_lat" > $output
fi

cd ..

if [ $exec_type -eq 1 ]; then

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 0.1 --num-models 32 --num-prompts 10 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 0.2 --num-models 32 --num-prompts 20 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 0.3 --num-models 32 --num-prompts 30 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 0.4 --num-models 32 --num-prompts 40 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 0.6 --num-models 32 --num-prompts 60 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 0.8 --num-models 32 --num-prompts 80 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

elif [ $exec_type -eq 2 ]; then

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 0.1 --num-models 32 --num-prompts 10 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 0.2 --num-models 32 --num-prompts 20 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 0.3 --num-models 32 --num-prompts 30 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 0.4 --num-models 32 --num-prompts 40 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 0.6 --num-models 32 --num-prompts 60 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 0.8 --num-models 32 --num-prompts 80 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

else

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 2 --num-models 32 --num-prompts 200 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 4 --num-models 32 --num-prompts 400 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 6 --num-models 32 --num-prompts 600 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 8 --num-models 32 --num-prompts 800 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 10 --num-models 32 --num-prompts 1000 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 12 --num-models 32 --num-prompts 1200 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

    python benchmarks/benchmark_serving.py --backend vllm --tokenizer hf-internal-testing/llama-tokenizer  --dataset ~/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 16 --num-models 32 --num-prompts 1600 --trace_name azure_v2 --trace_path ~/maf2/ --start_time 0.0.0 --end_time 0.2.0 --interval 60 --host $host --output $output --exec_type $exec_type --output_style 4

fi