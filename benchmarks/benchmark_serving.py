"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    (vLLM backend)
    python -m vllm.entrypoints.api_server \
        --model <your_model> --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_hf_server.sh <your_model>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --tokenizer <your_model> --dataset <target_dataset> \
        --request-rate <request_rate>
"""
import argparse
import asyncio
import json
import os
import random
import time
from typing import AsyncGenerator, List, Tuple

import aiohttp
import numpy as np
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.workload_generator import trace

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []
REQUEST_RECORD: List[Tuple[int, float, float, int, int]] = []
start_ts = [0]

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    force_output_len: int,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    if "ShareGPT" in dataset_path:
        # Filter out the conversations with less than 2 turns.
        dataset = [
            data for data in dataset
            if len(data["conversations"]) >= 2
        ]
        # Only keep the first two turns of each conversation.
        dataset = [
            (data["conversations"][0]["value"], data["conversations"][1]["value"])
            for data in dataset
        ]
    else:
        # Filter out the conversations with less than 2 turns.
        dataset = [
            data for data in dataset
            if len(data["instruction"]) + len(data["input"]) >= 2
        ]
        # Only keep the first two turns of each conversation.
        dataset = [
            (data["instruction"] + data["input"], data["output"])
            for data in dataset
        ]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
            continue
        if prompt_len > 1024 or prompt_len + output_len > args.max_len:
            # Prune too long sequences.
            continue
        if args.ratio != 1:
            tot_len = min((prompt_len + output_len) * args.ratio, args.max_len)
            prompt_len = int(prompt_len / (prompt_len + output_len) * tot_len)
            output_len = tot_len - prompt_len
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    if force_output_len > 0:
        ret_requests = []
        output_lens = [t[2] for t in sampled_requests]
        max_output_len = max(output_lens)
        divide = max_output_len // force_output_len
        print(max_output_len, divide)

        for prompt, prompt_len, output_len in sampled_requests:
            output_len = output_len // divide + 1
            ret_requests.append((prompt, prompt_len, output_len))
        
        sampled_requests = ret_requests

    prompt_lens = [t[1] for t in sampled_requests]
    output_lens = [t[2] for t in sampled_requests]
    max_output_len = max(output_lens)
    print(f"avg_prompt_len: {sum(prompt_lens) / len(prompt_lens)},",
          f"avg_output_len: {sum(output_lens) / len(output_lens)},",
          f"max_output_len: {max_output_len}")
    return sampled_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    num_models: int,
    workload: List[Tuple[int, float]],
) -> AsyncGenerator[Tuple[int, Tuple[str, int, int]], None]:
    input_requests = iter(input_requests)
    model_cnt = [0] * num_models
    for model_id, interval in workload:
        model_cnt[model_id] += 1
    print("request model cnt:", model_cnt)
    idx = 0
    for request in input_requests:
        model_id, interval = workload[idx]
        idx += 1
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)
        yield model_id, request

        # if request_rate == float("inf"):
        #     # If the request rate is infinity, then we don't need to wait.
        #     continue
        # Sample the request interval from the exponential distribution.
        # interval = np.random.exponential(1.0 / request_rate)


async def send_request(
    backend: str,
    api_url: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
    model_id: int,
    is_first: bool = False,
) -> None:
    request_start_time = time.time()

    headers = {"User-Agent": "Benchmark Client"}
    if backend == "vllm":
        if is_first:
            start_ts[0] = time.time()
        pload = {
            "prompt": prompt,
            "model_id": model_id,
            "n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": 0.0 if use_beam_search else 1.0,
            "top_p": 1.0,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": False,
            "ts": start_ts[0] if is_first else 0,
        }
    elif backend == "tgi":
        assert not use_beam_search
        params = {
            "best_of": best_of,
            "max_new_tokens": output_len,
            "do_sample": True,
        }
        pload = {
            "inputs": prompt,
            "parameters": params,
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")

    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(api_url, headers=headers, json=pload) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            output = json.loads(output)

            # Re-send the request if it failed.
            if "error" not in output:
                break

    request_end_time = time.time()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency))
    REQUEST_RECORD.append((model_id, request_start_time, request_end_time, prompt_len, output_len))


async def benchmark(
    backend: str,
    api_url: str,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    workload: List[Tuple[int, float]],
) -> None:
    tasks: List[asyncio.Task] = []
    is_first = True
    async for request in get_request(input_requests, args.num_models, workload):
        model_id, (prompt, prompt_len, output_len) = request
        task = asyncio.create_task(send_request(backend, api_url, prompt,
                                                prompt_len, output_len,
                                                best_of, use_beam_search, model_id, is_first=is_first))
        is_first = False
        tasks.append(task)
        # model_id = (model_id + 1) % args.num_models
    await asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/generate"
    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    input_requests = sample_requests(args.dataset, args.num_prompts, tokenizer, args.output_len)
    maf_trace = trace.Trace(args.trace_name, args.trace_path, args.start_time, args.end_time, need_sort=args.need_sort)
    workload = maf_trace.replay_to_workload(args.num_models, args.num_prompts, tot_rate=args.request_rate, cv=args.cv, interval_minutes=args.interval, map_stride=args.map_stride)

    benchmark_start_time = time.time()
    asyncio.run(benchmark(args.backend, api_url, input_requests, args.best_of,
                          args.use_beam_search, workload))
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {args.num_prompts / benchmark_time:.2f} requests/s")

    # Compute the latency statistics.
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.2f} s")
    avg_per_token_latency = np.mean([
        latency / (prompt_len + output_len)
        for prompt_len, output_len, latency in REQUEST_LATENCY
    ])
    print(f"Average latency per token: {avg_per_token_latency:.2f} s")
    lat_per_output_token = sorted([
        latency / output_len
        for _, output_len, latency in REQUEST_LATENCY
    ])
    avg_per_output_token_latency = np.mean(lat_per_output_token)
    p90_idx = int(args.num_prompts * 0.9)
    print("Average latency per output token: "
          f"{avg_per_output_token_latency:.2f} s\n"
          f"p90: {lat_per_output_token[p90_idx]:.2f} s\n")

    if args.output is not None:
        with open(args.output, "a") as file:
            if args.output_style == 0:
                file.write(f"{args.policy},{args.map_stride},{avg_per_output_token_latency},{lat_per_output_token[p90_idx]}\n")
            elif args.output_style == 1:
                file.write(f"{args.migration_type},{args.ratio},{avg_per_output_token_latency}\n")
            elif args.output_style == 2:
                file.write(f"{args.exec_type},{args.num_models},{args.num_prompts / benchmark_time}\n")
            elif args.output_style == 3:
                output = os.popen(f"bash ./ae_scripts/parse_log.sh ./ae_scripts/logs/{start_ts[0]}.log").read()
                output = output.split('\n')
                queueing_delay = float(output[0].split(':')[1])
                adjust_time = 0.0
                if len(output[1]) > 0:
                    adjust_time = float(output[1].split(':')[1])
                solver_time = 0.0
                if len(output[2]) > 0:
                    solver_time = float(output[2].split(':')[1])
                file.write(f"{args.migration_type},{args.request_rate},{queueing_delay},{adjust_time},{solver_time},{benchmark_time}\n")
            elif args.output_style == 4:
                file.write(f"{args.exec_type},{args.request_rate},{avg_per_output_token_latency}\n")
            elif args.output_style == 5:
                output = os.popen(f"bash ./ae_scripts/parse_log.sh ./ae_scripts/logs/{start_ts[0]}.log fig2b").read()
                output = output.split('\n')
                for line in output:
                    fields = line.split(' ')
                    if len(fields) != 3:
                        continue
                    replica_id = int(fields[0])
                    queueing_delay = float(fields[1])
                    exec_time = float(fields[2])
                    queueing_ratio = queueing_delay / (queueing_delay + exec_time)
                    file.write(f"{replica_id},{queueing_ratio}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["vllm", "tgi"])
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument("--best-of", type=int, default=1,
                        help="Generates `best_of` sequences per prompt and "
                             "returns the best one.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--output-len", type=int, default=-1)
    parser.add_argument("--num-models", type=int, default=1)
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                        help="Number of requests per second. If this is inf, "
                             "then all the requests are sent at time 0. "
                             "Otherwise, we use Poisson process to synthesize "
                             "the request arrival times.")
    parser.add_argument("--cv", type=float, default=1,
                        help="Coefficient of variation in gamma process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--trust-remote-code', action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument('--trace_name', type=str, default='azure_v1', help='')
    parser.add_argument('--trace_path', type=str, default='~/maf/', help='')
    parser.add_argument('--start_time', type=str, default='0.0.0', help='')
    parser.add_argument('--end_time', type=str, default='0.0.60', help='')
    parser.add_argument("--interval", type=int, default=60, help='')
    parser.add_argument('--need_sort', action='store_true', help='')
    parser.add_argument('--map_stride', type=int, default=1, help='')
    parser.add_argument('--ratio', type=float, default=1.0, help='')
    parser.add_argument('--max_len', type=int, default=2048, help='')
    parser.add_argument('--policy', type=str, default='credit', help='')
    parser.add_argument('--migration_type', type=int, default=3, help='')
    parser.add_argument('--exec_type', type=int, default=3, help='')
    parser.add_argument('--log_path', type=str, default=None, help='')
    parser.add_argument('--output', type=str, default=None, help='')
    parser.add_argument('--output_style', type=int, default=0, help='')
    args = parser.parse_args()
    main(args)
