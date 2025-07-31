"""Benchmark online serving latency.

On the server side, run:
    python -m vllm.entrypoints.api_server \
        --model <your_model> \
        --port 8000 \
        --kv-buffer-size 32

On the client side, run:
    python benchmark_serving.py \
        --model <your_model> \
        --port 8000 \
        --request-rate <request_rate> \
        --kv-prune-thresh <kv_prune_thresh> \
        --kv-quant-thresh <kv_quant_thresh> \
"""
import argparse
import time
from typing import List, Tuple
import asyncio
import aiohttp
import random
import json
import numpy as np

from transformers import AutoTokenizer
from tqdm import tqdm

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []


def create_request_pool(tokenizer, max_output_len):
    from datasets import load_dataset, concatenate_datasets

    subjects = ['algebra', 'counting_and_probability', 'geometry',
                'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    dataset = concatenate_datasets([
        load_dataset("EleutherAI/hendrycks_math", subject, split="test")
        for subject in subjects
    ])

    # Sample 100 requests randomly from the dataset to create the request pool.
    dataset = dataset.shuffle().select(range(100))
    request_pool = []
    for data in tqdm(dataset, desc="Creating request pool"):
        prompt = data["problem"]
        prompt_len = len(tokenizer(prompt).input_ids)
        request_pool.append((prompt, prompt_len, max_output_len))

    print("Finished creating request pool with {} requests.".format(len(request_pool)))
    return request_pool


async def send_request(
    api_url: str,
    tokenizer,
    prompt: str,
    prompt_len: int,
    max_output_len: int,
    quant_config: List[int] = None,
    compress_config: List[float] = None,
) -> None:
    request_start_time = time.perf_counter()
    headers = {"User-Agent": "Benchmark Client"}

    pload = {
        "prompt": prompt,
        "n": 1,
        "temperature": 0.0,
        "max_tokens": max_output_len,
        "ignore_eos": True,
        "stream": False,
        "quant_config": quant_config,
        "quant_groups": [1,1,1,1],
        "compress_config": compress_config,
    }

    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(api_url, headers=headers, json=pload) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            output = json.loads(output)
            # print("Output:", output)

            # Re-send the request if it failed.
            if "error" not in output:
                break

    output_len = len(tokenizer(output['text'][0]).input_ids) - prompt_len
    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency))


async def benchmark(
    api_url: str,
    request_pool: List[Tuple[str, int, int]],
    tokenizer,
    request_rate: float,
    max_completed: int,
    quant_config: List[int] = None,
    compress_config: List[float] = None,
) -> None:
    tasks: List[asyncio.Task] = []

    while True:
        # Stop once we've recorded enough completed requests.
        if len(REQUEST_LATENCY) >= max_completed:
            break

        # Pick a random prompt from the pool.
        prompt, prompt_len, max_output_len = random.choice(request_pool)
        task = asyncio.create_task(
            send_request(api_url, tokenizer, prompt, prompt_len, max_output_len, quant_config=quant_config, compress_config=compress_config)
        )
        tasks.append(task)

        # Poisson‑distributed inter‑arrival time.
        interval = 0.0 if request_rate == float("inf") \
            else np.random.exponential(1.0 / request_rate)
        await asyncio.sleep(interval)


def main(args: argparse.Namespace):
    print(args)
    api_url = f"http://{args.host}:{args.port}/generate"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    request_pool = create_request_pool(tokenizer, args.max_output_len)

    quant_config = [args.kbits_high, args.vbits_high, args.kbits_low, args.vbits_low]
    if args.kbits_high == args.kbits_low and args.vbits_high == args.vbits_low:
        quant_config = [args.kbits_high, args.vbits_high]
    compress_config = [args.kv_prune_thresh, args.kv_quant_thresh]

    benchmark_start_time = time.perf_counter()
    asyncio.run(
        benchmark(
            api_url,
            request_pool,
            tokenizer,
            args.request_rate,
            args.num_requests,
            quant_config=quant_config,
            compress_config=compress_config,
        )
    )
    benchmark_end_time = time.perf_counter()
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.2f} s")

    # Compute throughput statistics
    total_tokens = np.sum([prompt_len + output_len for prompt_len, output_len, _ in REQUEST_LATENCY])
    print(f"Throughput: {len(REQUEST_LATENCY) / benchmark_time:.3f} requests/s")
    print(f"Throughput: {total_tokens / benchmark_time:.3f} tokens/s")

    # Compute latency statistics
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.3f} s")
    avg_per_token_latency = np.mean([
        latency / (prompt_len + output_len)
        for prompt_len, output_len, latency in REQUEST_LATENCY
    ])
    print(f"Average latency per token: {avg_per_token_latency:.3f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--model",
                        type=str,
                        help="The model to use.")
    parser.add_argument("--download-dir",
                        type=str,
                        default="/data1/huggingface",
                        help="The directory to download models from huggingface.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--request-rate", type=float,
                        help="Number of requests per second. If this is inf, "
                             "then all the requests are sent at time 0. "
                             "Otherwise, we use Poisson process to synthesize "
                             "the request arrival times.")
    parser.add_argument("--num-requests",
                        type=int,
                        default=512,
                        help="The number of requests to finish.")
    parser.add_argument("--max-output-len",
                        type=int,
                        default=4096,
                        help="The maximum output length.")
    parser.add_argument("--tensor-parallel-size",
                        type=int,
                        default=1,
                        help="The tensor parallel size.")
    parser.add_argument("--kbits-high", type=int, default=8)
    parser.add_argument("--vbits-high", type=int, default=4)
    parser.add_argument("--kbits-low", type=int, default=4)
    parser.add_argument("--vbits-low", type=int, default=2)
    parser.add_argument("--kv-prune-thresh", type=float)
    parser.add_argument("--kv-quant-thresh", type=float)
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)

    args = parser.parse_args()
    main(args)
