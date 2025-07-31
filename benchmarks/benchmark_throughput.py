"""Benchmark offline inference throughput."""
import argparse
import time
from typing import List, Tuple

from transformers import AutoTokenizer
from tqdm import tqdm


def sample_requests(tokenizer, num_requests, max_output_len):
    from datasets import load_dataset, concatenate_datasets

    subjects = ['algebra', 'counting_and_probability', 'geometry',
                'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    dataset = concatenate_datasets([
        load_dataset("EleutherAI/hendrycks_math", subject, split="test")
        for subject in subjects
    ])

    requests = []
    for data in dataset:
        prompt = data["problem"]
        prompt_len = len(tokenizer(prompt).input_ids)
        requests.append((prompt, prompt_len, max_output_len))
        if len(requests) == num_requests:
            break

    return requests


def run_diffkv(
    model: str,
    download_dir: str,
    requests: List[Tuple[str, int, int]],
    tensor_parallel_size : int,
    max_batch_size: int,
    quant_config: List[int],
    compress_config: List[float],
    gpu_memory_utilization: float,
) -> float:
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model,
        download_dir=download_dir,
        load_format='safetensors',
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=max_batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype='float16',
        enforce_eager=True,
        kv_buffer_size=64,
    )

    for prompt, _, max_output_len in requests:
        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            use_beam_search=False,
            ignore_eos=True,
            max_tokens=max_output_len,
        )
        llm._add_request(
            prompt=prompt,
            prompt_token_ids=None,
            sampling_params=sampling_params,
            quant_configs=quant_config,
            quant_groups=[1,1,1,1],
            compress_configs=compress_config,
        )

    start = time.perf_counter()
    outputs = llm._run_engine(use_tqdm=True)
    end = time.perf_counter()
    elapsed_time = end - start

    total_num_tokens = 0
    for output in outputs:
        prompt_len = len(output.prompt_token_ids)
        output_len = len(output.outputs[0].token_ids)
        total_num_tokens += (prompt_len + output_len)

    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")


def main(args: argparse.Namespace):
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    requests = sample_requests(tokenizer, args.num_requests, args.max_output_len)

    quant_config = [args.kbits_high, args.vbits_high, args.kbits_low, args.vbits_low]
    if args.kbits_high == args.kbits_low and args.vbits_high == args.vbits_low:
        quant_config = [args.kbits_high, args.vbits_high]
    compress_config = [args.kv_prune_thresh, args.kv_quant_thresh]
    run_diffkv(args.model, args.download_dir, requests,
               args.tensor_parallel_size, args.max_batch_size,
               quant_config, compress_config,
               args.gpu_memory_utilization)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--model",
                        type=str,
                        help="The model to use.")
    parser.add_argument("--download-dir",
                        type=str,
                        help="The directory to download models from huggingface.")
    parser.add_argument("--num-requests",
                        type=int,
                        default=512,
                        help="The number of requests to sample.")
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
