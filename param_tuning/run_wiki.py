import argparse
import os
import sys
import pandas as pd
from typing import List, Tuple, Optional, Union
import numpy as np
import time

from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput

from vllm.dataset import (
    ModelingDataset,
    WikiDataset,
)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

np.random.seed(666)

REQUEST_ID = 0

# ------------ sequence modeling tasks ------------#
def add_modeling_requests(
    engine: LLMEngine,
    dataset: ModelingDataset,
    requests: List[Tuple[List[int], SamplingParams]],
    quant_configs: List[Tuple[int]],
    compress_configs: List[float],
) -> None:
    global REQUEST_ID
    for prompt_token_ids, sampling_params in requests:
        seq_group = engine.add_request(
            request_id=str(REQUEST_ID),
            prompt=None,
            sampling_params=sampling_params,
            attn_prune_thresh=0.0,
            prompt_token_ids=prompt_token_ids,
            quant_configs=quant_configs, 
            compress_configs=compress_configs,
        )
        dataset.register_request(seq_group)
        REQUEST_ID += 1

def log_llm_stats(dataset: ModelingDataset, engine: LLMEngine, log_path: str):
    os.makedirs(log_path, exist_ok=True)
    # log perplexity
    d = {
        'cum_logprobs': [dataset.cum_logprobs],
        'num_gen_tokens': [dataset.num_tokens],
        'perplexity': [dataset.perplexity()],
        'num_tokens': [engine.scheduler.num_processed_tokens],
        'num_seqs': [engine.scheduler.num_finished_seqs],
        'num_blocks': [engine.scheduler.baseline_num_blocks],
    } 
    df = pd.DataFrame(d)
    df.to_csv(f'{log_path}/perplexity.csv')
    # log sparse kv
    engine.log_sparse_kv_stats(log_path)
    

def run_wiki_dataset(
    engine: LLMEngine,
    batch_size: int,
    start_line: int,
    end_line: int,
    log_path: str,
    kbits_high: int,
    vbits_high: int,
    kbits_low: int,
    vbits_low: int,
    kv_prune_thresh: float,
    kv_quant_thresh: float,
    label: str = 'train',
    quiet: bool = True,
) -> None:
    ''' Args
    num_lines: how many lines to batch
    '''
    dataset = WikiDataset(
        prompt_tokens=512, 
        max_tokens=4096, 
        min_tokens=1024, 
        tokenizer=engine.tokenizer)
    
    total_questions = len(dataset.dataset[label])
    print(total_questions)

    # # 287113 lines in 'train'   
    # # for debug 
    # dataset.data_ptr = 287113
    
    if kbits_high == kbits_low and vbits_high == vbits_low:
        quant_configs = [kbits_high, vbits_high]
    else:
        quant_configs = [kbits_high, vbits_high, 
                         kbits_low, vbits_low]
    # compress_configs = [kv_prune_thresh, kv_quant_thresh, 
    #                     kv_prune_ratio, kv_quant_ratio]
    compress_configs = [kv_prune_thresh, kv_quant_thresh]
    
    
    # disable real-time perf logging
    engine.log_stats = not quiet
    
    num_lines = end_line - start_line
    # t0 = time.time()
    all_questions = dataset.sample(
        label='train',
        start_line=start_line,
        num_lines=num_lines,
        n=None,
        is_random=False,
    )
    
    while all_questions:
        questions = all_questions[:batch_size]
        # discard running questions
        all_questions = all_questions[batch_size:]
        
        add_modeling_requests(
            engine, dataset, questions, quant_configs, compress_configs)

        while engine.has_unfinished_requests():
            request_outputs: List[RequestOutput] = engine.step()
            for request_output in request_outputs:
                if request_output.finished:
                    # print(request_output)
                    dataset.complete_request(request_output)
                    # print(request_output)
                    if not quiet:
                        print(f"request {request_output.request_id} finished")
                    
                    if engine.scheduler.num_finished_seqs > 0 and engine.scheduler.num_finished_seqs % 100 == 0: 
                        log_llm_stats(dataset, engine, log_path)
    
    # print(f'{time.time() - t0} seconds elapsed')
    
    # log stats when the dataset is finished
    log_llm_stats(dataset, engine, log_path)


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    engine.scheduler.prompt_limit = args.prompt_limit
    batch_size = args.max_num_seqs
    assert args.indices_csv.endswith('.csv')
    df = pd.read_csv(args.indices_csv)
    start_line = df['start_line'].tolist()[0]
    end_line = df['end_line'].tolist()[0]
    # sparse_kv_config = (f'thresh_{engine.cache_config.kv_score_thresh}-'
    #                     f'buffer_{engine.cache_config.kv_buffer_size}')
    
    run_wiki_dataset(
        engine=engine, 
        batch_size=batch_size * 2, 
        start_line=start_line,
        end_line=end_line,
        log_path=args.log_path, 
        kbits_high=args.kbits_high,
        vbits_high=args.vbits_high,
        kbits_low=args.kbits_low,
        vbits_low=args.vbits_low,
        kv_prune_thresh=args.kv_prune_thresh,
        kv_quant_thresh=args.kv_quant_thresh,
        label=args.data_label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser = EngineArgs.add_cli_args(parser)
    parser.add_argument('--log-path', type=str, required=True)
    parser.add_argument('--indices-csv', type=str, required=True)
    parser.add_argument('--data-label', type=str, default='train')
    parser.add_argument('--prompt-limit', type=int, required=True)
    # dataset specific compression params
    parser.add_argument('--kbits-high', type=int, required=True)
    parser.add_argument('--vbits-high', type=int, required=True)
    parser.add_argument('--kbits-low', type=int, required=True)
    parser.add_argument('--vbits-low', type=int, required=True)
    parser.add_argument('--kv-prune-thresh', type=float, required=True)
    parser.add_argument('--kv-quant-thresh', type=float, required=True)
    
    args = parser.parse_args()
    main(args)
