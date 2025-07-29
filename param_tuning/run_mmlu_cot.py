import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import sys
import pandas as pd
from typing import List, Tuple, Optional, Union
import numpy as np
import time
import torch

from vllm import EngineArgs, LLMEngine, RequestOutput

from vllm.dataset import (
    MMLUCoTQuestion,
    MMLUCoTDataset,
)

from util import maybe_destroy_process_group, get_quant_configs_and_groups

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# NOTE: Maximum recursion depth exceeded in comparison
# https://github.com/pltrdy/rouge/issues/19
MAX_GEN_LEN = 512
sys.setrecursionlimit(MAX_GEN_LEN * MAX_GEN_LEN + 10)

np.random.seed(666)

REQUEST_ID = 0

# ------------ mbpp (code generation) tasks ------------#
def add_mmlu_cot_questions(
    engine: LLMEngine,
    dataset: MMLUCoTDataset,
    questions: List[MMLUCoTQuestion],
    quant_configs: List[int],
    quant_groups: List[int],
    compress_configs: List[float],
) -> None:
    global REQUEST_ID
    for question in questions:
        prompt, SamplingParams = question.make_request()
        engine.add_request(
            request_id=str(REQUEST_ID),
            prompt=prompt,
            sampling_params=SamplingParams,
            quant_configs=quant_configs, 
            quant_groups=quant_groups,
            compress_configs=compress_configs,
        )
        dataset.register_request(question, str(REQUEST_ID))
        REQUEST_ID += 1
        

def log_llm_stats(dataset: MMLUCoTDataset, engine: LLMEngine, log_path: str):
    assert os.path.isdir(log_path)
    # log perplexity
    n_correct, n_seqs = dataset.get_scores()
    d = {
        'num_correct': [n_correct],
        'num_seqs': [n_seqs],
        # 'num_seqs': [engine.scheduler.num_finished_seqs],
        'num_tokens': [engine.scheduler.num_processed_tokens],
        'num_blocks': [engine.scheduler.baseline_num_blocks],
    } 
    df = pd.DataFrame(d)
    df.to_csv(f'{log_path}/correctness.csv')
    # log sparse kv
    engine.log_sparse_kv_stats(log_path)
    

def run_mmlu_cot_dataset(
    engine: LLMEngine,
    batch_size: int,
    indices: List[int],
    log_path: str,
    label: str,
    kbits_high: int,
    vbits_high: int,
    kbits_low: int,
    vbits_low: int,
    kv_prune_thresh: float,
    kv_quant_thresh: float,
    # kv_prune_ratio: float,
    # kv_quant_ratio: float,
    quiet: bool = True,
) -> None:
    ''' Args
    num_lines: how many lines to batch
    '''
    dataset = MMLUCoTDataset()
    total_questions = len(dataset.dataset[label])
    print('total_questions = ', total_questions)

    # # 99842 questions in auxiliary_train
    # # 14042 questions in test   
    # # 1531 questions in test
    # # for debug 
    # dataset.data_ptr = 287113
    
    quant_configs, quant_groups = get_quant_configs_and_groups(
        kbits_high, vbits_high, kbits_low, vbits_low)
    compress_configs = [kv_prune_thresh, kv_quant_thresh]
        
    # disable real-time perf logging
    engine.log_stats = not quiet
    
    # t0 = time.time()
    all_questions = dataset.sample(
        label=label,
        n=None,
        is_random=False,
        indices=indices,
    )
    
    while all_questions:
        questions = all_questions[:batch_size]
        # discard running questions
        all_questions = all_questions[batch_size:]
        
        add_mmlu_cot_questions(
            engine, dataset, questions, quant_configs, quant_groups, compress_configs)

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
    if 'Mistral' in args.model:
        args.max_model_len = 4096
    elif 'Mixtral' in args.model:
        args.max_model_len = 8192
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    t0 = time.time()
    
    engine = initialize_engine(args)
    engine.scheduler.prompt_limit = args.prompt_limit
    batch_size = args.max_num_seqs
    assert args.indices_csv.endswith('.csv')
    df = pd.read_csv(args.indices_csv)
    indices = df['indices'].tolist()
    # sparse_kv_config = (f'thresh_{engine.cache_config.kv_score_thresh}-'
    #                     f'buffer_{engine.cache_config.kv_buffer_size}')
    
    run_mmlu_cot_dataset(
        engine=engine, 
        batch_size=batch_size * 5, 
        indices=indices, 
        log_path=args.log_path, 
        kbits_high=args.kbits_high,
        vbits_high=args.vbits_high,
        kbits_low=args.kbits_low,
        vbits_low=args.vbits_low,
        kv_prune_thresh=args.kv_prune_thresh,
        kv_quant_thresh=args.kv_quant_thresh,
        label=args.data_label)
    print(f'--- time elapsed = {time.time() - t0}s ---')
    
    maybe_destroy_process_group()

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
