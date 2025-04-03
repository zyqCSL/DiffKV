import argparse
import os
import pandas as pd
from typing import List, Tuple, Optional, Union
import numpy as np
import time

from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput

from vllm.dataset import (
    ModelingDataset,
    WikiDataset,
)

np.random.seed(int(time.time()))

REQUEST_ID = 0

# ------------ sequence modeling tasks ------------#
def add_modeling_requests(
    engine: LLMEngine,
    dataset: ModelingDataset,
    requests: List[Tuple[List[int], SamplingParams]],
) -> None:
    global REQUEST_ID
    for prompt_token_ids, sampling_params in requests:
        seq_group = engine.add_request(
            request_id=str(REQUEST_ID),
            prompt=None,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
        )
        dataset.register_request(seq_group)
        REQUEST_ID += 1

def log_llm_stats(dataset: ModelingDataset, engine: LLMEngine, log_path: str):
    os.makedirs(log_path, exist_ok=True)
    # log perplexity
    d = {
        'cum_logprobs': [dataset.cum_logprobs],
        'num_tokens': [dataset.num_tokens],
        'perplexity': [dataset.perplexity()],
        'num_seqs': [engine.scheduler.num_finished_seqs],
    } 
    df = pd.DataFrame(d)
    df.to_csv(f'{log_path}/perplexity.csv')
    # log sparse kv
    engine.log_sparse_kv_stats(log_path)
    

def run_wiki_dataset(
    engine: LLMEngine,
    batch_size: int,
    num_lines: int,
    log_path: str,
    label: str = 'train',
) -> None:
    ''' Args
    num_lines: how many lines to batch
    '''
    dataset = WikiDataset(
        prompt_tokens=256, max_tokens=4096, min_tokens=1024, tokenizer=engine.tokenizer)
    
    # dataset = WikiDataset(
    #     prompt_tokens=256, max_tokens=2048, min_tokens=1024, tokenizer=engine.tokenizer)
    
    # dataset = WikiDataset(
    #     prompt_tokens=128, max_tokens=512, min_tokens=256, tokenizer=engine.tokenizer)
    
    wiki_total_lines = len(dataset.dataset[label]['text'])
    
    # # 1801350 lines in 'train'   
    # # for debug 
    # dataset.line_ptr = 1800000
    
    # disable real-time perf logging
    engine.log_stats = False
    
    while True:
        requests = dataset.sample(
            label=label, num_lines=num_lines, n=None, 
            is_random=False, continued=True)
        if not requests:
            print(f'Wiki-text finished. line_ptr = {dataset.line_ptr}, '
                  'total_lines = ', len(dataset.dataset[label]['text']))
            break
        
        while requests:
            batch_requests = requests[:batch_size]
            requests = requests[batch_size:]
            
            for i, r in enumerate(batch_requests):
                print(i, len(r[1].truth_token_ids))
            
            # batch_requests = batch_requests[4:5]
            
            add_modeling_requests(engine, dataset, batch_requests)

            while engine.has_unfinished_requests():
                request_outputs: List[RequestOutput] = engine.step()
                for request_output in request_outputs:
                    if request_output.finished:
                        dataset.complete_request(request_output)
                        # print(request_output)
                        print(f"request {request_output.request_id} finished")
                        
                        if engine.scheduler.num_finished_seqs % 100 == 0:
                            log_llm_stats(dataset, engine, log_path)
            
            print(f'{dataset.line_ptr} lines processed')
            
            if dataset.line_ptr >= wiki_total_lines // 10:
                print(f'Stop: {dataset.line_ptr} lines processed, {wiki_total_lines} in total')
    
    # log stats when the dataset is finished
    log_llm_stats(dataset, engine, log_path)

def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    batch_size = args.max_num_seqs
    sparse_kv_config = (f'thresh_{engine.cache_config.kv_score_thresh}-'
                        f'buffer_{engine.cache_config.kv_buffer_size}')
    
    run_wiki_dataset(engine, batch_size, 2048, f'/data/ellm_logs/{sparse_kv_config}', 'train')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
