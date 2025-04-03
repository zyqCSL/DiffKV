import argparse
import os
import pandas as pd
from typing import List, Tuple, Optional, Union
import numpy as np
import time

from vllm import EngineArgs, LLMEngine, RequestOutput

from vllm.dataset import (
    SummaryDataset,
    SummaryQuestion,
    CNNDailyMailDataset,
    # XSumDataset,
)

np.random.seed(int(time.time()))

REQUEST_ID = 0

# ------------ sequence modeling tasks ------------#
def add_summary_requests(
    engine: LLMEngine,
    dataset: SummaryDataset,
    questions: List[SummaryQuestion],
) -> None:
    global REQUEST_ID
    for question in questions:
        # print(question)
        prompt, sampling_params = question.make_request()
        engine.add_request(
            request_id=str(REQUEST_ID), prompt=prompt, sampling_params=sampling_params
        )
        dataset.register_request(question, str(REQUEST_ID))
        REQUEST_ID += 1
        

def log_llm_stats(dataset: SummaryDataset, engine: LLMEngine, log_path: str):
    os.makedirs(log_path, exist_ok=True)
    # log perplexity
    rogue_1, rogue_2, rogue_l = dataset.get_scores()
    d = {
        'rogue-1': [rogue_1],
        'rogue-2': [rogue_2],
        'rogue-l': [rogue_l],
        'num_seqs': [engine.scheduler.num_finished_seqs],
    } 
    df = pd.DataFrame(d)
    df.to_csv(f'{log_path}/rogue.csv')
    # log sparse kv
    engine.log_sparse_kv_stats(log_path)
    

def run_cnn_daily_mail_dataset(
    engine: LLMEngine,
    batch_size: int,
    log_path: str,
    label: str = 'train',
) -> None:
    ''' Args
    num_lines: how many lines to batch
    '''
    dataset = CNNDailyMailDataset()
    total_questions = len(dataset.dataset[label])
    print(total_questions)

    
    # # 287113 lines in 'train'   
    # # for debug 
    # dataset.data_ptr = 287113
    
    
    # # disable real-time perf logging
    # engine.log_stats = False
    
    # t0 = time.time()
    
    while True:
        questions = dataset.sample(
            label="train", 
            n=batch_size * 5, 
            is_random=False,
            continued=True)
        
        if not questions:
            print(f'CNN-Daily-Mail finished. data_ptr = {dataset.data_ptr}, '
                  'total questions = ', len(dataset.dataset[label]))
            break
        
        # for q in questions:
        #     print(q)
        #     import sys
        #     sys.exit()
        
        add_summary_requests(engine, dataset, questions)

        while engine.has_unfinished_requests():
            request_outputs: List[RequestOutput] = engine.step()
            for request_output in request_outputs:
                if request_output.finished:
                    # print(request_output)
                    dataset.complete_request(request_output)
                    # print(request_output)
                    print(f"request {request_output.request_id} finished")
                    
                    if engine.scheduler.num_finished_seqs > 0 and engine.scheduler.num_finished_seqs % 100 == 0: 
                        log_llm_stats(dataset, engine, log_path)
            
        print(f'{dataset.data_ptr} questions processed')
        
            
        if dataset.data_ptr >= total_questions // 10:
        # if dataset.data_ptr >= 100:
            print(f'Stop: {dataset.data_ptr} lines processed, {total_questions} in total')
            print(dataset.get_scores())
            break
    
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
    batch_size = args.max_num_seqs
    sparse_kv_config = (f'thresh_{engine.cache_config.kv_score_thresh}-'
                        f'buffer_{engine.cache_config.kv_buffer_size}-'
                        f'compress_{engine.cache_config.kv_compress_ratio}')
    
    run_cnn_daily_mail_dataset(engine, batch_size, f'/data/ellm_logs/cnn_dailymail/{sparse_kv_config}', 'train')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
