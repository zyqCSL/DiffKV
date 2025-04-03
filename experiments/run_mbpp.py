import argparse
import os
import pandas as pd
from typing import List, Tuple, Optional, Union
import numpy as np
import time
from pathlib import Path

from vllm import EngineArgs, LLMEngine, RequestOutput

from vllm.dataset import (
    CodeGenerationQuestion,
    CodeGenerationDataset,
    HumanEvalDataset,
    MbppDataset,
    MathQAPythonDataset,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MAX_GEN_LEN = 512
np.random.seed(int(time.time()))

FILE_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

REQUEST_ID = 0

# ------------ squad (reading comprehension) tasks ------------#
def add_code_generation_requests(
    engine: LLMEngine,
    dataset: CodeGenerationDataset,
    questions: List[CodeGenerationQuestion],
) -> None:
    global REQUEST_ID
    for question in questions:
        # print(question)
        prompt, sampling_params = question.make_request()
        for _i in range(dataset.sampling_n):
            engine.add_request(
                request_id=str(REQUEST_ID), prompt=prompt, sampling_params=sampling_params
            )
            dataset.register_request(question, str(REQUEST_ID))
            REQUEST_ID += 1
        

def log_llm_stats(dataset: CodeGenerationDataset, engine: LLMEngine, log_path: str):
    assert os.path.isdir(log_path)
    # log perplexity
    n_correct, n_seqs = dataset.get_scores()
    d = {
        'num_correct': [n_correct],
        'num_seqs': [n_seqs],
        # 'num_seqs': [engine.scheduler.num_finished_seqs],
        'num_tokens': [engine.scheduler.num_processed_tokens],
    } 
    df = pd.DataFrame(d)
    df.to_csv(f'{log_path}/correctness.csv')
    # log sparse kv
    engine.log_sparse_kv_stats(log_path)
    
    

def run_mbpp_dataset(
    engine: LLMEngine,
    batch_size: int,
    log_path: str,
    sampling_n: int,
    temperature: float,
    label: str,
    quiet: bool = False,
) -> None:
    ''' Args
    num_lines: how many lines to batch
    '''
    dataset = MbppDataset(sampling_n, temperature)
    total_questions = len(dataset.dataset[label])
    print('total_questions = ', total_questions)

    
    # # 287113 lines in 'train'   
    # # for debug 
    # dataset.data_ptr = 287113
    
    
    # # disable real-time perf logging
    # engine.log_stats = False
    
    # t0 = time.time()
    
    engine.log_stats = not quiet
    
    while True:
        questions = dataset.sample(
            label=label, 
            n=batch_size * 5, 
            is_random=False,
            continued=True)
        
        if not questions:
            print(f'MBPP finished. data_ptr = {dataset.data_ptr}, '
                  'total questions = ', len(dataset.dataset[label]))
            break
        
        # for q in questions:
        #     print(q)
        #     import sys
        #     sys.exit()
        
        add_code_generation_requests(engine, dataset, questions)

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
            print(f'Stop: {dataset.data_ptr} samples processed, {total_questions} in total')
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
    
    log_path = Path(FILE_DIR_PATH) / '..' / 'tmp' / 'tests' / 'mbpp' / sparse_kv_config
    os.makedirs(log_path, exist_ok=True)
    run_mbpp_dataset(
        engine=engine, 
        batch_size=batch_size, 
        log_path=str(log_path),
        sampling_n=1,
        temperature=0.0,
        label='test')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
