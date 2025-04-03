# squad
import argparse
import os
import subprocess
import time
import pandas as pd
import numpy as np
from typing import List

from vllm.dataset import (
    SquadDatasetV1,
    SquadDatasetV2,
)

FILE_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.insert(0, FILE_DIR_PATH)
from util import check_vllm_stderr, compose_cmd

np.random.seed(666)

# vllm config
# GPUS = list(range(8))
GPUS = list(range(0, 4))
GPU_MEM_UTIL = 0.75
BATCH_SIZE = 256 # max number of concurrently running seqs in vllm
# MAX_PADDING = 4096
MAX_BATCHED_TOKENS = 40960
MAX_PADDING = 2048
# MAX_BATCHED_TOKENS = 32768

# resource config
MODEL_SIZE_TO_GPUS = {
    7: 1,
    8: 1,
    14: 1,
    32: 2,
    # 70: 8,
    56: 4,
    70: 4,
    72: 4,
}


# TODO: return eval metrics
def eval_qa(
    dataset: str,
    model_size: int,
    model_name: str,
    num_samples: int,
    num_evals: int,  # number of measurements required
    buffer_size: int,
    # compression configs
    kbits_high: int,
    vbits_high: int,
    kbits_low: int,
    vbits_low: int,
    kv_prune_thresh: float,
    kv_quant_thresh: float,
    # miscs
    label: str,
    log_dir: str, 
    promot_limit: int,
    reset_seed: bool = False,
):
    # we need to persist the params    
    config_d = {
        'buffer_size': [buffer_size],
        'kbits_high': [kbits_high],
        'vbits_high': [vbits_high],
        'kbits_low': [kbits_low],
        'vbits_low': [vbits_low],
        'kv_prune_thresh': [kv_prune_thresh],
        'kv_quant_thresh': [kv_quant_thresh],
    }
    config_df = pd.DataFrame(config_d)
    config_df.to_csv(f'{log_dir}/compress_config.csv')
    
    # decide sample indices
    if reset_seed:
        np.random.seed(int(time.time()))
    # indices = np.random.choice(TOTAL_SAMPLES, num_samples, replace=False).tolist()
    indices = list(range(num_samples))
    np.random.shuffle(indices)
    
    per_eval_gpus = MODEL_SIZE_TO_GPUS[model_size]
    per_eval_samples = num_samples // num_evals
    assert len(GPUS) >= num_evals * per_eval_gpus
    
    all_f1 = []
    all_exact_match = []
    
    all_num_seqs = []
    all_num_tokens = []
    all_num_baseline_blocks = []    # full FP16 KV cache blocks
    cum_f1 = []
    cum_exact_match = []
    cum_kv_lens = None
    cum_block_nums = None

    processes: List[subprocess.Popen] = []
    all_cmds: List[str] = []
    num_tasks: List[int] = []
    eval_log_dirs: List[str] = []
    start_sample_pos = 0
    
    for i in range(num_evals):
        # partition and save sample indices
        indices_csv = f'{log_dir}/sample_indices_{i}.csv'
        eval_log = f'{log_dir}/eval_{i}'
        eval_log_dirs.append(eval_log)
        os.makedirs(eval_log, exist_ok=True)
        
        d = {'indices': []}
        if i == len(GPUS) - 1:
            d['indices'] = indices[start_sample_pos:]
        else:
            d['indices'] = indices[start_sample_pos:start_sample_pos + per_eval_samples]
        num_tasks.append(len(d['indices']))
        start_sample_pos += per_eval_samples
        df = pd.DataFrame(d)
        df.to_csv(indices_csv)
        
        # rsc
        if per_eval_gpus  == 1:
            gpu_id = GPUS[i]
        else:
            gpu_id = GPUS[i * per_eval_gpus: (i + 1) * per_eval_gpus]
        
        # run vllm
        cmd = compose_cmd(
            dataset=dataset,
            model_size=model_size,
            model_name=model_name,
            gpu_id=gpu_id,
            max_num_seqs=BATCH_SIZE,
            buffer_size=buffer_size,
            # compression configs
            kbits_high=kbits_high,
            vbits_high=vbits_high,
            kbits_low=kbits_low,
            vbits_low=vbits_low,
            kv_prune_thresh=kv_prune_thresh,
            kv_quant_thresh=kv_quant_thresh,
            # framework configs
            gpu_mem_util=GPU_MEM_UTIL,
            max_batched_tokens=MAX_BATCHED_TOKENS,
            max_padding=MAX_PADDING,
            prompt_limit=promot_limit,
            # dataset
            indices_csv=indices_csv,
            label=label,
            # logging
            log_dir=eval_log,
            file_dir_path=FILE_DIR_PATH,
        )
        all_cmds.append(cmd)
        p = subprocess.Popen(cmd, shell=True, 
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
        # p = subprocess.Popen(cmd, shell=True)
        processes.append(p)
        time.sleep(0.1)
    
    for i, p in enumerate(processes):
        out, err = p.communicate()
        while check_vllm_stderr(err):
            print(f'*** gpu {GPUS[i]}\nstdout: {out}\nstderr: {err}\n***')
            rerun_cmd = all_cmds[i]
            print(f'*** rerun cmd: {rerun_cmd}')
            # rerun the failed pass
            rerun_p = subprocess.Popen(rerun_cmd, shell=True, 
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            out, err = rerun_p.communicate()
        # assert len(err) == 0
        print(f'*** gpu {GPUS[i]}\nstdout: {out}\n***')
        # p.wait()
        
        eval_log = eval_log_dirs[i]
        
        # TODO: read the eval metrics
        partition_metric_df = pd.read_csv(f'{eval_log}/rogue.csv')
        worker_id = 0   # assume single gpu for 7b model
        
        partition_kv_len_np = None
        partition_block_num_np = None
        for worker_id in range(0, 4):
            if not os.path.isfile(f'{eval_log}/kv_len_{worker_id}.npy'):
                break
            if partition_kv_len_np is None:
                partition_kv_len_np = np.load(f'{eval_log}/kv_len_{worker_id}.npy')
                partition_block_num_np = np.load(f'{eval_log}/block_num_{worker_id}.npy')
            else:
                partition_kv_len_np = np.concatenate((partition_kv_len_np,
                                                     np.load(f'{eval_log}/kv_len_{worker_id}.npy')), 
                                                     axis=1)
                partition_block_num_np = np.concatenate((partition_block_num_np,
                                                         np.load(f'{eval_log}/block_num_{worker_id}.npy')),
                                                         axis=1)
        
        num_seqs = partition_metric_df['num_seqs'][0]
        num_tokens = partition_metric_df['num_tokens'][0]
        num_blocks = partition_metric_df['num_blocks'][0]
        if num_seqs != num_tasks[i]: 
            print(f'Warning: process {i}, num_seqs: {num_seqs} != expected: {num_tasks[i]}')
        
        all_num_seqs.append(num_seqs)
        all_num_tokens.append(num_tokens)
        all_num_baseline_blocks.append(num_blocks)
        if cum_kv_lens is None:
            cum_kv_lens = partition_kv_len_np
            cum_block_nums = partition_block_num_np
        else:
            cum_kv_lens += partition_kv_len_np
            cum_block_nums += partition_block_num_np
        
        # accuracy metrics
        cum_f1.append(
            partition_metric_df['rogue-1'][0] * num_seqs) 
        cum_exact_match.append(
           partition_metric_df['exact_match'][0] * num_seqs)
        
        all_f1.append(
            partition_metric_df['rogue-1'][0])
        all_exact_match.append(
            partition_metric_df['exact_match'][0])
        
        # / num_seqs cancel off for np.mean(partition_kv_len_np) & num_tokens
        _high_prec_ratio = np.mean(partition_kv_len_np[:, :, 0]) / num_tokens
        _low_prec_ratio = np.mean(partition_kv_len_np[:, :, 1]) / num_tokens
        _prune_ratio = 1 - _high_prec_ratio - _low_prec_ratio   
        assert _high_prec_ratio >= 0 and _high_prec_ratio <= 1, _high_prec_ratio
        assert _low_prec_ratio >= 0 and _low_prec_ratio <= 1, _low_prec_ratio
        assert _prune_ratio > 0 and _prune_ratio <= 1, _prune_ratio
        # computed in terms of blocks
        # NOTE: we need to first sum up the block nums of high & low precisions
        _phy_compress_ratio = np.mean(np.sum(partition_block_num_np, axis=2)) / num_blocks
        assert _phy_compress_ratio > 0 and _phy_compress_ratio <= 1, _phy_compress_ratio
        # computed in terms of tokens
        _compress_ratio = _high_prec_ratio * (kbits_high + vbits_high) / 32 + \
                          _low_prec_ratio * (kbits_low + vbits_low) / 32
        assert _compress_ratio > 0 and _compress_ratio <= 1, _compress_ratio 
        
        mem_df = {
            'high_prec_ratio': [_high_prec_ratio],
            'low_prec_ratio': [_low_prec_ratio],
            'prune_ratio': [_prune_ratio],
            'phyiscal_compress_ratio': [_phy_compress_ratio],
            'compress_ratio': [_compress_ratio],
        }
        pd.DataFrame(mem_df).to_csv(f'{eval_log}/compress_ratio.csv')
                
    unified_f1 = np.sum(cum_f1) / np.sum(all_num_seqs)
    unified_exact_match = np.sum(cum_exact_match) / np.sum(all_num_seqs)
    
    unified_high_prec_ratio = np.mean(cum_kv_lens[:, :, 0]) / np.sum(all_num_tokens)
    unified_low_prec_ratio = np.mean(cum_kv_lens[:, :, 1]) / np.sum(all_num_tokens)
    unified_prune_ratio = 1 - unified_high_prec_ratio - unified_low_prec_ratio
    
    # physical compress ratio (blocks)
    unified_phy_compress_ratio = np.mean(np.sum(cum_block_nums, axis=2)) / np.sum(all_num_baseline_blocks)
    assert unified_phy_compress_ratio > 0 and unified_phy_compress_ratio <= 1, unified_phy_compress_ratio
    # theoretic compress ratio (tokens)
    unified_compress_ratio = unified_high_prec_ratio * (kbits_high + vbits_high) / 32 + \
                             unified_low_prec_ratio * (kbits_low + vbits_low) / 32
    assert unified_compress_ratio > 0 and unified_compress_ratio <= 1, unified_compress_ratio
       
    # pretend to be noise-free
    noise_free_eval = {
        'rogue': (unified_f1 * 100, 0), 
        'rogue-1': (unified_f1, 0), 
        'exact_match': (unified_exact_match, 0),
        'high_prec_ratio': (unified_high_prec_ratio, 0),
        'low_prec_ratio': (unified_low_prec_ratio, 0),
        'prune_ratio': (unified_prune_ratio, 0),
        'physical_compress_ratio': (unified_phy_compress_ratio, 0),
        'compress_ratio': (unified_compress_ratio, 0),
    }
    pd.DataFrame({k: list(noise_free_eval[k]) for k in noise_free_eval}).to_csv(
                f'{log_dir}/eval.csv')
        

def main(args: argparse.Namespace):
    model_gen = args.model_gen
    model_size = args.model_size
    
    if args.model == 'llama':
        if model_gen == 2:
            assert model_size in [7, 13, 70]
            model_name = f'meta-llama/Llama-2-{model_size}b-chat-hf'
        else:
            assert model_size in [8, 70]
            model_name = f'meta-llama/Meta-Llama-3-{model_size}B-Instruct'
    elif args.model == 'mistral':
        model_name = 'mistralai/Mistral-7B-Instruct-v0.2'
    elif args.model == 'mixtral':
        model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    elif args.model == 'qwen2':
        assert model_size in [7, 32, 72]
        model_name = f'/data1/modelscope/Qwen2.5-{model_size}B-Instruct'
    elif args.model == 'qwq':
        assert model_size in [32]
        model_name = f'/data1/modelscope/QWQ-{model_size}B'
    
    assert model_size in MODEL_SIZE_TO_GPUS
    num_gpus_per_eval = MODEL_SIZE_TO_GPUS[model_size]
    assert len(GPUS) >= num_gpus_per_eval
    num_evals = len(GPUS) // num_gpus_per_eval 
    
    # prompt limit
    global PROMPT_LIMIT
    if args.model_gen == 3 or args.model == 'mixtral':
        prompt_limit = 7000
    else:
        prompt_limit=3500
    
    assert args.dataset in ['squad']
    args.log_path = (f'{args.log_path}/{args.dataset}/'
                     f'k{args.kbits_high}v{args.vbits_high}_k{args.kbits_low}v{args.vbits_low}/'
                     f'target_{int(100 *args.target_mem_util)}_buffer_{args.kv_buffer}/'
                     f'p{int(1000 * args.kv_prune_thresh)}_q{int(1000 * args.kv_quant_thresh)}/'
                     )
    os.makedirs(args.log_path, exist_ok=True)
    
    label = args.label
    use_v2 = not args.v1
    num_samples = args.num_samples
    if num_samples == 0:
        # dataset config
        if args.dataset == 'squad':
            dataset = SquadDatasetV2() if use_v2 else SquadDatasetV1()
            num_samples = len(dataset.dataset[label])
        else:
            print(f'Error: invalid dataset {args.dataset}')
            sys.exit()
    
    rounds = args.rounds
    for round in range(rounds):
        round_log_path = f'{args.log_path}/round_{round}/'
        os.makedirs(round_log_path, exist_ok=True)
        
        t0 = time.time()

        eval_qa(
            dataset=args.dataset,
            model_size=model_size,
            model_name=model_name,
            num_samples=num_samples,
            num_evals=num_evals,
            buffer_size=args.kv_buffer,
            # compression configs
            kbits_high=args.kbits_high,
            vbits_high=args.vbits_high,
            kbits_low=args.kbits_low,
            vbits_low=args.vbits_low,
            kv_prune_thresh=args.kv_prune_thresh,
            kv_quant_thresh=args.kv_quant_thresh,
            # miscs
            label=label,
            log_dir=round_log_path, 
            promot_limit=prompt_limit,
            reset_seed=rounds > 1)
        
        print(f'--- master elapsed time = {time.time() - t0} s---')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='squad')
    parser.add_argument('--log-path', type=str, required=True)
    parser.add_argument('--label', type=str, default='validation')
    parser.add_argument('--v1', action='store_true')
    parser.add_argument('--num-samples', type=int, default=0)
    
    # dataset specific compression params
    parser.add_argument('--kv-buffer', type=int, default=32)
    parser.add_argument('--kbits-high', type=int, required=True)
    parser.add_argument('--vbits-high', type=int, required=True)
    parser.add_argument('--kbits-low', type=int, required=True)
    parser.add_argument('--vbits-low', type=int, required=True)
    parser.add_argument('--kv-prune-thresh', type=float, required=True)
    parser.add_argument('--kv-quant-thresh', type=float, required=True)
    parser.add_argument('--target-mem-util', type=float, required=True)
    # parser.add_argument('--kv-prune-ratio', type=float, required=True)
    # parser.add_argument('--kv-quant-ratio', type=float, required=True)
    
    # model config
    parser.add_argument('--model', type=str, default='llama')
    parser.add_argument('--model-gen', type=int, required=True)
    parser.add_argument('--model-size', type=int, required=True)
    args = parser.parse_args()
    main(args)