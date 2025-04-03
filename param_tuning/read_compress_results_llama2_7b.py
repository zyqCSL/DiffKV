import os
import pandas as pd
import numpy as np

model_gen = 2
model_size = 7
rounds = 5
workloads = [
    # 'cnn_dailymail',
    'cnn_daily',
    'squad',
    'gsm8k',
    # 'gsm8k_zeroshot',
    'gsm8k-zeroshot',
    'humaneval_k1',
    'mmlu_cot'
    ]

workloads = [
    # 'cnn_daily',
    'squad',
    'gsm8k',
    # 'humaneval_k1',
    # 'mmlu_cot'
    ]

metrics = {
    # 'cnn_dailymail': 'rogue-1',
    'cnn_daily': 'rogue-1',
    'squad': 'rogue-1',
    'gsm8k': 'correct_p',
    # 'gsm8k_zeroshot': 'correct_p',
    'gsm8k-zeroshot': 'correct_p',
    'humaneval_k1': 'correct_p',
    'mmlu_cot': 'correct_p',
}

quant_configs = {
    'cnn_daily': [
        [8, 4, 4, 2],
        # [4, 4, 4, 2], 
        [4, 2, 4, 1],
    ],
    'squad': [
        # [8, 4, 4, 2],
        # [4, 4, 4, 2], 
        [4, 2, 4, 1],
    ],
    'gsm8k': [
        # [8, 4, 8, 2],
        # [8, 4, 4, 2],
        # [4, 4, 4, 2],
        [4, 2, 4, 1],
    ],
    # 'gsm8k_zeroshot': 'correct_p',
    'gsm8k-zeroshot': [
        # [8, 4, 8, 2],
        [8, 4, 4, 2],
        # [4, 4, 4, 2],
        [4, 2, 4, 1],
    ],
    'humaneval_k1': [
        [8, 4, 4, 2],
        # [4, 4, 4, 2],
        [4, 2, 4, 1],
    ],
    'mmlu_cot': [
        [8, 4, 4, 2],
        # [4, 4, 4, 2],
        [4, 2, 4, 1],
    ],
}

target_ratio = 10   # percent
buffer = 64

summary_dir = (f'/home/zhangyanqi/Projects/eLLM/logs/compress_summary_{rounds}rounds'
    f'/llama{model_gen}-{model_size}b/')

# summary_dir = (f'/home/zhangyanqi/Projects/eLLM/logs/new_kernel_compress_summary_{rounds}rounds'
#     f'/llama{model_gen}-{model_size}b/')

os.makedirs(summary_dir, exist_ok=True)

for workload in workloads:
    print(workload)
    
    # dir = (f'/home/zhangyanqi/Projects/eLLM/logs/param_tune_{rounds}rounds'
    #        f'/llama{model_gen}-{model_size}b/{workload}/')
    
    dir = (f'/home/zhangyanqi/Projects/eLLM/logs/new_kernel_param_tune_{rounds}rounds'
           f'/llama{model_gen}-{model_size}b/{workload}/')
    metric = metrics[workload]
    
    for high_k, high_v, low_k, low_v in quant_configs[workload]:
        sub_dir = f'{dir}/k{high_k}v{high_v}_k{low_k}v{low_v}/target_{target_ratio}_buffer_{buffer}/'
        all_prune_threshes = []
        prune_thresh_to_quant_thresh = {}   # indexed by prune_thresh
        for params_f in os.listdir(sub_dir):
            assert params_f.startswith('p')
            p_thresh, q_thresh = params_f.split('_')
            p_thresh = int(p_thresh.replace('p', ''))
            _q_thresh = q_thresh.replace('q', '')
            if '.' in _q_thresh:
                q_thresh = round(float(_q_thresh), 1)
            else:
                q_thresh = int(_q_thresh)
            
            if p_thresh not in prune_thresh_to_quant_thresh:
                all_prune_threshes.append(p_thresh)
                prune_thresh_to_quant_thresh[p_thresh] = []
            prune_thresh_to_quant_thresh[p_thresh].append(q_thresh)
            
            records = {
                'prune_thresh': [],
                'quant_thresh': [],
                'low_prec_ratio': [],
                'mem_usage': [],
                'corrected_compress_ratio': [],
                metric: [],
                'std_dev': [],
            }
            
            for p_thresh in sorted(all_prune_threshes, reverse=True):
                for q_thresh in sorted(prune_thresh_to_quant_thresh[p_thresh], reverse=True):
                    records['prune_thresh'].append(p_thresh / 1000)
                    records['quant_thresh'].append(q_thresh / 1000)
                    
                    # read per round metric values
                    rounds_metrics = []
                    rounds_mem_usage = []
                    rounds_low_prec_ratio = []
                    rounds_compress_ratio = []
                    
                    for round_id in range(rounds):
                        if not os.path.isfile(f'{sub_dir}/p{p_thresh}_q{q_thresh}/round_{round_id}/eval.csv'):
                            print(f'WARNING: {sub_dir}/p{p_thresh}_q{q_thresh}/round_{round_id}/eval.csv does not exist!')
                            continue
                        df = pd.read_csv(f'{sub_dir}/p{p_thresh}_q{q_thresh}/round_{round_id}/eval.csv')
                        rounds_low_prec_ratio.append(df['low_prec_ratio'][0])
                        rounds_mem_usage.append(df['compress_ratio'][0])
                        
                        # -------------------------------------
                        # each eval is a partition of the data
                        num_tokens = 0
                        cum_kv_lens = None
                        for eval_id in range(0, 4):
                            eval_dir = f'{sub_dir}/p{p_thresh}_q{q_thresh}/round_{round_id}/eval_{eval_id}'
                            if not os.path.isdir(eval_dir):
                                break
                            
                            if os.path.isfile(f'{eval_dir}/correctness.csv'):
                                num_tokens += pd.read_csv(f'{eval_dir}/correctness.csv')['num_tokens'][0]
                            else:
                                assert os.path.isfile(f'{eval_dir}/rogue.csv')
                                num_tokens += pd.read_csv(f'{eval_dir}/rogue.csv')['num_tokens'][0]
                            
                            # each worker is a partition of the model
                            np_kv_lens = None
                            for worker_id in range(0, 4):
                                if not os.path.isfile(f'{eval_dir}/kv_len_{worker_id}.npy'):
                                    break
                                if np_kv_lens is None:
                                    np_kv_lens = np.load(f'{eval_dir}/kv_len_{worker_id}.npy')
                                    # print(np_kv_lens.shape)
                                else:
                                    np_kv_lens = np.concatenate((np_kv_lens,
                                                                np.load(f'{eval_dir}/kv_len_{worker_id}.npy')),
                                                                axis=1)
                        
                            if cum_kv_lens is None:
                                cum_kv_lens = np_kv_lens
                            else:
                                cum_kv_lens += np_kv_lens
                            # print(f'cum_kv_lens shape = {cum_kv_lens.shape}')
                        
                        _high_prec_ratio = np.mean(cum_kv_lens[:, :, 0]) / num_tokens
                        _low_prec_ratio = np.mean(cum_kv_lens[:, :, 1]) / num_tokens
                        _prune_ratio = 1 - _high_prec_ratio - _low_prec_ratio   
                        assert _high_prec_ratio >= 0 and _high_prec_ratio <= 1, _high_prec_ratio
                        assert _low_prec_ratio >= 0 and _low_prec_ratio <= 1, _low_prec_ratio
                        assert _prune_ratio > 0 and _prune_ratio <= 1, _prune_ratio
                        # computed in terms of blocks
                        # NOTE: we need to first sum up the block nums of high & low precisions
                        # computed in terms of tokens
                        _compress_ratio = _high_prec_ratio * (high_k + high_v) / 32 + \
                                        _low_prec_ratio * (low_k + low_v) / 32

                        rounds_compress_ratio.append(_compress_ratio)
                        # -------------------------------------
                        
                        if 'rogue' in metric:
                            rounds_metrics.append(df[metric][0] * 100)
                        else:
                            rounds_metrics.append(df[metric][0])
                    
                    records['low_prec_ratio'].append(np.mean(rounds_low_prec_ratio))
                    records['mem_usage'].append(np.mean(rounds_mem_usage))
                    records['corrected_compress_ratio'].append(np.mean(rounds_compress_ratio))
                    records[metric].append(np.mean(rounds_metrics))
                    records['std_dev'].append(np.std(rounds_metrics))
                    
                    
            df = pd.DataFrame(records)
            
            os.makedirs(f'{summary_dir}/{workload}', exist_ok=True)
            df.to_csv(f'{summary_dir}/{workload}/k{high_k}v{high_v}_k{low_k}v{low_v}.csv')

