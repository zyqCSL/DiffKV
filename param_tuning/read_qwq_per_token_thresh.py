import os
import pandas as pd
import numpy as np

model_size = 32

LOG_DIR = os.getenv('DIFFKV_LOG_DIR', '/home/zhangyanqi/git_repos/DiffKV/logs')
AIME_ROLLOUT = 3
AIME_NUM_PROBLEMS = 30

def count_actual_problems(round_dir: str) -> int:
    n: int = 0
    for fn in os.listdir(round_dir):
        if fn.startswith('eval_') and os.path.isdir(f'{round_dir}/{fn}'):
            n += pd.read_csv(f'{round_dir}/{fn}/correctness.csv')['num_seqs'][0]
    return n

workloads = [
    'minerva_math',
    'aime',
    'gpqa',
    ]

metrics = {
    'minerva_math': 'correct_p',
    'aime': 'correct_p',
    'gpqa': 'correct_p',
}

quant_configs = {
    'minerva_math': [
        [8, 8, 8, 8],
        [8, 4, 4, 2],
        [4, 4, 4, 4],
    ],
    'aime': [
        [8, 8, 8, 8],
        [8, 4, 4, 2],
        [4, 4, 4, 4],
    ],
    'gpqa': [
        [8, 8, 8, 8],
        [8, 4, 4, 2],
        [4, 4, 4, 4],
    ],
}

buffer = 64
rounds = 15

summary_dir = (
    f'{LOG_DIR}/per_token_thresh_compress_summary'
    f'/kv_buffer_{buffer}/qwq-{model_size}b')

os.makedirs(summary_dir, exist_ok=True)

for workload in workloads:
    print(workload)
    
    dir = (f'{LOG_DIR}/per_token_thresh'
           f'/qwq-{model_size}b/{workload}/')
    metric = metrics[workload]
    
    for high_k, high_v, low_k, low_v in quant_configs[workload]:
        sub_dir = f'{dir}/k{high_k}v{high_v}_k{low_k}v{low_v}/buffer_{buffer}/'
        all_prune_threshes = []
        prune_thresh_to_quant_thresh = {}   # indexed by prune_thresh
        
        p_to_pstr = {}
        q_to_qstr = {}
        
        for params_f in os.listdir(sub_dir):
            if not params_f.startswith('p'):
                continue
            assert params_f.startswith('p')
            p_thresh_str, q_thresh_str = params_f.split('_')
            p_thresh = float(p_thresh_str.replace('p', ''))
            q_thresh = float(q_thresh_str.replace('q', ''))
            
            p_to_pstr[p_thresh] = p_thresh_str
            q_to_qstr[q_thresh] = q_thresh_str
            
            if p_thresh not in prune_thresh_to_quant_thresh:
                all_prune_threshes.append(p_thresh)
                prune_thresh_to_quant_thresh[p_thresh] = []
            prune_thresh_to_quant_thresh[p_thresh].append(q_thresh)
            
        records = {
            'prune_thresh': [],
            'quant_thresh': [],
            'low_prec_ratio': [],
            'mem_usage': [],
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
                
                for round in range(rounds):                
                    p_thresh_str = p_to_pstr[p_thresh]
                    q_thresh_str = q_to_qstr[q_thresh]
                    
                    csv_n = f'{sub_dir}/{p_thresh_str}_{q_thresh_str}/round_{round}/eval.csv'
                    if not os.path.isfile(csv_n):
                        print(f'Warning: {csv_n} does not exist')
                        continue
                    
                    if workload == 'aime':
                        actual_problems = count_actual_problems(f'{sub_dir}/{p_thresh_str}_{q_thresh_str}/round_{round}')
                        if actual_problems < AIME_NUM_PROBLEMS * AIME_ROLLOUT:
                            print(f'Warning: Only {actual_problems} problems recorded, expected {AIME_NUM_PROBLEMS * AIME_ROLLOUT}, round_{round} data aborted')
                            continue
                    
                    df = pd.read_csv(csv_n)
                    rounds_low_prec_ratio.append(df['low_prec_ratio'][0])
                    rounds_mem_usage.append(df['compress_ratio'][0])
                    if 'rogue' in metric:
                        rounds_metrics.append(df[metric][0] * 100)
                    else:
                        rounds_metrics.append(df[metric][0])
                
                records['low_prec_ratio'].append(np.mean(rounds_low_prec_ratio))
                records['mem_usage'].append(np.mean(rounds_mem_usage))
                records[metric].append(np.mean(rounds_metrics))
                records['std_dev'].append(np.std(rounds_metrics))
                
        df = pd.DataFrame(records)
        
        os.makedirs(f'{summary_dir}/{workload}', exist_ok=True)
        df.to_csv(f'{summary_dir}/{workload}/k{high_k}v{high_v}_k{low_k}v{low_v}.csv')

