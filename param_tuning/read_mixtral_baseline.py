import os
import pandas as pd
import numpy as np

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
        [8, 8, 8, 8],
        [8, 4, 8, 4],
        [4, 4, 4, 4],
        [4, 2, 4, 2],
    ],
    'squad': [
        [8, 8, 8, 8],
        [8, 4, 8, 4],
        [4, 4, 4, 4],
        [4, 2, 4, 2],
    ],
    'gsm8k': [
        [8, 8, 8, 8],
        [8, 4, 8, 4],
        [4, 4, 4, 4],
        [4, 2, 4, 2],
    ],
    # 'gsm8k_zeroshot': 'correct_p',
    'gsm8k-zeroshot': [
        [8, 8, 8, 8],
        [8, 4, 8, 4],
        [4, 4, 4, 4],
        [4, 2, 4, 2],
    ],
    'humaneval_k1': [
        [8, 8, 8, 8],
        [8, 4, 8, 4],
        [4, 4, 4, 4],
        [4, 2, 4, 2],
    ],
    'mmlu_cot': [
        [8, 8, 8, 8],
        [8, 4, 8, 4],
        [4, 4, 4, 4],
        [4, 2, 4, 2],
    ],
}

# target_ratio = 100   # percent
# buffer = 8192
target_ratio = 10   # percent
buffer = 32
rounds = 5

summary_dir = (
    f'/home/zhangyanqi/Projects/eLLM/logs/quant_baseline_summary_{rounds}rounds'
    f'/mixtral-8x7b')

os.makedirs(summary_dir, exist_ok=True)

for workload in workloads:
    print(workload)
    
    dir = (f'/home/zhangyanqi/Projects/eLLM/logs/param_tune_{rounds}rounds'
           f'/mixtral-8x7b/{workload}')
    metric = metrics[workload]
    
    for high_k, high_v, low_k, low_v in quant_configs[workload]:
        sub_dir = f'{dir}/k{high_k}v{high_v}_k{low_k}v{low_v}/target_{target_ratio}_buffer_{buffer}/'
        all_prune_threshes = []
        prune_thresh_to_quant_thresh = {}   # indexed by prune_thresh
        for params_f in os.listdir(sub_dir):
            assert params_f.startswith('p')
            p_thresh, q_thresh = params_f.split('_')
            p_thresh = int(p_thresh.replace('p', ''))
            q_thresh = int(q_thresh.replace('q', ''))
            
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
                        csv_n = f'{sub_dir}/p{p_thresh}_q{q_thresh}/round_{round}/eval.csv'
                        if not os.path.isfile(csv_n):
                            print(f'WARNING: {csv_n} does not exist!')
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

