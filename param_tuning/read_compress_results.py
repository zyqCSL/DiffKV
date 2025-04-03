import os
import pandas as pd

model_gen = 2
model_size = 70
workloads = [
    # 'cnn_dailymail',
    'cnn_daily',
    'squad',
    'gsm8k',
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
        [8, 4, 4, 2],
        # [4, 4, 4, 2], 
        [4, 2, 4, 1],
    ],
    'squad': [
        [8, 4, 4, 2],
        # [4, 4, 4, 2], 
        [4, 2, 4, 1],
    ],
    'gsm8k': [
        # [8, 4, 8, 2],
        [8, 4, 4, 2],
        # [4, 4, 4, 2],
        # [4, 2, 4, 1],
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
buffer = 32

summary_dir = ('/home/zhangyanqi/Projects/eLLM/logs/new_kernel_compress_summary'
    f'/llama{model_gen}-{model_size}b/')

os.makedirs(summary_dir, exist_ok=True)

for workload in workloads:
    print(workload)
    
    dir = ('/home/zhangyanqi/Projects/eLLM/logs/new_kernel_param_tune'
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
            }
            
            for p_thresh in sorted(all_prune_threshes, reverse=True):
                for q_thresh in sorted(prune_thresh_to_quant_thresh[p_thresh], reverse=True):
                    records['prune_thresh'].append(p_thresh / 1000)
                    records['quant_thresh'].append(q_thresh / 1000)
                    
                    df = pd.read_csv(f'{sub_dir}/p{p_thresh}_q{q_thresh}/eval.csv')
                    
                    records['low_prec_ratio'].append(df['low_prec_ratio'][0])
                    records['mem_usage'].append(df['compress_ratio'][0])
                    if 'rogue' in metric:
                        records[metric].append(df[metric][0] * 100)
                    else:
                        records[metric].append(df[metric][0])
                    
            df = pd.DataFrame(records)
            
            os.makedirs(f'{summary_dir}/{workload}', exist_ok=True)
            df.to_csv(f'{summary_dir}/{workload}/k{high_k}v{high_v}_k{low_k}v{low_v}.csv')

