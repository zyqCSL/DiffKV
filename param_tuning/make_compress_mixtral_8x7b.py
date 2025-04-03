MODEL_GEN = 1
MODEL_SIZE = 56

ENV = "export PYTHONPATH=/home/zhangyanqi/Projects/eLLM:$PYTHONPATH"

ROUNDS = 5
MODEL = 'mixtral'

COMMON_TEMPLATE = "--model {} \
--model-gen {} \
--model-size {}  \
--log-path ../logs/param_tune_{}rounds/mixtral-8x7b \
--kbits-high {} \
--vbits-high {} \
--kbits-low {} \
--vbits-low {} \
--kv-prune-thresh {} \
--kv-quant-thresh {} \
--kv-buffer {} \
--target-mem-util {} \
--rounds {}"

CNN_TEMPLATE = "python3 eval_summarize.py \
--dataset cnn_daily " + COMMON_TEMPLATE

SQUAD_TEMPLATE = "python3 eval_qa_rogue.py \
--dataset squad " + COMMON_TEMPLATE

GSM8K_TEMPLATE = "python3 eval_qa_correct.py \
--dataset gsm8k " + COMMON_TEMPLATE

GSM8K_ZERO_SHOT_TEMPLATE = GSM8K_TEMPLATE + ' --zero-shot'

MMLU_TEMPLATE = "python3 eval_qa_correct.py \
--dataset mmlu_cot " + COMMON_TEMPLATE

HUMANEVAL_TEMPLATE = "python3 eval_codegen.py \
--dataset humaneval " + COMMON_TEMPLATE

def compose_cmd(
    template: str,
    high_k: int,
    high_v: int,
    low_k: int,
    low_v: int,
    kv_prune_thresh: float,
    kv_quant_thresh: float,
    target_util: float,
    kv_buffer: int,
    rounds: int,
):
    return template.format(
        MODEL,
        MODEL_GEN, 
        MODEL_SIZE, 
        ROUNDS, 
        high_k, high_v, low_k, low_v,
        kv_prune_thresh, kv_quant_thresh, kv_buffer,
        target_util,
        rounds,
    )

cmd_texts = [ENV]
# 88, 84, 44, 42 baselines
baseline_quant_configs = [
    (8, 8, 8, 8),
    (8, 4, 8, 4),
    (4, 4, 4, 4),
    (4, 2, 4, 2),
]

kv_buffer = 32

#----------- gsm8k -----------#
quant_configs = [
    [8, 4, 4, 2],
    # [4, 4, 4, 2],
    [4, 2, 4, 1],
]

prune_threshes = [1.0]
quant_threshes = [0.5, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]
zero_shot_quant_threshes = [0.99, 0.999, 0.9995, 0.9999]

for high_k, high_v, low_k, low_v in baseline_quant_configs:
    cmd_texts.append(f"# baseline gsm8k {high_k}{high_v}-{low_k}{low_v}")
    cmd_texts.append(
        compose_cmd(
            template=GSM8K_TEMPLATE,
            high_k=high_k,
            high_v=high_v,
            low_k=low_k,
            low_v=low_v,
            kv_prune_thresh=1.0,
            kv_quant_thresh=1.0,
            target_util=0.1,
            kv_buffer=kv_buffer,
            rounds=5,
        )
    )
    
    cmd_texts.append(f"# baseline gsm8k-zeroshot {high_k}{high_v}-{low_k}{low_v}")
    cmd_texts.append(
        compose_cmd(
            template=GSM8K_ZERO_SHOT_TEMPLATE,
            high_k=high_k,
            high_v=high_v,
            low_k=low_k,
            low_v=low_v,
            kv_prune_thresh=1.0,
            kv_quant_thresh=1.0,
            target_util=0.1,
            kv_buffer=kv_buffer,
            rounds=5,
        )
    )


for high_k, high_v, low_k, low_v in quant_configs:
    cmd_texts.append(f"# gsm8k {high_k}{high_v}-{low_k}{low_v}")
    for prune_thresh in prune_threshes:
        for quant_thresh in quant_threshes:
            cmd_texts.append(
                compose_cmd(
                    template=GSM8K_TEMPLATE,
                    high_k=high_k,
                    high_v=high_v,
                    low_k=low_k,
                    low_v=low_v,
                    kv_prune_thresh=prune_thresh,
                    kv_quant_thresh=quant_thresh,
                    target_util=0.1,
                    kv_buffer=kv_buffer,
                    rounds=5,
                )
            )
            # cmd_texts.append(
            #     compose_cmd(
            #         template=GSM8K_ZERO_SHOT_TEMPLATE,
            #         high_k=high_k,
            #         high_v=high_v,
            #         low_k=low_k,
            #         low_v=low_v,
            #         kv_prune_thresh=prune_thresh,
            #         kv_quant_thresh=quant_thresh,
            #         target_util=0.1,
            #         kv_buffer=kv_buffer
            #     )
            # )
        
        # additional quant thresh for zero-shot
        for quant_thresh in zero_shot_quant_threshes:
            cmd_texts.append(
                compose_cmd(
                    template=GSM8K_ZERO_SHOT_TEMPLATE,
                    high_k=high_k,
                    high_v=high_v,
                    low_k=low_k,
                    low_v=low_v,
                    kv_prune_thresh=prune_thresh,
                    kv_quant_thresh=quant_thresh,
                    target_util=0.1,
                    kv_buffer=kv_buffer,
                    rounds=5,
                )
            )


#----------- humaneval -----------#
quant_configs = [
    [8, 4, 4, 2],
    # [4, 4, 4, 2],
    [4, 2, 4, 1],
]

prune_threshes = [1.0]
quant_threshes = [0.8, 0.9, 0.95, 0.99, 0.999, 0.9995, 0.9999]

for high_k, high_v, low_k, low_v in baseline_quant_configs:
    cmd_texts.append(f"# baseline humaneval {high_k}{high_v}-{low_k}{low_v}")
    cmd_texts.append(
        compose_cmd(
            template=HUMANEVAL_TEMPLATE,
            high_k=high_k,
            high_v=high_v,
            low_k=low_k,
            low_v=low_v,
            kv_prune_thresh=1.0,
            kv_quant_thresh=1.0,
            target_util=0.1,
            kv_buffer=kv_buffer,
            rounds=5,
        )
    )

for high_k, high_v, low_k, low_v in quant_configs:
    cmd_texts.append(f"# humaneval {high_k}{high_v}-{low_k}{low_v}")
    for prune_thresh in prune_threshes:
        for quant_thresh in quant_threshes:
            cmd_texts.append(
                compose_cmd(
                    template=HUMANEVAL_TEMPLATE,
                    high_k=high_k,
                    high_v=high_v,
                    low_k=low_k,
                    low_v=low_v,
                    kv_prune_thresh=prune_thresh,
                    kv_quant_thresh=quant_thresh,
                    target_util=0.1,
                    kv_buffer=kv_buffer,
                    rounds=5,
                )
            )


#----------- mmlu -----------#
quant_configs = [
    [8, 4, 4, 2],
    # [4, 4, 4, 2],
    [4, 2, 4, 1],
]

prune_threshes = [1.0]
quant_threshes = [0.5, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]

for high_k, high_v, low_k, low_v in baseline_quant_configs:
    cmd_texts.append(f"# baseline mmlu {high_k}{high_v}-{low_k}{low_v}")
    cmd_texts.append(
        compose_cmd(
            template=MMLU_TEMPLATE,
            high_k=high_k,
            high_v=high_v,
            low_k=low_k,
            low_v=low_v,
            kv_prune_thresh=1.0,
            kv_quant_thresh=1.0,
            target_util=0.1,
            kv_buffer=kv_buffer,
            rounds=3,
        )
    )

for high_k, high_v, low_k, low_v in quant_configs:
    cmd_texts.append(f"# mmlu {high_k}{high_v}-{low_k}{low_v}")
    for prune_thresh in prune_threshes:
        for quant_thresh in quant_threshes:
            cmd_texts.append(
                compose_cmd(
                    template=MMLU_TEMPLATE,
                    high_k=high_k,
                    high_v=high_v,
                    low_k=low_k,
                    low_v=low_v,
                    kv_prune_thresh=prune_thresh,
                    kv_quant_thresh=quant_thresh,
                    target_util=0.1,
                    kv_buffer=kv_buffer,
                    rounds=3,
                )
            )


#----------- squad -----------#
quant_configs = [
    [8, 4, 4, 2],
    [4, 2, 4, 1],
    # [4, 4, 4, 2],
]

prune_threshes = [1.0]
quant_threshes = [0.5, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]

for high_k, high_v, low_k, low_v in baseline_quant_configs:
    cmd_texts.append(f"# baseline squad {high_k}{high_v}-{low_k}{low_v}")
    cmd_texts.append(
        compose_cmd(
            template=SQUAD_TEMPLATE,
            high_k=high_k,
            high_v=high_v,
            low_k=low_k,
            low_v=low_v,
            kv_prune_thresh=1.0,
            kv_quant_thresh=1.0,
            target_util=0.1,
            kv_buffer=kv_buffer,
            rounds=3,
        )
    )

for high_k, high_v, low_k, low_v in quant_configs:
    cmd_texts.append(f"# squad {high_k}{high_v}-{low_k}{low_v}")
    for prune_thresh in prune_threshes:
        for quant_thresh in quant_threshes:
            cmd_texts.append(
                compose_cmd(
                    template=SQUAD_TEMPLATE,
                    high_k=high_k,
                    high_v=high_v,
                    low_k=low_k,
                    low_v=low_v,
                    kv_prune_thresh=prune_thresh,
                    kv_quant_thresh=quant_thresh,
                    target_util=0.1,
                    kv_buffer=kv_buffer,
                    rounds=3,
                )
            )


#----------- cnn-dailymail -----------#
quant_configs = [
    [8, 4, 4, 2],
    # [4, 4, 4, 2],
    [4, 2, 4, 1],
]

prune_threshes = [1.0]
quant_threshes = [0.5, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]

for high_k, high_v, low_k, low_v in baseline_quant_configs:
    cmd_texts.append(f"# baseline cnn-dailymail {high_k}{high_v}-{low_k}{low_v}")
    cmd_texts.append(
        compose_cmd(
            template=CNN_TEMPLATE,
            high_k=high_k,
            high_v=high_v,
            low_k=low_k,
            low_v=low_v,
            kv_prune_thresh=1.0,
            kv_quant_thresh=1.0,
            target_util=0.1,
            kv_buffer=kv_buffer,
            rounds=3,
        )
    )

for high_k, high_v, low_k, low_v in quant_configs:
    cmd_texts.append(f"# cnn-dailymail {high_k}{high_v}-{low_k}{low_v}")
    for prune_thresh in prune_threshes:
        for quant_thresh in quant_threshes:
            cmd_texts.append(
                compose_cmd(
                    template=CNN_TEMPLATE,
                    high_k=high_k,
                    high_v=high_v,
                    low_k=low_k,
                    low_v=low_v,
                    kv_prune_thresh=prune_thresh,
                    kv_quant_thresh=quant_thresh,
                    target_util=0.1,
                    kv_buffer=kv_buffer,
                    rounds=3,
                )
            )


with open(f'run_compress_mixtral_8x7b.sh', 'w+') as f:
    f.write('\n'.join(cmd_texts))
    
    


