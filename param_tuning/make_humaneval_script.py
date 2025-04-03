MODEL_GEN = 3
MODEL_SIZE = 8


ENV = "export PYTHONPATH=/home/zhangyanqi/Projects/eLLM_dev:$PYTHONPATH"

TEMPLATE = "python3 eval_codegen.py \
--dataset humaneval \
--model-gen {} \
--model-size {}  \
--log-path ../logs/param_tune/llama{}-{}b \
--kbits-high {} \
--vbits-high {} \
--kbits-low {} \
--vbits-low {} \
--kv-prune-thresh {} \
--kv-quant-thresh {} \
--kv-buffer {} \
--target-mem-util {}"
    

def compose_cmd(
    high_k: int,
    high_v: int,
    low_k: int,
    low_v: int,
    kv_prune_thresh: float,
    kv_quant_thresh: float,
    target_util: float,
    kv_buffer: int,
):
    template = TEMPLATE
    return template.format(
        MODEL_GEN, MODEL_SIZE, MODEL_GEN, MODEL_SIZE,
        high_k, high_v, low_k, low_v,
        kv_prune_thresh, kv_quant_thresh, kv_buffer,
        target_util
    )

cmd_texts = [ENV]
# 88, 84, 44, 42 baselines
baseline_quant_configs = [
    (8, 8, 8, 8),
    (8, 4, 8, 4),
    (4, 4, 4, 4),
    (4, 2, 4, 2),
]

if MODEL_GEN == 2:
    kv_buffer = 4096
else:
    kv_buffer = 8192

for high_k, high_v, low_k, low_v in baseline_quant_configs:
    cmd_texts.append(
        compose_cmd(
            high_k=high_k,
            high_v=high_v,
            low_k=low_k,
            low_v=low_v,
            kv_prune_thresh=1.0,
            kv_quant_thresh=1.0,
            target_util=1.0,
            kv_buffer=kv_buffer,
        )
    )

with open(f'run_humaneval_quant_base_llama{MODEL_GEN}_{MODEL_SIZE}b.sh', 'w+') as f:
    f.write('\n'.join(cmd_texts))
    
    


