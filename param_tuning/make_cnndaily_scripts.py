MODEL_GEN = 2
MODEL_SIZE = 7


ENV = "export PYTHONPATH=/home/zhangyanqi/Projects/eLLM_dev:$PYTHONPATH"

TEMPLATE = "python3 eval_summarize.py \
--dataset cnn_daily \
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

# for high_k, high_v, low_k, low_v in baseline_quant_configs:
#     cmd_texts.append(
#         compose_cmd(
#             high_k=high_k,
#             high_v=high_v,
#             low_k=low_k,
#             low_v=low_v,
#             kv_prune_thresh=1.0,
#             kv_quant_thresh=1.0,
#             target_util=1.0,
#             kv_buffer=kv_buffer
#         )
#     )

high_k = 4
high_v = 2
low_k = 4
low_v = 1

kv_buffer = 32
prune_threshes = [0.96, 0.97, 0.98, 1.0]
quant_threshes = [0.8, 0.85, 0.9, 0.95]

prune_threshes = [0.95, 0.96]
quant_threshes = [0.3, 0.5, 0.8]

for prune_thresh in prune_threshes:
    for quant_thresh in quant_threshes:
        cmd_texts.append(
            compose_cmd(
                high_k=high_k,
                high_v=high_v,
                low_k=low_k,
                low_v=low_v,
                kv_prune_thresh=prune_thresh,
                kv_quant_thresh=quant_thresh,
                target_util=0.1,
                kv_buffer=kv_buffer
            )
        )


with open(f'run_cnn_daily_compress_llama{MODEL_GEN}_{MODEL_SIZE}b.sh', 'w+') as f:
    f.write('\n'.join(cmd_texts))
    
    


