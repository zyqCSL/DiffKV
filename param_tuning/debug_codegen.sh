export PYTHONPATH=/home/zhangyanqi/Projects/eLLM:$PYTHONPATH
# # 8
# python3 eval_codegen.py \
#     --dataset humaneval \
#     --model-gen 3 \
#     --model-size 70  \
#     --log-path ../logs/param_tune/llama3-70b \
#     --kbits-high 8 \
#     --vbits-high 8 \
#     --kbits-low 8 \
#     --vbits-low 8 \
#     --kv-prune-thresh 1.0 \
#     --kv-quant-thresh 1.0 \
#     --target-mem-util 1.0
# # 84
# python3 eval_codegen.py \
#     --dataset humaneval \
#     --model-gen 3 \
#     --model-size 70  \
#     --log-path ../logs/param_tune/llama3-70b \
#     --kbits-high 8 \
#     --vbits-high 4 \
#     --kbits-low 8 \
#     --vbits-low 4 \
#     --kv-prune-thresh 1.0 \
#     --kv-quant-thresh 1.0 \
#     --target-mem-util 1.0
# # 4
# python3 eval_codegen.py \
#     --dataset humaneval \
#     --model-gen 3 \
#     --model-size 70  \
#     --log-path ../logs/param_tune/llama3-70b \
#     --kbits-high 4 \
#     --vbits-high 4 \
#     --kbits-low 4 \
#     --vbits-low 4 \
#     --kv-prune-thresh 1.0 \
#     --kv-quant-thresh 1.0 \
#     --target-mem-util 1.0
# # 42
# python3 eval_codegen.py \
#     --dataset humaneval \
#     --model-gen 3 \
#     --model-size 70  \
#     --log-path ../logs/param_tune/llama3-70b \
#     --kbits-high 4 \
#     --vbits-high 2 \
#     --kbits-low 4 \
#     --vbits-low 2 \
#     --kv-prune-thresh 1.0 \
#     --kv-quant-thresh 1.0 \
#     --target-mem-util 1.0
# 4
# python3 eval_codegen.py \
#     --dataset humaneval \
#     --model-gen 3 \
#     --model-size 70  \
#     --log-path ../logs/param_tune/llama3-70b \
#     --kbits-high 4 \
#     --vbits-high 4 \
#     --kbits-low 4 \
#     --vbits-low 4 \
#     --kv-prune-thresh 1.0 \
#     --kv-quant-thresh 1.0 \
#     --target-mem-util 1.0
# 42
python3 eval_codegen.py \
    --dataset humaneval \
    --model-gen 3 \
    --model-size 70  \
    --log-path ../logs/param_tune/llama3-70b \
    --kbits-high 4 \
    --vbits-high 4 \
    --kbits-low 4 \
    --vbits-low 2 \
    --kv-prune-thresh 1.0 \
    --kv-quant-thresh 0.95 \
    --target-mem-util 0.1
python3 eval_codegen.py \
    --dataset humaneval \
    --model-gen 3 \
    --model-size 70  \
    --log-path ../logs/param_tune/llama3-70b \
    --kbits-high 4 \
    --vbits-high 4 \
    --kbits-low 4 \
    --vbits-low 2 \
    --kv-prune-thresh 1.0 \
    --kv-quant-thresh 0.9 \
    --target-mem-util 0.1
python3 eval_codegen.py \
    --dataset humaneval \
    --model-gen 3 \
    --model-size 70  \
    --log-path ../logs/param_tune/llama3-70b \
    --kbits-high 4 \
    --vbits-high 4 \
    --kbits-low 4 \
    --vbits-low 2 \
    --kv-prune-thresh 1.0 \
    --kv-quant-thresh 0.8 \
    --target-mem-util 0.1