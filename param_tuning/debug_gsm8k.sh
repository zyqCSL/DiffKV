export PYTHONPATH=/home/zhangyanqi/Projects/eLLM:$PYTHONPATH
# 8
# python3 eval_qa_correct.py \
#     --dataset gsm8k \
#     --model-gen 3 \
#     --model-size 70  \
#     --log-path ../logs/param_tune/llama3-70b \
#     --kbits-high 8 \
#     --vbits-high 4 \
#     --kbits-low 8 \
#     --vbits-low 2 \
#     --kv-prune-thresh 1.0 \
#     --kv-quant-thresh 0.95 \
#     --target-mem-util 0.1 \
#     --zero-shot
python3 eval_qa_correct.py \
    --dataset gsm8k \
    --model-gen 3 \
    --model-size 70  \
    --log-path ../logs/param_tune/llama3-70b \
    --kbits-high 8 \
    --vbits-high 4 \
    --kbits-low 8 \
    --vbits-low 2 \
    --kv-prune-thresh 1.0 \
    --kv-quant-thresh 0.95 \
    --target-mem-util 0.1 \
# # 84
# python3 eval_qa_correct.py \
#     --dataset gsm8k \
#     --model-gen 3 \
#     --model-size 70  \
#     --log-path ../logs/param_tune/llama3-70b \
#     --kbits-high 8 \
#     --vbits-high 4 \
#     --kbits-low 8 \
#     --vbits-low 4 \
#     --kv-prune-thresh 1.0 \
#     --kv-quant-thresh 1.0 \
#     --target-mem-util 1.0 \
#     --zero-shot
# python3 eval_qa_correct.py \
#     --dataset gsm8k \
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
# python3 eval_qa_correct.py \
#     --dataset gsm8k \
#     --model-gen 3 \
#     --model-size 70  \
#     --log-path ../logs/param_tune/llama3-70b \
#     --kbits-high 4 \
#     --vbits-high 4 \
#     --kbits-low 4 \
#     --vbits-low 2 \
#     --kv-prune-thresh 1.0 \
#     --kv-quant-thresh 0.95 \
#     --target-mem-util 0.5 \
#     --zero-shot
# python3 eval_qa_correct.py \
#     --dataset gsm8k \
#     --model-gen 3 \
#     --model-size 70  \
#     --log-path ../logs/param_tune/llama3-70b \
#     --kbits-high 4 \
#     --vbits-high 4 \
#     --kbits-low 4 \
#     --vbits-low 2 \
#     --kv-prune-thresh 1.0 \
#     --kv-quant-thresh 0.95 \
#     --target-mem-util 0.5
# python3 eval_qa_correct.py \
#     --dataset gsm8k \
#     --model-gen 3 \
#     --model-size 70  \
#     --log-path ../logs/param_tune/llama3-70b \
#     --kbits-high 4 \
#     --vbits-high 4 \
#     --kbits-low 4 \
#     --vbits-low 2 \
#     --kv-prune-thresh 1.0 \
#     --kv-quant-thresh 0.9 \
#     --target-mem-util 0.5 \
#     --zero-shot
# python3 eval_qa_correct.py \
#     --dataset gsm8k \
#     --model-gen 3 \
#     --model-size 70  \
#     --log-path ../logs/param_tune/llama3-70b \
#     --kbits-high 4 \
#     --vbits-high 4 \
#     --kbits-low 4 \
#     --vbits-low 2 \
#     --kv-prune-thresh 1.0 \
#     --kv-quant-thresh 0.9 \
#     --target-mem-util 0.5
# # 42
# python3 eval_qa_correct.py \
#     --dataset gsm8k \
#     --model-gen 3 \
#     --model-size 70  \
#     --log-path ../logs/param_tune/llama3-70b \
#     --kbits-high 4 \
#     --vbits-high 2 \
#     --kbits-low 4 \
#     --vbits-low 2 \
#     --kv-prune-thresh 1.0 \
#     --kv-quant-thresh 1.0 \
#     --target-mem-util 1.0 \
#     --zero-shot
# python3 eval_qa_correct.py \
#     --dataset gsm8k \
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
# # 42-41
# python3 eval_qa_correct.py \
#     --dataset gsm8k \
#     --model-gen 3 \
#     --model-size 70  \
#     --log-path ../logs/param_tune/llama3-70b \
#     --kbits-high 4 \
#     --vbits-high 2 \
#     --kbits-low 4 \
#     --vbits-low 1 \
#     --kv-prune-thresh 1.0 \
#     --kv-quant-thresh 0.975 \
#     --target-mem-util 0.5 \
#     --zero-shot
# python3 eval_qa_correct.py \
#     --dataset gsm8k \
#     --model-gen 3 \
#     --model-size 70  \
#     --log-path ../logs/param_tune/llama3-70b \
#     --kbits-high 4 \
#     --vbits-high 2 \
#     --kbits-low 4 \
#     --vbits-low 1 \
#     --kv-prune-thresh 1.0 \
#     --kv-quant-thresh 0.975 \
#     --target-mem-util 0.5