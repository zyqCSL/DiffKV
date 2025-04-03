export PYTHONPATH=/home/zhangyanqi/Projects/eLLM:$PYTHONPATH
python3 eval_summarize.py \
    --model-gen 2 \
    --model-size 7  \
    --log-path ../logs/param_tune/llama2-7b \
    --kbits-high 4 \
    --vbits-high 4 \
    --kbits-low 4 \
    --vbits-low 2 \
    --kv-prune-thresh 0.99 \
    --kv-quant-thresh 0.95 \
    --target-mem-util 0.5