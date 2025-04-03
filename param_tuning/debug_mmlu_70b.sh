export PYTHONPATH=/home/zhangyanqi/Projects/eLLM_dev:$PYTHONPATH
python3 eval_qa_correct.py \
    --dataset mmlu_cot \
    --model-gen 3 \
    --model-size 70  \
    --log-path ../logs/param_tune/llama3-70b \
    --kbits-high 8 \
    --vbits-high 4 \
    --kbits-low 8 \
    --vbits-low 2 \
    --kv-prune-thresh 1.0 \
    --kv-quant-thresh 0.95 \
    --target-mem-util 0.5