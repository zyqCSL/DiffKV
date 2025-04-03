export PYTHONPATH=/home/zhangyanqi/Projects/eLLM:$PYTHONPATH
python3 eval_qa_correct.py \
    --dataset mmlu_cot \
    --model-gen 2 \
    --model-size 7  \
    --log-path ../logs/param_tune/llama2-7b \
    --kbits-high 8 \
    --vbits-high 8 \
    --kbits-low 8 \
    --vbits-low 8 \
    --kv-prune-thresh 1.0 \
    --kv-quant-thresh 1.0 \
    --target-mem-util 1.0