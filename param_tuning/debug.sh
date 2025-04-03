export PYTHONPATH=/home/zhangyanqi/Projects/eLLM:$PYTHONPATH
# # gsm8k 88-88
CUDA_VISIBLE_DEVICES=4,5 RAY_DEDUP_LOGS=0 \
    python3 /home/zhangyanqi/Projects/eLLM/param_tuning/run_gsm8k.py \
        --model /data1/modelscope/Qwen2.5-32B-Instruct \
        --load-format safetensors \
        --enforce-eager \
        --dtype float16 \
        --kv-buffer-size 64 \
        --kbits-high 8 \
        --vbits-high 8 \
        --kbits-low 8 \
        --vbits-low 8 \
        --kv-prune-thresh 0.0 \
        --kv-quant-thresh 0.0 \
        --gpu-memory-utilization 0.97 \
        --max-num-batched-tokens 40960 \
        --max-paddings 2048 \
        --tensor-parallel-size 2 \
        --prompt-limit 40960 \
        --max-num-seqs 16 \
        --log-path ../logs/per_token_thresh/qwen25-32b/gsm8k/k8v8_k8v8/target_10_buffer_64/p0_q1000/round_0/eval_0 \
        --indices-csv ../logs/per_token_thresh/qwen25-32b/gsm8k/k8v8_k8v8/target_10_buffer_64/p0_q1000/round_0/sample_indices_0.csv \
        --data-label test