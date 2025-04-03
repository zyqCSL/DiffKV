export PYTHONPATH=/home/zhangyanqi/Projects/eLLM:$PYTHONPATH
RAY_DEDUP_LOGS=4,5,6,7 \
    python3 /home/zhangyanqi/Projects/eLLM/param_tuning/run_mmlu_cot.py \
    --model meta-llama/Llama-2-70b-chat-hf \
    --download-dir /data1/huggingface \
    --enforce-eager \
    --dtype float16 \
    --kv-buffer-size 32 \
    --kbits-high 4 \
    --vbits-high 2 \
    --kbits-low 4 \
    --vbits-low 1 \
    --kv-prune-thresh 1.0 \
    --kv-quant-thresh 0.5 \
    --kv-prune-ratio 0.1 \
    --kv-quant-ratio 0.1 \
    --gpu-memory-utilization 0.75 \
    --max-num-batched-tokens 40960 \
    --max-paddings 4096 \
    --tensor-parallel-size 4 \
    --prompt-limit 3500 \
    --max-num-seqs 256 \
    --log-path ../tmp/tmp_debug/42_41_kernel_opt \
    --data-label test \
    --indices-csv ../logs/param_tune_5rounds/llama3-70b/mmlu_cot/k4v2_k4v1/target_10_buffer_32/p1000_q500/round_0/sample_indices_0.csv