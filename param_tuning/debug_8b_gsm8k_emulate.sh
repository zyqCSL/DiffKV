export PYTHONPATH=/home/zhangyanqi/Projects/eLLM_sparsity_profile:$PYTHONPATH
RAY_DEDUP_LOGS=0 \
    python3 /home/zhangyanqi/Projects/eLLM_sparsity_profile/param_tuning/run_gsm8k_emulate.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --download-dir /data1/huggingface \
    --enforce-eager \
    --dtype float16 \
    --kv-buffer-size 64 \
    --attn-prune-thresh 0.5 \
    --kbits-high 8 \
    --vbits-high 8 \
    --kbits-low 8 \
    --vbits-low 8 \
    --kv-prune-thresh 1.0 \
    --kv-quant-thresh 1.0 \
    --kv-prune-ratio 1.0 \
    --kv-quant-ratio 1.0 \
    --gpu-memory-utilization 0.75 \
    --max-num-batched-tokens 40960 \
    --max-paddings 4096 \
    --tensor-parallel-size 1 \
    --prompt-limit 3500 \
    --max-num-seqs 256 \
    --log-path ../tmp/tmp_debug/emulate_gsm8k \
    --data-label test \
    --indices-csv ../logs/emulate_5roundss/llama3-8b/gsm8k/k8v8_k8v8/target_10_buffer_64/attn_0-5_p1000_q1000/round_0/_sample_indices_0.csv
    