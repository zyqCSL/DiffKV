export PYTHONPATH=/home/zhangyanqi/Projects/eLLM_sparsity_profile:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 RAY_DEDUP_LOGS=0 \
    python3 /home/zhangyanqi/Projects/eLLM_sparsity_profile/param_tuning/run_longbench_v2_emulate.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --load-format safetensors \
        --download-dir /data1/huggingface \
        --enforce-eager \
        --dtype float16 \
        --kv-buffer-size 64 \
        --kbits-high 8 \
        --vbits-high 8 \
        --kbits-low 8 \
        --vbits-low 8 \
        --kv-prune-thresh 1.0 \
        --kv-quant-thresh 1.0 \
        --kv-prune-ratio 0.1 \
        --kv-quant-ratio 0.1 \
        --gpu-memory-utilization 0.9 \
        --max-num-batched-tokens 131072 \
        --max-paddings 128 \
        --tensor-parallel-size 1 \
        --prompt-limit 3500 \
        --max-num-seqs 8 \
        --log-path ../tmp/tmp_debug/debug_longbench_v2 \
        --indices-csv ../logs/emulate_5rounds/llama3.1-8b/longbench_v2/k8v8_k8v8/target_10_buffer_64/attn_0-0_p1000_q1000/round_0/_sample_indices_0.csv \
        --attn-prune-thresh 0.0