export PYTHONPATH=/home/zhangyanqi/git_repos/DiffKV:$PYTHONPATH
# # gsm8k 88-88
CUDA_VISIBLE_DEVICES=2,3 RAY_DEDUP_LOGS=0 \
    python3 /home/zhangyanqi/Projects/eLLM/param_tuning/run_minerva_math.py \
        --model /data1/modelscope/QwQ-32B \
        --load-format safetensors \
        --enforce-eager \
        --dtype float16 \
        --kv-buffer-size 64 \
        --kbits-high 8 \
        --vbits-high 4 \
        --kbits-low 4 \
        --vbits-low 2 \
        --kv-prune-thresh 0.0 \
        --kv-quant-thresh 0.1 \
        --gpu-memory-utilization 0.75 \
        --max-num-batched-tokens 40960 \
        --max-paddings 2048 \
        --tensor-parallel-size 2 \
        --prompt-limit 40960 \
        --max-num-seqs 16 \
        --log-path ../logs/per_token_thresh/qwq-32b/minerva_math/k8v8_k8v8/target_10_buffer_64/p0_q100/round_0/eval_0 \
        --indices-csv ../logs/per_token_thresh/qwq-32b/minerva_math/k8v4_k4v2/target_10_buffer_64/p0_q100/round_0/sample_indices_0.csv \
        --data-label test