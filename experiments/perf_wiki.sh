CUDA_VISIBLE_DEVICES=6 RAY_DEDUP_LOGS=0 python3 run_wiki_text.py --model meta-llama/Llama-2-7b-chat-hf \
    --download-dir /data/huggingface \
    --enforce-eager \
    --block-size 16 \
    --kv-buffer-size 64 \
    --kv-score-thresh 0.999 \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 8192 \
    --tensor-parallel-size 1 \
    --max-num-seqs 128