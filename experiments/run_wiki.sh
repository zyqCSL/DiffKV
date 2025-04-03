BUFFER=$2
THRESH=$3
CUDA_VISIBLE_DEVICES=$1 RAY_DEDUP_LOGS=0 python3 run_wiki_text.py --model meta-llama/Llama-2-7b-chat-hf \
    --download-dir /data/huggingface \
    --enforce-eager \
    --block-size 16 \
    --kv-buffer-size $BUFFER \
    --kv-score-thresh $THRESH \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 8192 \
    --tensor-parallel-size 1 \
    --max-num-seqs 10