BUFFER=$2
THRESH=$3
COMPRESS=$4
CUDA_VISIBLE_DEVICES=$1 RAY_DEDUP_LOGS=0 python3 run_mbpp.py --model meta-llama/Llama-2-7b-chat-hf \
    --download-dir /data/huggingface \
    --enforce-eager \
    --block-size 16 \
    --kv-buffer-size $BUFFER \
    --kv-score-thresh $THRESH \
    --kv-compress-ratio $COMPRESS \
    --gpu-memory-utilization 0.85 \
    --max-num-batched-tokens 40960 \
    --max-paddings 8192 \
    --tensor-parallel-size 1 \
    --max-num-seqs 64