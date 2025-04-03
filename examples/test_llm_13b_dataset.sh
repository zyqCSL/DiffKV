CUDA_VISIBLE_DEVICES=0,1 RAY_DEDUP_LOGS=0 python3 my_llm_dataset_example.py \
    --model meta-llama/Llama-2-13b-chat-hf \
    --download-dir /data1/huggingface \
    --enforce-eager \
    --kv-buffer-size 4096 \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 8192 \
    --tensor-parallel-size 2