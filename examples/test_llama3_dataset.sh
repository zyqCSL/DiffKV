CUDA_VISIBLE_DEVICES=0 RAY_DEDUP_LOGS=0 python3 my_llm_dataset_example.py --model meta-llama/Meta-Llama-3-8B-Instruct \
    --download-dir /data1/huggingface \
    --dtype float16 \
    --enforce-eager \
    --kv-buffer-size 8192 \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 8192 \
    --tensor-parallel-size 1