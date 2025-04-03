export PYTHONPATH=/home/zhangyanqi/Projects/eLLM:$PYTHONPATH
CUDA_VISIBLE_DEVICES=2,3 \
    RAY_DEDUP_LOGS=0 python3 debug_qwen2.py