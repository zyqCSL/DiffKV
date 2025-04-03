export PYTHONPATH=/home/zhangyanqi/Projects/eLLM:$PYTHONPATH
CUDA_VISIBLE_DEVICES=6 \
    RAY_DEDUP_LOGS=0 python3 debug_generate_example.py
# CUDA_LAUNCH_BLOCKING=1 \
#     CUDA_VISIBLE_DEVICES=0 \
#     RAY_DEDUP_LOGS=0 python3 debug_generate_example.py