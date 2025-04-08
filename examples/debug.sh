export PYTHONPATH=/home/zhangyanqi/git_repos/DiffKV:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0,1 \
    RAY_DEDUP_LOGS=0 python3 debug.py
# CUDA_LAUNCH_BLOCKING=1 \
#     CUDA_VISIBLE_DEVICES=0 \
#     RAY_DEDUP_LOGS=0 python3 debug_generate_example.py