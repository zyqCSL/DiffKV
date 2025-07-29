export PYTHONPATH=/home/zhangyanqi/git_repos/DiffKV:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0,1 \
    RAY_DEDUP_LOGS=0 python3 debug_qwen3_moe.py