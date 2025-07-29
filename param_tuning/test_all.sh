export PYTHONPATH=/home/zhangyanqi/git_repos/DiffKV:$PYTHONPATH
# # ******** gsm8k 88-88
# python3 _eval_qa_correct.py --model qwen2 --dataset gsm8k --model-gen 2 --model-size 7  --log-path ../logs/per_token_thresh/qwen25-7b --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.04 --kv-quant-thresh 0.04   --kv-buffer 64 --rounds 1
# # # ******** minerva_math 88-88
# python3 _eval_qa_correct.py --model qwen2 --dataset minerva_math --model-gen 2 --model-size 7  --log-path ../logs/per_token_thresh/qwen25-7b --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.04 --kv-quant-thresh 0.04   --kv-buffer 64 --rounds 1
# # # ******** humaneval
# python3 _eval_codegen.py --model qwen2 --dataset humaneval --model-gen 2 --model-size 7  --log-path ../logs/per_token_thresh/qwen25-7b --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.04 --kv-quant-thresh 0.04  --kv-buffer 64 --rounds 1
# # ******** mbpp_plus
# python3 _eval_codegen.py --model qwen2 --dataset mbpp_plus --model-gen 2 --model-size 7  --log-path ../logs/per_token_thresh/qwen25-7b --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.04  --kv-quant-thresh 0.04  --kv-buffer 64 --rounds 1
# # ******** mmlu 84-42
# python3 _eval_qa_correct.py --model qwen2 --dataset mmlu_cot --model-gen 2 --model-size 7  --log-path ../logs/per_token_thresh/qwen25-7b --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.04  --kv-quant-thresh 0.04  --kv-buffer 64 --rounds 1
# ******** mmlu-pro 84-42
# python3 _eval_qa_correct.py --model qwen2 --dataset mmlu_pro_cot --model-gen 2 --model-size 7  --log-path ../logs/per_token_thresh/qwen25-7b --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.04  --kv-quant-thresh 0.04  --kv-buffer 64 --rounds 1
# ******** aime24
python3 _eval_qa_correct.py --model qwq --dataset aime --model-gen 2 --model-size 32  --log-path ../logs/per_token_thresh/qwq-32b --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.0 --kv-quant-thresh 3.0 --kv-buffer 64 --rounds 1
# ******** gptq
python3 _eval_qa_correct.py --model qwq --dataset gpqa --model-gen 2 --model-size 32  --log-path ../logs/per_token_thresh/qwq-32b --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.0 --kv-quant-thresh 3.0 --kv-buffer 64 --rounds 1