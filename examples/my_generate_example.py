import torch
torch.autograd.set_detect_anomaly(True)

from vllm import LLM, SamplingParams

# [kbits_high, vbits_high, kbits_low, vbits_low]
QUANT_CONFIG = [4, 4, 4, 2]
# [prune_thresh, quant_thresh, prune_ratio, quant_ratio]
COMPRESS_CONFIG = [0.0, 0.1, 0.0, 0.0]
KV_BUFFER = 32

# # [kbits_high, vbits_high, kbits_low, vbits_low]
# QUANT_CONFIG = [8, 8]
# # [prune_thresh, quant_thresh, prune_ratio, quant_ratio]
# COMPRESS_CONFIG = [1.0, 1.0, 1.0, 1.0]
# KV_BUFFER = 8192


# model = LLM('meta-llama/Llama-2-7b-chat-hf',
#             gpu_memory_utilization=0.6,
#             kv_buffer_size=KV_BUFFER,
#             download_dir='/data/huggingface',
#             enforce_eager=True)

model = LLM('meta-llama/Meta-Llama-3-8B-Instruct',
            gpu_memory_utilization=0.6,
            kv_buffer_size=KV_BUFFER,
            download_dir='/data1/huggingface',
            enforce_eager=True,
            dtype='float16')

sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)  # greedy sampling

# gives wrong answer in the 1st batch
prompts = [
    "A robot may not injure a human being",
    # "To be or not to be,",
    # "What is the meaning of life?",
    # "It is only with the heart that one can see rightly",
    # "Which American-born Sinclair won the Nobel Prize for Literature in 1930?",
    # "Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , " \
    # "commonly referred to as Valkyria Chronicles III outside Japan , " \
    # "is a tactical role playing video game developed by Sega and Media.Vision for the PlayStation Portable .",
    # "Are power and wealth important in life?",
    # "How to recover from a breakup?",
    # "What is the meaning of life?",
]

# # different results when batch_size = 1 and 8
# prompts = [
#     "Is power and wealth important in life?",
#     # "Are power and wealth important in life?",
#     # # "What is the meaning of life?",
#     # "Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , " \
#     # "commonly referred to as Valkyria Chronicles III outside Japan , " \
#     # "is a tactical role playing video game developed by Sega and Media.Vision for the PlayStation Portable .",
# ]

TESTS = 1
# TESTS = 4

for _ in range(TESTS):
    # outputs = model.generate(prompts * 10, sampling_params)
    # outputs = model.generate(prompts * 8, sampling_params)
    outputs = model.generate(prompts, sampling_params, QUANT_CONFIG, COMPRESS_CONFIG)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")