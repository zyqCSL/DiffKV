# import torch
# torch.autograd.set_detect_anomaly(True)

import os
os.environ['CURL_CA_BUNDLE'] = ''

from vllm import LLM, SamplingParams

model = LLM(
    # 'meta-llama/Llama-2-7b-chat-hf',
    # 'meta-llama/Meta-Llama-3-8B-Instruct',
    # 'meta-llama/Llama-2-7b-chat-hf',
    # 'mistralai/Mistral-7B-Instruct-v0.1',
    # 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    # '/data1/modelscope/Qwen3-14B',
    '/data1/modelscope/Qwen2.5-7B-Instruct',
    dtype='float16',
    gpu_memory_utilization=0.8,
    load_format='safetensors',
    kv_buffer_size=64,
    tensor_parallel_size=1,
    download_dir='/data1/huggingface',
    enforce_eager=True)

sampling_params = SamplingParams(
    temperature=0.0, max_tokens=512, stop=["</s>"])  # greedy sampling

# gives wrong answer in the 1st batch
prompts = [
    "A robot may not injure a human being",
    "To be or not to be,",
    "What is the meaning of life?",
    "It is only with the heart that one can see rightly",
    "Which American-born Sinclair won the Nobel Prize for Literature in 1930?",
    "Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , " \
    "commonly referred to as Valkyria Chronicles III outside Japan , " \
    "is a tactical role playing video game developed by Sega and Media.Vision for the PlayStation Portable .",
    "Are power and wealth important in life?",
    "How to recover from a breakup?",
    "What is the meaning of life?",
] * 1

# # different results when batch_size = 1 and 8
# prompts = [
#     "Is power and wealth important in life?",
#     # "Are power and wealth important in life?",
#     # # "What is the meaning of life?",
#     # "Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , " \
#     # "commonly referred to as Valkyria Chronicles III outside Japan , " \
#     # "is a tactical role playing video game developed by Sega and Media.Vision for the PlayStation Portable .",
# ]

# [kbits_high, vbits_high, kbits_low, vbits_low]
quant_configs = [
    # [8, 8, 8, 4],
    # [8, 4, 4, 2],
    # [8, 8, 4, 4],
    # [8, 8],
    [8, 4],
    # [4, 4, 4, 2],
    # [4, 4],
    # [4, 2],
    # [4, 1],
]

# [prune_ratio (alpha_low), quant_ratio(alpha_high)]
compress_config = [0.02, 0.02]
# compress_config = [0.0, 0.0]

TESTS = 1
for _ in range(TESTS):
    for quant_config in quant_configs:
        print(f"Quant config: {quant_config}")
        if len(quant_config) == 2:
            _compress_config = [0.0, 0.0]
        else:
            _compress_config = compress_config
        outputs = model.generate(prompts, sampling_params, 
                                 quant_config, _compress_config)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")