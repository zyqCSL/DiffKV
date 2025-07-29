# import torch
# torch.autograd.set_detect_anomaly(True)

import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from vllm import LLM, SamplingParams

# MAX_TOKENS = 32 * 1024
MAX_TOKENS = 256

model = LLM(
    # '/data1/modelscope/Qwen2.5-7B-Instruct',
    # '/data1/modelscope/Qwen2.5-32B-Instruct',
    # '/data1/modelscope/Qwen2.5-72B-Instruct',
    # '/data1/modelscope/QwQ-32B',
    # '/data1/modelscope/Qwen3-14B',
    '/data1/modelscope/Qwen3-8B-AWQ',
    # '/data1/modelscope/Qwen3-8B-FP8',
    # '/data1/modelscope/Qwen3-30B-A3B-AWQ',
    dtype='float16',
    gpu_memory_utilization=0.8,
    load_format='safetensors',
    kv_buffer_size=64,
    tensor_parallel_size=1,
    enforce_eager=True)

sampling_params = SamplingParams(
    temperature=0.6, top_p=0.95, min_p=0.0, 
    max_tokens=MAX_TOKENS, 
    # stop=["<\think>"],
    )

# gives wrong answer in the 1st batch
problems = [
    r"Let $b \geq 2$ be an integer. Call a positive integer $n$ $b$\textit{-eautiful} if it has exactly two digits when expressed in base $b$, and these two digits sum to $\sqrt{n}$. For example, $81$ is $13$-eautiful because $81=\underline{6}\underline{3}_{13}$ and $6+3=\sqrt{81}$. Find the least integer $b \geq 2$ for which there are more than ten $b$-eautiful integers.",
]

thinking_prompt = r" Please reason step by step, and put your final answer within \boxed{}. "
prompts = []

for p in problems:
    prompts.append(p + thinking_prompt + "<think>\n")



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
    [8, 4, 4, 2],
    # [8, 8, 4, 4],
    # [8, 8],
    # [8, 4],
    # [4, 4, 4, 2],
    # [4, 4],
    # [4, 2],
    # [4, 1],
]

# [k_ngroups_high, v_ngroups_high, k_ngroups_low, v_ngroups_low]
quant_groups = [
    [1, 2, 2, 4],
]


# [alpha_prune, alpha_quant]
compress_config = [0.0, 1.0]


# # [kbits_high, vbits_high, kbits_low, vbits_low]
# quant_configs = [
#     # [8, 8, 8, 4],
#     # [8, 4, 4, 2],
#     # [8, 8, 4, 4],
#     [8, 8],
#     # [8, 4],
#     # [4, 4, 4, 2],
#     # [4, 4],
#     # [4, 2],
#     # [4, 1],
# ]

# # [k_ngroups_high, v_ngroups_high, k_ngroups_low, v_ngroups_low]
# quant_groups = [
#     [1, 1],
# ]
# # [alpha_prune, alpha_quant]
# compress_config = [0.0, 0.0]

TESTS = 1
for _ in range(TESTS):
    for quant_config, quant_group in zip(quant_configs, quant_groups):
        print(f"Quant config: {quant_config}")
        if len(quant_config) == 2:
            _compress_config = [0.0, 0.0]
        else:
            _compress_config = compress_config
        outputs = model.generate(prompts, sampling_params, 
                                 quant_config, quant_group,
                                 _compress_config)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")