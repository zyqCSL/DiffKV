from vllm import LLM, SamplingParams

model = LLM('meta-llama/Llama-2-7b-chat-hf',
            #'meta-llama/Llama-3.1-8B-Instruct',
            download_dir='/data1/huggingface',
            gpu_memory_utilization=0.75,
            load_format='safetensors',
            kv_buffer_size=32,
            max_paddings=4096,
            dtype='float16',
            disable_log_stats=False,
            enforce_eager=True)

sampling_params = SamplingParams(temperature=0.0, max_tokens=5)  # greedy sampling

quant_config = [8, 4, 4, 2]
compress_config = [0.0, 1.0]

prompts = [
    "Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , " \
    "commonly referred to as Valkyria Chronicles III outside Japan , " \
    "is a tactical role playing video game developed by Sega and Media.Vision for the PlayStation Portable ."
]

prompts += [
    "A robot may not injure a human being",
    "To be or not to be,",
    # "What is the meaning of life?",
    # "It is only with the heart that one can see rightly",
    # "How to recover from a breakup?",
    # "Which American-born Sinclair won the Nobel Prize for Literature in 1930?",
]

outputs = model.generate(prompts, sampling_params, quant_config, compress_config)
