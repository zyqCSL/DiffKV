from vllm import LLM, SamplingParams

model = LLM('meta-llama/Llama-2-7b-chat-hf',
            #'meta-llama/Meta-Llama-3-8B-Instruct',
            download_dir='/data1/huggingface',
            gpu_memory_utilization=0.8,
            kv_buffer_size=32,
            dtype='float16',
            disable_log_stats=False,
            enforce_eager=True)

sampling_params = SamplingParams(temperature=0.0, max_tokens=64)  # greedy sampling

# [kbits_high, vbits_high, kbits_low, vbits_low]
quant_config = [8, 8, 8, 4]

# [prune_thresh, quant_thresh, prune_ratio, quant_ratio]
# compress_config = [1.0, 0.99, 0.5, 0.25]
compress_config = [0.0, 0.1]

# dummy prompts consisting of 'hi' repeated seq_len times
batch_size = 4
seq_len = 3500
seq_lens = [seq_len] * batch_size

prompts = []
for seq_len in seq_lens:
    prompt = ' '.join(['hi' for _ in range(seq_len)])
    prompts.append(prompt)

outputs = model.generate(prompts, sampling_params, quant_config, compress_config)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
