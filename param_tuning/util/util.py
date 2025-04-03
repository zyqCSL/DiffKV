from typing import List, Union, Optional
import os

# log filtering
LLAMA3_WARNING = ('Special tokens have been added in the vocabulary, '
                  'make sure the associated word embeddings are fine-tuned or trained')
LONG_SEQ_WARNING = 'Token indices sequence length is longer than the specified maximum sequence length for this model'
ALLOWED_MSG = [LLAMA3_WARNING,
               LONG_SEQ_WARNING,
               ' -- Started a local Ray instance.']

# return True if error exists
def check_vllm_stderr(log: str):
    if not log:
        return False
    # print(f'{log} is True')
    log = str(log)
    err = False
    lines = log.split('\n')
    for line in lines:
        if not line:
            continue
        _match = False
        for msg in ALLOWED_MSG:
            if msg in line:
                _match = True
                break
        err = err or (not _match)
        if err:
            break
    return err


def compose_cmd(
    dataset: str,
    model_size: int,
    model_name: str,
    gpu_id: Union[int, List[int]], 
    max_num_seqs: int,
    buffer_size: int,
    # compression configs
    kbits_high: int,
    vbits_high: int,
    kbits_low: int,
    vbits_low: int,
    kv_prune_thresh: float,
    kv_quant_thresh: float,
    # framework configs
    gpu_mem_util: float,
    max_batched_tokens: int,
    max_padding: int,
    prompt_limit: int,
    # dataset
    indices_csv: str,
    label: Optional[str],
    # logging
    log_dir: str,
    file_dir_path: str,
    # dataset default
    zero_shot: bool = False,
    emulate: bool = False,
    attn_prune_thresh: Optional[float] = None,
    sample_rate: Optional[int] = None,
) -> str:
    assert dataset in [
        'wiki',
        'humaneval',
        'mbpp_plus',
        'mbpp',
        'gsm8k',
        'minerva_math',
        'aime',
        'mmlu_cot',
        'mmlu_pro_cot',
        'gpqa',
        'cnn_daily',
        'squad',
        'longbench',
        'longbench_v2',
    ]
    
    if model_size >= 56:
        assert type(gpu_id) is list and len(gpu_id) >= 4
    
    if type(gpu_id) is list:
        _gpu_id = ','.join([str(x) for x in gpu_id])
        tp_size = len(gpu_id)
    else:
        _gpu_id = gpu_id
        tp_size = 1
    
    if emulate:
        script_name = f'run_{dataset}_emulate.py'
        assert attn_prune_thresh is not None
    else:
        script_name = f'run_{dataset}.py'
    assert os.path.isfile(f'{file_dir_path}/{script_name}')
    
    cmd = (f'CUDA_VISIBLE_DEVICES={_gpu_id} RAY_DEDUP_LOGS=0 '
           f'python3 {file_dir_path}/{script_name} '
           f'--model {model_name} '
           '--load-format safetensors '
           '--download-dir /data1/huggingface '
           '--enforce-eager '
           '--dtype float16 '
           f'--kv-buffer-size {buffer_size} '
           # dataset compress configs
           f'--kbits-high {kbits_high} '
           f'--vbits-high {vbits_high} '
           f'--kbits-low {kbits_low} '
           f'--vbits-low {vbits_low} '
           f'--kv-prune-thresh {kv_prune_thresh} '
           f'--kv-quant-thresh {kv_quant_thresh} '
           # framework configs
           f'--gpu-memory-utilization {gpu_mem_util} '
           f'--max-num-batched-tokens {max_batched_tokens} '
           f'--max-paddings {max_padding} '
           f'--tensor-parallel-size {tp_size} '
           f'--prompt-limit {prompt_limit} '
           f'--max-num-seqs {max_num_seqs} '
           f'--log-path {log_dir} '
           f'--indices-csv {indices_csv}')
    if label:
        cmd += f' --data-label {label}'
    if emulate:
        cmd += f' --attn-prune-thresh {attn_prune_thresh}'
    if zero_shot:
        cmd += ' --zero-shot'
    if sample_rate:
        cmd += f' --sample-rate {sample_rate}'
    return cmd

def compose_codegen_cmd(
    dataset: str,
    model_size: int,
    model_name: str,
    gpu_id: Union[int, List[int]], 
    max_num_seqs: int,
    buffer_size: int,
    # compression configs
    kbits_high: int,
    vbits_high: int,
    kbits_low: int,
    vbits_low: int,
    kv_prune_thresh: float,
    kv_quant_thresh: float,
    # framework configs
    gpu_mem_util: float,
    max_batched_tokens: int,
    max_padding: int,
    prompt_limit: int,
    # dataset
    indices_csv: str,
    label: str,
    sampling_n: int,
    temperature: float,
    # logging
    log_dir: str,
    file_dir_path: str,
):
    cmd = compose_cmd(
        dataset=dataset,
        model_size=model_size,
        model_name=model_name,
        gpu_id=gpu_id, 
        max_num_seqs=max_num_seqs,
        buffer_size=buffer_size,
        # compression configs
        kbits_high=kbits_high,
        vbits_high=vbits_high,
        kbits_low=kbits_low,
        vbits_low=vbits_low,
        kv_prune_thresh=kv_prune_thresh,
        kv_quant_thresh=kv_quant_thresh,
        # framework configs
        gpu_mem_util=gpu_mem_util,
        max_batched_tokens=max_batched_tokens,
        max_padding=max_padding,
        prompt_limit=prompt_limit,
        # dataset
        indices_csv=indices_csv,
        label=label,
        # logging
        log_dir=log_dir,
        file_dir_path=file_dir_path,
    )
    
    cmd += (f' --sampling-n {sampling_n} '
            f'--temperature {temperature}')
    
    return cmd