<h3 align="center">
DiffKV: Differentiated KV Cache Management for LLM Inference
</h3>

---

_**DiffKV**_ is an LLM inference framework that enables efficent KV cache compression by jointly exploiting three levels of differentiation in the KV cache:

- The differing impact of keys and values on attention computation.

- The varying importance of tokens.

- The diverse dynamic sparsity patterns across attention heads.

These levels of differentiation introduce **irregular memory usage patterns across different requests and attention heads**, posing significant scalability challenges for memory management. To address these challenges, DiffKV proposes an **on-GPU memory manager** that compacts fragmented free memory list into contiguous regions in parallel, effectively translating sparsity in the KV cache into performance gains.

DiffKV is built on top of vLLM (commit [1db83e3](https://github.com/vllm-project/vllm/commit/1db83e31a2468cae37f326a642c0a4c4edbb5e4f)) and currently supports the following HuggingFace model architectures:

- **LLaMA-2 & LLaMA-3** (`meta-llama/Llama-2-7b-hf`, `meta-llama/Meta-Llama-3-8B-Instruct`, `meta-llama/Meta-Llama-3-70B-Instruct`, etc.)
- **Mistral** (`mistralai/Mistral-7B-v0.1`, etc.)
- **Mixtral** (`mistralai/Mixtral-8x7B-v0.1`, etc.)
- **Qwen-2.5** (`Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`, `Qwen/QwQ-32B`, etc.)
- **Qwen-3** (`Qwen/Qwen3-8B`, `Qwen/Qwen3-32B`, etc.)
- **Qwen-3 MoE** (`Qwen/Qwen3-30B-A3B`, etc.)

DiffKV supports model weight quantization using [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), and FP8 formats.


## Installation

#### Prerequisites:
- Python >= 3.10
- Nvidia GPUs with Ada, Hopper or newer architectures

#### Install DiffKV from source:

```bash
pip install -e .
```


## Usage

### Getting Started

> **Note:** Before running any scripts, make sure to add the DiffKV installation directory to your `PYTHONPATH`:
> ```bash
> export PYTHONPATH=$DiffKV_DIR:$PYTHONPATH
> ```

The `examples/` folder contains several demo scripts for experimenting with DiffKV, including:
`debug_generate_example.py`, `debug_qwen2.py` and `debug.py`.
Each script has a corresponding shell command (`*.sh`) for execution. Be sure to update the `PYTHONPATH` inside the shell scripts before running them.


### Generation Quality Experiments

The `param_tuning/` folder contains scripts for evaluating DiffKV's generation quality on several benchmarks. Specifically:

- `try_calibrate_group_*.sh` scripts contain commands to run hyperparameter calibration experiments for $\alpha_{high}$ and $\alpha_{low}$.
- `read_calibrate_per_token_thresh.py` parses the raw experiment outputs and summarizes the results.

Similarly:

- `try_$MODEL.sh` scripts are used to evaluate generation quality for various models and benchmarks.
- `read_$MODEL_per_token_thresh.py` scripts parse and summarize the corresponding results.

> **Note:** Before running any `read_*_per_token_thresh.py` scripts, set the DiffKV logs directory as an environment variable:
> ```bash
> export DIFFKV_LOG_DIR=${PATH_TO_DiffKV}/logs
> ```
> Or modify the default path directly in the script:
> ```python
> LOG_DIR = os.getenv('DIFFKV_LOG_DIR', '{PATH_TO_DiffKV}/logs')
> ```


### Functional Verification of the Artifact ###

To quickly verify the artifact setup:

1. Create a `logs/` directory in your DiffKV installation directory:
    ```bash
    mkdir -p DiffKV/logs
   ```

2. In the `param_tuning/` folder, run the script:
    ```bash
    ./try_small_models.sh
    ```
   Make sure `PYTHONPATH` is set to your DiffKV installation directory. This will save the raw results in: `DiffKV/logs/per_token_thresh/`.

3. Parse and summarize the results:
    ``` bash
    python read_small_models_results.py
    ```
   The summary will be saved in: `DiffKV/logs/per_token_thresh_compress_summary/`.


### Performance Benchmark

The `benchmarks/` directory contains scripts for evaluating DiffKV's performance.

#### Throughput

To reproduce the throughput numbers of DiffKV on the five models evaluated in the paper, you can use the `benchmark_throughput.sh` script. The threshold numbers used are the same as those reported in the paper.

#### Online Serving Latency

The `benchmarks/benchmark_serving.py` script measures online serving latency using a client-server setup.

**1. Start the Server**

```bash
python -m vllm.entrypoints.api_server \
  --model <your_model> \
  --port 8000 \
  --kv-buffer-size 32
```

**2. Run the Client**

```bash
python benchmarks/benchmark_serving.py \
  --model <your_model> \
  --port 8000 \
  --request-rate <request_rate> \
  --kv-prune-thresh <kv_prune_thresh> \
  --kv-quant-thresh <kv_quant_thresh>
```

## Quantization Configuration Guide

To add or modify quantization options, you must update several components in lockstep. This guide walks through each file and the required changes.

---

### 1. `vllm/config.py::CacheConfig`

- **Parameter:** `self.quantized_kv_bits`\
  A list of supported quantization tuples `(kbits, vbits, n_kgroups, n_vgroups)`:

  ```python
  self.quantized_kv_bits = [
      (8, 8, 1, 1),
      (8, 4, 1, 2),
      (8, 4, 1, 1),
      (8, 2, 1, 1),
      (4, 4, 1, 1),
      (4, 2, 2, 4),
      (4, 2, 1, 1),
      (4, 1, 1, 1),
  ]
  ```

  - `n_kgroups` and `n_vgroups` specify how many subgroups each head (typically with a hidden size of 128) is split into for quantization metadata (scale & zero point).
  - Up to **4 groups per token** are supported by the current CUDA kernels.
  - By default, each token is quantized with a single group of metadata. Using **multiple** sets of metadata is potentially more effective for GQA architectures with high queries-per-KV ratios, as well as for long CoT (Chain-of-Thought) generation.

- **Method:** `CacheConfig.compute_cache_block()`\
  Computes:

  1. Optimal block size (in INT16 units).
  2. Tokens per page (for both high- and low-precision data).

- **Constructor args:**

  - `num_thread_groups_k`: Number of key thread-groups per warp.
  - `key_vec_size` & `value_vec_size`: Number of INT16 values read contiguously by each thread to maximize memory bandwidth.

---

### 2. CUDA Kernels: `csrc/cache_kernel.cu`

1. **Register new quant configs** in these macros:

   - `CALL_PROMPT_PHASE_CACHE_LAUNCHER_QUANT_CONFIG`
   - `CALL_DECODE_PHASE_CACHE_LAUNCHER_QUANT_CONFIG`

   ```cpp
   #define CALL_PROMPT_PHASE_CACHE_LAUNCHER_QUANT_CONFIG(T, HEAD_SIZE)               \
   if (quant_config == std::vector<int>{                                              \
       kbits_high, vbits_high, kbits_low, vbits_low,                                 \
       n_kgroups_high, n_vgroups_high, n_kgroups_low, n_vgroups_low                   \
   }) {                                                                                \
       assert(num_tokens_per_page_high == <computed_high>);                           \
       assert(num_tokens_per_page_low  == <computed_low>);                            \
       assert(thread_group_size_v      == <computed_vgroup_size>);                     \
       LAUNCH_CACHE_KERNEL_PROMPT_PHASE(                                              \
           T, HEAD_SIZE,
           kbits_high, vbits_high, kbits_low, vbits_low,
           n_kgroups_high, n_vgroups_high,
           n_kgroups_low, n_vgroups_low,
           num_tokens_per_page_high,
           num_tokens_per_page_low,
           thread_group_size_v
       );
   }
   ```

2. **Page-size assertions** in:

   - `compress_and_append_cache_decode_phase()`
   - `compress_and_append_cache_prompt_phase()`

   ```cpp
   assert(unified_page_size == <computed_page_size>);
   ```

> ⚠️ Ensure all `assert(...)` values match your `compute_cache_block()` outputs.

---

### 3. Long-prompt Kernel: `csrc/long_prompt_cache_kernels.cu`

- Follow the same pattern as above:
  1. Register in `CALL_PROMPT_PHASE_CACHE_LAUNCHER_QUANT_CONFIG`.
  2. Update the page-size assertion in `compress_and_append_cache_long_prompt_phase()`.

---

### 4. Sparse Attention: `csrc/attention/sparse_attention_kernels.cu`

- Add your config to the `CALL_LAUNCHER_QUANT_CONFIG` macro:
  ```cpp
  #define CALL_LAUNCHER_QUANT_CONFIG(T)                                      \
  if (quant_config == std::vector<int>{                                       \
      kbits_high, vbits_high, kbits_low, vbits_low,                          \
      n_kgroups_high, n_vgroups_high, n_kgroups_low, n_vgroups_low             \
  }) {                                                                        \
      assert(num_tokens_per_page_high == <computed_high>);                     \
      assert(num_tokens_per_page_low  == <computed_low>);                      \
      assert(thread_group_size_v      == <computed_vgroup_size>);               \
      CALL_LAUNCHER(
          T,
          kbits_high, vbits_high, kbits_low, vbits_low,
          n_kgroups_high, n_vgroups_high,
          n_kgroups_low, n_vgroups_low,
          num_tokens_per_page_high,
          num_tokens_per_page_low,
          thread_group_size_v
      );
  }
  ```

---

### 5. Benchmark Utilities: `param_tuning/util/util.py`

- **Mapping:** Update `BITS_TO_QUANT_GROUPS` to reflect new bit-to-group mappings:

  ```python
  BITS_TO_QUANT_GROUPS = {
      8: 1,
      4: 2,
      2: 4,
      # add new_bit: group_count,
  }
  ```

- `` assumes a single group when `kbits == vbits`:

  ```python
  if kbits_high == vbits_high:
      # baseline: one group for both key & value
      quant_groups = [1, 1]
  ```

  Adjust this logic if your design requires different behavior.

---

Keep all components synchronized. Any mismatch between code, assertions, and computed values will cause runtime errors.

## Citation
If you use DiffKV for your research, please cite our [paper](https://arxiv.org/abs/2412.03131):
```bibtex
@inproceedings{zhang2025diffkv,
  title={DiffKV: Differentiated Memory Management for Large Language Models with Parallel KV Compaction},
  author={Yanqi Zhang, Yuwei Hu, Runyuan Zhao, John C.S. Lui and Haibo Chen},
  booktitle={Proceedings of the ACM SIGOPS 31th Symposium on Operating Systems Principles},
  year={2025}
}
```
