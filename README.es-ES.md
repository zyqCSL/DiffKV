<h3 align="center">
DiffKV: Gestión Diferenciada de la Caché KV para la Inferencia de LLM
</h3>

---

_**DiffKV**_ es un marco de inferencia de LLM que permite la compresión eficiente de la caché KV aprovechando conjuntamente tres niveles de diferenciación en la caché KV:

- El impacto diferente de las claves y los valores en el cálculo de la atención.

- La importancia variable de los tokens.

- Los diversos patrones de dispersidad dinámica entre los cabezales de atención.

Estos niveles de diferenciación introducen **patrones de uso de memoria irregulares entre diferentes solicitudes y cabezales de atención**, lo que plantea desafíos significativos de escalabilidad en la gestión de memoria. Para abordar estos desafíos, DiffKV propone un **gestor de memoria en GPU** que compacta listas fragmentadas de memoria libre en regiones contiguas en paralelo, traduciendo eficazmente la dispersidad en la caché KV en ganancias de rendimiento.

DiffKV está construido sobre vLLM (commit [1db83e3](https://github.com/vllm-project/vllm/commit/1db83e31a2468cae37f326a642c0a4c4edbb5e4f)) y actualmente admite las siguientes arquitecturas de modelos de HuggingFace:

- **LLaMA-2 y LLaMA-3** (`meta-llama/Llama-2-7b-hf`, `meta-llama/Meta-Llama-3-8B-Instruct`, `meta-llama/Meta-Llama-3-70B-Instruct`, etc.)
- **Mistral** (`mistralai/Mistral-7B-v0.1`, etc.)
- **Mixtral** (`mistralai/Mixtral-8x7B-v0.1`, etc.)
- **Qwen-2.5** (`Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`, `Qwen/QwQ-32B`, etc.)
- **Qwen-3** (`Qwen/Qwen3-8B`, `Qwen/Qwen3-32B`, etc.)
- **Qwen-3 MoE** (`Qwen/Qwen3-30B-A3B`, etc.)

DiffKV admite la cuantización de pesos del modelo utilizando formatos [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978) y FP8.


## Instalación

#### Requisitos previos:
- Python >= 3.10
- GPUs Nvidia con arquitecturas Ada, Hopper o posteriores

#### Instalar DiffKV desde el código fuente:

```bash
pip install -e .
```


## Uso

### Primeros Pasos

> **Nota:** Antes de ejecutar cualquier script, asegúrese de añadir el directorio de instalación de DiffKV a su `PYTHONPATH`:
> ```bash
> export PYTHONPATH=$DiffKV_DIR:$PYTHONPATH
> ```

La carpeta `examples/` contiene varios scripts de demostración para experimentar con DiffKV, incluyendo:
`debug_generate_example.py`, `debug_qwen2.py` y `debug.py`.
Cada script tiene un comando de shell correspondiente (`*.sh`) para su ejecución. Asegúrese de actualizar el `PYTHONPATH` dentro de los scripts de shell antes de ejecutarlos.


### Experimentos de Calidad de Generación

La carpeta `param_tuning/` contiene scripts para evaluar la calidad de generación de DiffKV en varios benchmarks. Específicamente:

- Los scripts `try_calibrate_group_*.sh` contienen comandos para ejecutar experimentos de calibración de hiperparámetros para $\alpha_{high}$ y $\alpha_{low}$.
- `read_calibrate_per_token_thresh.py` analiza las salidas brutas de los experimentos y resume los resultados.

De manera similar:

- Los scripts `try_$MODEL.sh` se utilizan para evaluar la calidad de generación de varios modelos y benchmarks.
- Los scripts `read_$MODEL_per_token_thresh.py` analizan y resumen los resultados correspondientes.

> **Nota:** Antes de ejecutar cualquier script `read_*_per_token_thresh.py`, establezca el directorio de logs de DiffKV como una variable de entorno:
> ```bash
> export DIFFKV_LOG_DIR=${PATH_TO_DiffKV}/logs
> ```
> O modifique la ruta predeterminada directamente en el script:
> ```python
> LOG_DIR = os.getenv('DIFFKV_LOG_DIR', '{PATH_TO_DiffKV}/logs')
> ```


### Verificación Funcional del Artefacto ###

Para verificar rápidamente la configuración del artefacto:

1. Cree un directorio `logs/` en su directorio de instalación de DiffKV:
    ```bash
    mkdir -p DiffKV/logs
   ```

2. En la carpeta `param_tuning/`, ejecute el script:
    ```bash
    ./try_small_models.sh
    ```
   Asegúrese de que `PYTHONPATH` esté configurado como su directorio de instalación de DiffKV. Esto guardará los resultados brutos en: `DiffKV/logs/per_token_thresh/`.

3. Analice y resuma los resultados:
    ``` bash
    python read_small_models_results.py
    ```
   El resumen se guardará en: `DiffKV/logs/per_token_thresh_compress_summary/`.


### Benchmark de Rendimiento

El directorio `benchmarks/` contiene scripts para evaluar el rendimiento de DiffKV.

#### Rendimiento (Throughput)

Para reproducir los números de rendimiento de DiffKV en los cinco modelos evaluados en el artículo, puede usar el script `benchmark_throughput.sh`. Los números de umbral utilizados son los mismos que los reportados en el artículo.

#### Latencia de Servicio en Línea

El script `benchmarks/benchmark_serving.py` mide la latencia de servicio en línea utilizando una configuración de cliente-servidor.

**1. Iniciar el Servidor**

```bash
python -m vllm.entrypoints.api_server \
  --model <your_model> \
  --port 8000 \
  --kv-buffer-size 64
```

**2. Ejecutar el Cliente**

```bash
python benchmarks/benchmark_serving.py \
  --model <your_model> \
  --port 8000 \
  --request-rate <request_rate> \
  --kv-prune-thresh <kv_prune_thresh> \
  --kv-quant-thresh <kv_quant_thresh>
```

## Guía de Configuración de Cuantización

Para añadir o modificar opciones de cuantización, debe actualizar varios componentes de forma coordinada. Esta guía le guía a través de cada archivo y los cambios requeridos.

---

### 1. `vllm/config.py::CacheConfig`

- **Parámetro:** `self.quantized_kv_bits`\
  Una lista de tuplas de cuantización admitidas `(kbits, vbits, n_kgroups, n_vgroups)`:

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

  - `n_kgroups` y `n_vgroups` especifican en cuántos subgrupos se divide cada cabezal (típicamente con un tamaño oculto de 128) para la metadatos de cuantización (escala y punto cero).
  - Hasta **4 grupos por token** son compatibles con los kernels CUDA actuales.
  - De forma predeterminada, cada token se cuantiza con un solo grupo de metadatos. Utilizar **múltiples** conjuntos de metadatos puede ser potencialmente más efectivo para arquitecturas GQA con altas ratios de consultas-por-KV, así como para generación larga de CoR (Chain-of-Thought).

- **Método:** `CacheConfig.compute_cache_block()`\
  Calcula:

  1. Tamaño de bloque óptimo (en unidades INT16).
  2. Tokens por página (para datos de alta y baja precisión).

- **Argumentos del constructor:**

  - `num_thread_groups_k`: Número de grupos de hilos de clave por warp.
  - `key_vec_size` y `value_vec_size`: Número de valores INT16 leídos contiguamente por cada hilo para maximizar el ancho de banda de memoria.

---

### 2. Kernels CUDA: `csrc/cache_kernel.cu`

1. **Registre nuevas configuraciones de cuantización** en estas macros:

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

2. **Aserciones de tamaño de página** en:

   - `compress_and_append_cache_decode_phase()`
   - `compress_and_append_cache_prompt_phase()`

   ```cpp
   assert(unified_page_size == <computed_page_size>);
   ```

> ⚠️ Asegúrese de que todos los valores de `assert(...)` coincidan con las salidas de su `compute_cache_block()`.

---

### 3. Kernel de prompts largos: `csrc/long_prompt_cache_kernels.cu`

- Siga el mismo patrón que arriba:
  1. Regístrela en `CALL_PROMPT_PHASE_CACHE_LAUNCHER_QUANT_CONFIG`.
  2. Actualice la aserción del tamaño de página en `compress_and_append_cache_long_prompt_phase()`.

---

### 4. Atención Dispersa: `csrc/attention/sparse_attention_kernels.cu`

- Añada su configuración a la macro `CALL_LAUNCHER_QUANT_CONFIG`:
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

### 5. Utilidades de Benchmark: `param_tuning/util/util.py`

- **Mapeo:** Actualice `BITS_TO_QUANT_GROUPS` para reflejar los nuevos mapeos de bits a grupos:

  ```python
  BITS_TO_QUANT_GROUPS = {
      8: 1,
      4: 2,
      2: 4,
      # add new_bit: group_count,
  }
  ```

- `` asume un solo grupo cuando `kbits == vbits`:

  ```python
  if kbits_high == vbits_high:
      # baseline: one group for both key & value
      quant_groups = [1, 1]
  ```

  Ajuste esta lógica si su diseño requiere un comportamiento diferente.

---

Mantenga todos los componentes sincronizados. Cualquier discrepancia entre el código, las aserciones y los valores calculados provocará errores en tiempo de ejecución.

## Cita
Si utiliza DiffKV para su investigación, por favor cite nuestro [artículo](https://arxiv.org/abs/2412.03131):
```bibtex
@inproceedings{zhang2025diffkv,
  title={DiffKV: Differentiated Memory Management for Large Language Models with Parallel KV Compaction},
  author={Zhang, Yanqi and Hu, Yuwei and Zhao, Runyuan and Lui, John CS and Chen, Haibo},
  booktitle={Proceedings of the ACM SIGOPS 31st Symposium on Operating Systems Principles},
  pages={431--445},
  year={2025}
}
```
