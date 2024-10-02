## Nvidia GPU profiling

The code supports benchmarking different linear layers, e.g. `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`,  using [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html). 
For now, only [MobileLLaMA-1.4B-Chat](https://huggingface.co/mtgv/MobileLLaMA-1.4B-Chat), [Phi-1.5](https://huggingface.co/microsoft/phi-1_5), [Vicuna-7B-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5), and [StableLM Zephyr 3B](https://huggingface.co/stabilityai/stablelm-zephyr-3b) have been tested.

To benchmark the fp16 operations:

```
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${FP16_MODEL_PATH} --mode fp \
  --layer [q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj]
```

To benchmark the quantized operations:

```
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${QUANT_MODEL_PATH} --mode quant \
  --wbit [2|3|4] \
  --layer [q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj]
```

Note that there is no `gate_proj` for [Phi-1.5](https://huggingface.co/microsoft/phi-1_5).

To reproduce the results of our paper:

```
sh pmpd_bench.sh
```
