# FP

MODEL_PATH=checkpoints/lmsys/vicuna-7b-v1.5

CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer q_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer k_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer v_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer o_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer gate_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer up_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer down_proj



MODEL_PATH=checkpoints/mtgv/MobileLLaMA-1.4B-Chat

CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer q_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer k_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer v_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer o_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer gate_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer up_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer down_proj



MODEL_PATH=checkpoints/microsoft/phi-1_5

CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer q_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer k_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer v_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer o_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer up_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer down_proj


MODEL_PATH=checkpoints/stabilityai/stablelm-zephyr-3b

CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer q_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer k_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer v_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer o_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer gate_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer up_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode fp --layer down_proj


# Quant

MODEL_PATH=cache/packed/anyprec-MobileLLaMA-1.4B-Chat-w4_orig2-gc1-c4_s100_blk512

WBIT=2

CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer q_proj 
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer k_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer v_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer o_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer gate_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer up_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer down_proj


WBIT=3

CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer q_proj 
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer k_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer v_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer o_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer gate_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer up_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer down_proj


WBIT=4

CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer q_proj 
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer k_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer v_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer o_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer gate_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer up_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer down_proj


MODEL_PATH=cache/packed/anyprec-phi-1_5-w4_orig2-gc1-c4_s100_blk512

WBIT=2

CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer q_proj 
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer k_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer v_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer o_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer up_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer down_proj


WBIT=3

CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer q_proj 
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer k_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer v_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer o_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer up_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer down_proj


WBIT=4

CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer q_proj 
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer k_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer v_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer o_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer up_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer down_proj 


MODEL_PATH=cache/packed/anyprec-stablelm-zephyr-3b-w4_orig2-gc1-c4_s100_blk512

WBIT=2

CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer q_proj 
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer k_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer v_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer o_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer gate_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer up_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer down_proj 


WBIT=3

CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer q_proj 
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer k_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer v_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer o_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer gate_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer up_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer down_proj


WBIT=4

CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer q_proj 
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer k_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer v_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer o_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer gate_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer up_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer down_proj


MODEL_PATH=cache/packed/anyprec-vicuna-7b-v1.5-w4_orig2-gc1-c4_s100_blk512

WBIT=2

CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer q_proj 
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer k_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer v_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer o_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer gate_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer up_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer down_proj


WBIT=3

CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer q_proj 
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer k_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer v_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer o_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer gate_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer up_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer down_proj


WBIT=4

CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer q_proj 
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer k_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer v_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer o_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer gate_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer up_proj
CUDA_VISIBLE_DEVICES=0 python bench.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --layer down_proj
