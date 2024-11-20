MODEL_PATH=checkpoints/mtgv/MobileLLaMA-1.4B-Chat

CUDA_VISIBLE_DEVICES=0 python bench_graph.py --model_path ${MODEL_PATH} --mode fp --use_cuda_graph

MODEL_PATH=cache/packed/anyprec-MobileLLaMA-1.4B-Chat-w4_orig2-gc1-c4_s100_blk512

WBIT=2

CUDA_VISIBLE_DEVICES=0 python bench_graph.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --use_cuda_graph

WBIT=3

CUDA_VISIBLE_DEVICES=0 python bench_graph.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --use_cuda_graph

WBIT=4

CUDA_VISIBLE_DEVICES=0 python bench_graph.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --use_cuda_graph





MODEL_PATH=checkpoints/microsoft/phi-1_5

CUDA_VISIBLE_DEVICES=0 python bench_graph.py --model_path ${MODEL_PATH} --mode fp --use_cuda_graph

MODEL_PATH=cache/packed/anyprec-phi-1_5-w4_orig2-gc1-c4_s100_blk512

WBIT=2

CUDA_VISIBLE_DEVICES=0 python bench_graph.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --use_cuda_graph

WBIT=3

CUDA_VISIBLE_DEVICES=0 python bench_graph.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --use_cuda_graph

WBIT=4

CUDA_VISIBLE_DEVICES=0 python bench_graph.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --use_cuda_graph





MODEL_PATH=checkpoints/stabilityai/stablelm-zephyr-3b

CUDA_VISIBLE_DEVICES=0 python bench_graph.py --model_path ${MODEL_PATH} --mode fp --use_cuda_graph

MODEL_PATH=cache/packed/anyprec-stablelm-zephyr-3b-w4_orig2-gc1-c4_s100_blk512

WBIT=2

CUDA_VISIBLE_DEVICES=0 python bench_graph.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --use_cuda_graph
 
WBIT=3

CUDA_VISIBLE_DEVICES=0 python bench_graph.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --use_cuda_graph
 
WBIT=4

CUDA_VISIBLE_DEVICES=0 python bench_graph.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --use_cuda_graph





MODEL_PATH=checkpoints/lmsys/vicuna-7b-v1.5

CUDA_VISIBLE_DEVICES=0 python bench_graph.py --model_path ${MODEL_PATH} --mode fp --use_cuda_graph

MODEL_PATH=cache/packed/anyprec-vicuna-7b-v1.5-w4_orig2-gc1-c4_s100_blk512

WBIT=2

CUDA_VISIBLE_DEVICES=0 python bench_graph.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --use_cuda_graph

WBIT=3

CUDA_VISIBLE_DEVICES=0 python bench_graph.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --use_cuda_graph

WBIT=4

CUDA_VISIBLE_DEVICES=0 python bench_graph.py --model_path ${MODEL_PATH} --mode quant --wbit ${WBIT} --use_cuda_graph