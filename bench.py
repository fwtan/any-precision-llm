import os
import json
import torch
import argparse
import os.path as osp

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.phi.modeling_phi import PhiDecoderLayer
from any_precision.bench.utils import profile_model
from any_precision.modules.AnyPrecisionLinear import AnyPrecisionLinear
from any_precision.evaluate.helpers import utils
from any_precision.evaluate import eval as quant_eval


@torch.no_grad()
def main(args):    
    if args.mode == "fp":
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, legacy=False, trust_remote_code=True)
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, 
            config=config, 
            device_map='auto', 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            trust_remote_code=True, 
        )
    else:
        tokenizer_type, tokenizer, model = quant_eval.auto_model_load(args.model_path)
        model.set_precision(args.wbit)
        model = model.model
    model = model.model.layers[-1]
    if args.layer is not None:
        if args.layer == "q_proj":
            model = model.self_attn.q_proj
        elif args.layer == "k_proj":
            model = model.self_attn.k_proj
        elif args.layer == "v_proj":
            model = model.self_attn.v_proj
        elif args.layer == "o_proj":
            if isinstance(model, PhiDecoderLayer):
                model = model.self_attn.dense
            else:
                model = model.self_attn.o_proj
        elif args.layer == "gate_proj":
            if isinstance(model, PhiDecoderLayer):
                raise NotImplementedError
            else:
                model = model.mlp.gate_proj
        elif args.layer == "up_proj":
            if isinstance(model, PhiDecoderLayer):
                model = model.mlp.fc1
            else:
                model = model.mlp.up_proj
        elif args.layer == "down_proj":
            if isinstance(model, PhiDecoderLayer):
                model = model.mlp.fc2
            else:
                model = model.mlp.down_proj
        else:
            raise NotImplementedError
    print(model)

    if isinstance(model, (torch.nn.Linear, AnyPrecisionLinear)):
        inp = torch.randn(1, 1, model.in_features, dtype=torch.float16).cuda()
    else:
        inp = torch.randn(1, 1, model.self_attn.q_proj.in_features, dtype=torch.float16).cuda()
    
    profile_model(model, (inp,), args.output_dir)

        
if __name__ == '__main__':
    seed = 1337
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--mode', type=str, default="fp", choices=["fp", "quant"])
    parser.add_argument('--wbit', type=int, default=2, choices=[2,3,4])
    parser.add_argument('--layer', type=str, default=None)
    parser.add_argument("--output_dir", default='results/bench', type=str)
    args = parser.parse_args()

    if args.model_path.endswith('/'):
        args.model_path = args.model_path[:-1]
    args.model_name = osp.basename(args.model_path)
    if args.mode == "quant":
        args.model_name = f"{args.model_name}_w{args.wbit}"
    if args.layer is not None:
        args.model_name = f"{args.model_name}_{args.layer}"
    args.output_dir = osp.join(args.output_dir, args.model_name)
    main(args)