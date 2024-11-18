import os
import json
import torch
import argparse
import os.path as osp

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.phi.modeling_phi import PhiDecoderLayer
from any_precision.bench.utils import profile_graph
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
 
    # create the cache
    context = model(torch.randint(0, tokenizer.vocab_size, size=(args.batch_size, args.context_length), dtype=torch.int32).cuda())

    if isinstance(model, (transformers.PreTrainedModel, )):
        input_ids = torch.randint(0, tokenizer.vocab_size, size=(args.batch_size, 1), dtype=torch.int32).cuda()
        past_seen_tokens = args.context_length
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + 1, device=input_ids.device)
        position_ids = cache_position.unsqueeze(0)
        inp = {
            "input_ids": input_ids,
            "attention_mask": None,
            "position_ids": position_ids,
            "past_key_values": context.past_key_values,
            "inputs_embeds": None,
            "labels": None,
            "use_cache": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": None,
            "cache_position": cache_position,
        }
    # else:
    #     inp = (inp, )
    
    profile_graph(model, inp, args.output_dir)

        
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
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--context_length', type=int, default=1024)
    parser.add_argument('--autoregressive', default=False, action="store_true")
    parser.add_argument("--output_dir", default='results/bench', type=str)
    args = parser.parse_args()

    if args.model_path.endswith('/'):
        args.model_path = args.model_path[:-1]
    args.model_name = osp.basename(args.model_path)
    if args.mode == "quant":
        args.model_name = f"{args.model_name}_w{args.wbit}"
    args.output_dir = osp.join(args.output_dir, args.model_name)
    main(args)