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


def _make_causal_mask(q_len, kv_seq_len, valid_len):
    assert (kv_seq_len >= valid_len) and (kv_seq_len >= q_len)
    msk = torch.tril(torch.ones((1, 1, kv_seq_len, kv_seq_len)))
    msk = msk[:, :, (-q_len):]
    msk = msk.reshape((1, 1, q_len, kv_seq_len))
    invalid_len = kv_seq_len - valid_len
    if invalid_len > 0:
        msk[:,:,:,:invalid_len] = 0
    out = torch.zeros((1, 1, q_len, kv_seq_len), dtype=torch.float16).masked_fill(msk == 0, 1.0)
    return out


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
 
    # # create the cache
    # context = model(torch.randint(0, tokenizer.vocab_size, size=(args.batch_size, args.context_length), dtype=torch.int32).cuda())
    # print((context.past_key_values[-1][0].shape, context.past_key_values[-1][1].shape))

    past_key_values = tuple([(
        torch.rand(args.batch_size, model.config.num_key_value_heads, args.context_length, model.config.hidden_size // model.config.num_attention_heads).to(torch.float16).cuda(),
        torch.rand(args.batch_size, model.config.num_key_value_heads, args.context_length, model.config.hidden_size // model.config.num_attention_heads).to(torch.float16).cuda()
        ) for _ in range(len(model.model.layers))])

    if isinstance(model, (transformers.PreTrainedModel, )):
        input_ids = torch.randint(0, tokenizer.vocab_size, size=(args.batch_size, 1), dtype=torch.int32).cuda()
        attention_mask = _make_causal_mask(1, args.context_length+1, args.context_length+1).cuda()
        past_seen_tokens = args.context_length
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + 1, device=input_ids.device)
        position_ids = cache_position.unsqueeze(0).expand(args.batch_size, 1).contiguous()
        inp = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            # "past_key_values": context.past_key_values,
            "past_key_values": past_key_values,
            "inputs_embeds": None,
            "labels": None,
            "use_cache": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": None,
            # "cache_position": cache_position,
        }
    else:
        raise NotImplementedError
    
    profile_graph(model, inp, args.output_dir, num_iter=args.num_iter, use_cuda_graph=args.use_cuda_graph)

        
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
    parser.add_argument('--num_iter', type=int, default=100)
    parser.add_argument('--use_cuda_graph', default=False, action="store_true")
    parser.add_argument("--output_dir", default='results/bench', type=str)
    args = parser.parse_args()

    if args.model_path.endswith('/'):
        args.model_path = args.model_path[:-1]
    args.model_name = osp.basename(args.model_path)
    if args.mode == "quant":
        args.model_name = f"{args.model_name}_w{args.wbit}"
    args.output_dir = osp.join(args.output_dir, args.model_name)
    main(args)