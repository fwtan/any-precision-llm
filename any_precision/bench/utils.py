import torch
from torch.profiler import profile, record_function, ProfilerActivity
import os
import torch.backends.cudnn as cudnn
from tqdm import trange


this_dir = os.path.abspath(os.path.dirname(__file__))


@torch.no_grad()
def profile_model(model, inputs, export_path, device='cuda', num_iter=100):
    cudnn.benchmark = True
    model.eval()
    model.to(device)
    inputs = tuple(input.to(device) for input in inputs)
    print('Warming up...')
    for _ in trange(10):
        model(*inputs)
    print('Profiling...')
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, with_stack=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            for _ in trange(num_iter):
                model(*inputs)
    os.makedirs(export_path, exist_ok=True)
    print('Exporting profile to', export_path)
    profile_text_path = os.path.join(export_path, 'profile.txt')
    with open(profile_text_path, 'w') as f:
        print("CPU Time total:", file=f)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20),
              file=f)
        print("CUDA Time total:", file=f)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20),
              file=f)
        print("CPU Memory:", file=f)
        print(prof.key_averages().table(
            sort_by="self_cpu_memory_usage", row_limit=20), file=f)
        print("CUDA Memory:", file=f)
        print(prof.key_averages().table(
            sort_by="self_cuda_memory_usage", row_limit=20), file=f)
    time_stacks_path = os.path.join(export_path, 'cuda_time.stacks')
    prof.export_stacks(time_stacks_path, "self_cuda_time_total")
    # # Generate a flame graph
    # flame_graph_path = os.path.join(export_path, 'cuda_time_flame.svg')
    # os.system(f'{this_dir}/flamegraph.pl --title "CUDA Time" --countname "us." {time_stacks_path} > {flame_graph_path}')


def count_nan(x):
    return len(torch.nonzero(torch.isnan(x.view(-1))))

@torch.no_grad()
def profile_graph(model, inputs, export_path, device='cuda', num_iter=100, use_cuda_graph=False):
    cudnn.benchmark = True
    model.eval()
    model.to(device)
    # inputs = tuple(input.to(device) for input in inputs)
    print('Warming up...')

    # warmup
    # Uses static_input and static_target here for convenience,
    # but in a real setting, because the warmup includes optimizer.step()
    # you must use a few batches of real data.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(10):
            y_pred = model(*tuple([inputs["input_ids"], None, None, inputs["past_key_values"], None, None, True]))
    torch.cuda.current_stream().wait_stream(s)

    if use_cuda_graph:
        # capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_output = model(*tuple([inputs["input_ids"], None, None, inputs["past_key_values"], None, None, True]))
        # model = torch.cuda.make_graphed_callables(model, tuple([inputs["input_ids"], inputs["attention_mask"], inputs["position_ids"], inputs["past_key_values"]]), num_warmup_iters=10, allow_unused_input=True)

    real_inputs = [torch.randint_like(inputs["input_ids"], 0, 1000) for _ in range(num_iter)]

    print('Profiling...')
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, with_stack=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            for i in trange(num_iter):
                if use_cuda_graph:
                    inputs["input_ids"].copy_(real_inputs[i])
                    g.replay()
                    # print("out", torch.mean(static_output.logits).item(), torch.amax(static_output.logits).item(), torch.amin(static_output.logits).item())
                    # print("out", len(static_output.logits.view(-1)), count_nan(static_output.logits))
                else:
                    model(*tuple([real_inputs[i], None, None, inputs["past_key_values"], None, None, True]))
                # inputs["input_ids"].copy_(real_inputs[i])
                # g.replay()
                # dynamic_output = model(*tuple([real_inputs[i], None, None, inputs["past_key_values"], None, None, True]))
                # print("diff", torch.mean(static_output.logits - dynamic_output.logits).item(), torch.amax(dynamic_output.logits).item(), torch.amin(dynamic_output.logits).item(), torch.amax(static_output.logits).item(), torch.amin(static_output.logits).item())

    os.makedirs(export_path, exist_ok=True)
    print('Exporting profile to', export_path)
    profile_text_path = os.path.join(export_path, 'profile.txt')
    with open(profile_text_path, 'w') as f:
        print("CPU Time total:", file=f)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20),
              file=f)
        print("CUDA Time total:", file=f)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20),
              file=f)
        print("CPU Memory:", file=f)
        print(prof.key_averages().table(
            sort_by="self_cpu_memory_usage", row_limit=20), file=f)
        print("CUDA Memory:", file=f)
        print(prof.key_averages().table(
            sort_by="self_cuda_memory_usage", row_limit=20), file=f)
    time_stacks_path = os.path.join(export_path, 'cuda_time.stacks')
    prof.export_stacks(time_stacks_path, "self_cuda_time_total")
    # # Generate a flame graph
    # flame_graph_path = os.path.join(export_path, 'cuda_time_flame.svg')
    # os.system(f'{this_dir}/flamegraph.pl --title "CUDA Time" --countname "us." {time_stacks_path} > {flame_graph_path}')