import sys
sys.path.append(".") # enter from Megatron-LM
import os
import pickle
import torch
import torch.distributed as dist

from megatron import get_args
from megatron.arguments import core_transformer_config_from_args
from megatron.initialize import initialize_megatron, set_jit_fusion_options
from megatron.model.transformer import ParallelTransformerLayer
from megatron.model.flatten_transformer_layer import FlattenTransformerLayer
from megatron.core.tensor_parallel import get_cuda_rng_tracker, checkpoint
from megatron.model.enums import AttnMaskType, LayerType
from megatron.model import Float16Module, DistributedDataParallel as LocalDDP

from benchmark_utils import dist_fwd_bwd_benchmark, profile


def init(module, input_tensor, grad_tensor, rng_tracker):
    input_tensor.grad = None
    module.zero_grad()
    with rng_tracker.fork():
        torch.nn.init.uniform_(input_tensor)
        torch.nn.init.uniform_(grad_tensor)


def forward(module, input_tensor):
    output_tensor = module(input_tensor, None)
    # output_tensor = checkpoint(
    #     lambda _x: module(_x, None),
    #     False,
    #     input_tensor)
    return output_tensor


def backward(output_tensor, grad_tensor):
    output_tensor.backward(grad_tensor)


def step(module, input_tensor, grad_tensor):
    with torch.profiler.record_function(f"Megatron FWD"):
        output_tensor = module(input_tensor, None)
        # output_tensor = checkpoint(
        #     lambda _x: module(_x, None),
        #     False,
        #     input_tensor)
    
    with torch.profiler.record_function(f"Megatron BWD"):
        output_tensor.backward(grad_tensor)


def main(run_benchmark, run_profile, 
         extra_args_provider=None, args_defaults={}):
    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)
    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    args = get_args()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    rng_tracker = get_cuda_rng_tracker()
    device = torch.cuda.current_device()

    fmt_str = "batch size: {}, seq len: {}, hidden size: {}"

    bs = args.micro_batch_size
    sl = args.seq_length
    hs = args.hidden_size
    dt = torch.float16
    
    local_sl = sl
    if args.sequence_parallel:
        local_sl = sl//world_size
    input_tensor = torch.empty(
        [local_sl, bs, hs], device=device, requires_grad=True, dtype=dt)
    grad_tensor = torch.empty(
        [local_sl, bs, hs], device=device, requires_grad=True, dtype=dt)

    config = core_transformer_config_from_args(args)

    if args.checkpoint_without_attn:
        layer_module = FlattenTransformerLayer
    else:
        layer_module = ParallelTransformerLayer
    model = layer_module(
        config,
        layer_number=1,
        layer_type=LayerType.encoder,
        self_attn_mask_type=AttnMaskType.causal,
        drop_path_rate=0.
    ).cuda(device)
    
    if args.fp16 or args.bf16:
        model = Float16Module(model, args)
    model = LocalDDP(
        model, args.accumulate_allreduce_grads_in_fp32, 
        args.use_contiguous_buffers_in_local_ddp)

    if run_benchmark:
        (fwd_std, fwd_avg), (bwd_std, bwd_avg) = dist_fwd_bwd_benchmark(
            forward, (model, input_tensor,),
            backward, (grad_tensor, ), warm_up=10, iters=10,
            excluded_func=init, exfunc_inputs=(model, input_tensor, grad_tensor, rng_tracker)
        )
    
        if rank == 0:
            print(f">>> {fmt_str.format(bs, sl, hs)}\n"
                f"forward avg: {fwd_avg:.1f} ms, "
                f"forward std: {fwd_std:.1f} ms, "
                f"backward avg: {bwd_avg:.1f} ms, "
                f"backward std: {bwd_std:.1f} ms"
                f"\n\n" ,flush=True, end="")
    
    if run_profile:
        profile(step, (model, input_tensor, grad_tensor), 10, 10,
                f"{args.profile_prefix}/tp{world_size}-bs{bs}-sl{sl}-hs{hs}",
                excluded_func=init, exfunc_inputs=(model, input_tensor, grad_tensor, rng_tracker, ))


if __name__ == "__main__":    
    run_benchmark = int(os.getenv("BENCHMARK", str(0)))
    run_profile = int(os.getenv("PROFILE", str(0)))

    main(run_benchmark, run_profile,
         args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                        'num_layers': 64,
                        'vocab_file':'benchmark/gpt2tokenizer/gpt2-vocab.json',
                        'merge_file':'benchmark/gpt2tokenizer/gpt2-merges.txt',
                       })
