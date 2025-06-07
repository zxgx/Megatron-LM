import torch
import torch.distributed as dist

from benchmark_utils import profile


def init(mat_a, mat_b):
    torch.nn.init.uniform_(mat_a)
    torch.nn.init.uniform_(mat_b)


def step(mat_a, mat_b):
    with torch.profiler.record_function(f"matmul"):
        output = torch.matmul(mat_a, mat_b)


def main():
    dist.init_process_group(backend='nccl')
    
    hidden_size = 2048
    mat_a = torch.randn(32*1024, hidden_size//8, device='cuda')
    mat_b = torch.randn(hidden_size//8, hidden_size, device='cuda')
    profile(step, (mat_a, mat_b), 10, 10,
            trace_path='./trace_matmul',
            excluded_func=init, exfunc_inputs=(mat_a, mat_b))


if __name__ == "__main__":
    main()