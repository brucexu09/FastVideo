import argparse
import contextlib
import logging
from typing import Tuple

import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity  # pyre-ignore

# Import video sparse attention
from vsa import video_sparse_attn
from xformers.ops import fmha, memory_efficient_attention

from csrc.attn.flex_attn_vsa import VSA_attention_flex

# use flash attention v3
fmha._set_use_fa3(True)

# VSA VARIABLES
BLOCK_SIZE = (4, 4, 4)
BLOCK_VOLUME = BLOCK_SIZE[0] * BLOCK_SIZE[1] * BLOCK_SIZE[2]
CONTEXT_PARALLEL = 2

def create_qkv_tensors(batch, num_heads, seq_len, head_dim, device="cuda"):
    """
    Create random input tensors for attention with shape of [batch_size, seq_len, num_heads, head_dim],
    which is required by Flash Attention v3.
    """

    q = torch.randn(
        batch, seq_len // CONTEXT_PARALLEL, num_heads, head_dim, requires_grad=True, dtype=torch.bfloat16, device=device
    )
    k = torch.randn(
        batch, seq_len, num_heads, head_dim, requires_grad=True, dtype=torch.bfloat16, device=device
    )
    v = torch.randn(
        batch, seq_len, num_heads, head_dim, requires_grad=True, dtype=torch.bfloat16, device=device
    )

    return q, k, v


def create_og_tensors(batch, num_heads, seq_len, head_dim, device="cuda"):
    """
    Create random output gradient tensors for attention with shape of [batch_size, seq_len, num_heads, head_dim],
    which is required by Flash Attention v3.
    """

    og = torch.randn(
        batch, seq_len // CONTEXT_PARALLEL, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )

    return og


def create_tiled_qkv_tensors(batch, num_heads, seq_len, head_dim, device="cuda"):
    """
    Create random input tensors for attention with shape of [batch_size, num_heads, seq_len, head_dim]
    which is required shape by Video Sparse Attention to avoid permute.
    """

    q = torch.randn(
        batch, num_heads, seq_len // CONTEXT_PARALLEL, head_dim, requires_grad=True, dtype=torch.bfloat16, device=device
    )
    k = torch.randn(
        batch, num_heads, seq_len, head_dim, requires_grad=True, dtype=torch.bfloat16, device=device
    )
    v = torch.randn(
        batch, num_heads, seq_len, head_dim, requires_grad=True, dtype=torch.bfloat16, device=device
    )

    return q, k, v


def create_tiled_og_tensors(batch, num_heads, seq_len, head_dim, device="cuda"):
    """
    Create random output gradient tensors for attention with shape of [batch_size, num_heads, seq_len, head_dim]
    which is required shape by Video Sparse Attention to avoid permute.
    """

    og = torch.randn(
        batch, num_heads, seq_len // CONTEXT_PARALLEL, head_dim, dtype=torch.bfloat16, device=device
    )

    return og


def profiler_or_nullcontext(
    enabled: bool,
    with_stack: bool = False,
    trace_file_name: str = "video_sparse_attn_trace",
    trace_file_path_dir: str = "video_sparse_attn_file_path_dir",
    record_shapes: bool = False,
):
    def _kineto_trace_handler(p: torch.profiler.profile) -> None:
        p.export_chrome_trace(f"{trace_file_path_dir}/{trace_file_name}.json")

    return (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # pyre-ignore
            on_trace_ready=_kineto_trace_handler,
            with_stack=with_stack,
            record_shapes=record_shapes,
        )
        if enabled
        else contextlib.nullcontext()
    )


def prepare_tensors(
    bench_atten_type, batch_size, num_heads, seq_len, head_dim
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if bench_atten_type == "FA3":
        q, k, v = create_qkv_tensors(
            batch_size, num_heads, seq_len, head_dim, device="cuda"
        )
        og = create_og_tensors(
            batch_size, num_heads, seq_len, head_dim, device="cuda"
        )
    else:
        q, k, v = create_tiled_qkv_tensors(
            batch_size, num_heads, seq_len, head_dim, device="cuda"
        )
        og = create_tiled_og_tensors(
            batch_size, num_heads, seq_len, head_dim, device="cuda"
        )

    return (q, k, v, og)


def run_inference(inferencer, q, k, v, active_kv_num, bench_atten_type):

    if bench_atten_type == "FA3":
        return inferencer(q, k, v)

    elif bench_atten_type in ["VSA", "VSA_flex"]:
        return inferencer(
            q,
            k,
            v,
            topk=active_kv_num,
            block_size=BLOCK_SIZE,
            # mode=bench_atten_type,
        )

    else:
        raise NotImplementedError("Unknown attention type")


def run_backward(o, og, bench_atten_type):
    if bench_atten_type == "FA3":
        o.backward(gradient=og)

    elif bench_atten_type in ["VSA", "VSA_flex"]:
        o.backward(gradient=og)
    else:
        raise NotImplementedError("Unknown attention type")


def setup_inferencer(bench_atten_type: str, torch_compile: bool) -> torch.nn.Module:
    """
    if torch_compile is True, return a compiled version of the attention function.
    """
    if bench_atten_type == "FA3":
        return (
            torch.compile(memory_efficient_attention)
            if torch_compile
            else memory_efficient_attention
        )

    elif bench_atten_type == "VSA":
        return torch.compile(video_sparse_attn) if torch_compile else video_sparse_attn

    elif bench_atten_type == "VSA_flex":
        return torch.compile(VSA_attention_flex) if torch_compile else VSA_attention_flex

    else:
        raise NotImplementedError("Unknown attention type")


def run_warmup(
    inferencer: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    og: torch.Tensor,
    active_kv_num: int,
    bench_atten_type: str,
):
    for _ in range(2):
        o = run_inference(inferencer, q, k, v, active_kv_num, bench_atten_type)
        _ = run_backward(o, og, bench_atten_type)


def collect_trace(
    inferencer,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    og: torch.Tensor,
    active_kv_num: int,
    bench_atten_type: str,
    num_iters: int,
    trace_file_path_dir: str,
    trace_file_name: str,
    logger: logging.Logger,
):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    with profiler_or_nullcontext(
        enabled=True,
        trace_file_path_dir=trace_file_path_dir,
        trace_file_name=trace_file_name,
        with_stack=True,
        record_shapes=True,
    ):
        for _ in range(num_iters):
            o = run_inference(inferencer, q, k, v, active_kv_num, bench_atten_type)
            _ = run_backward(o, og, bench_atten_type)

    logger.info(f"Trace is saved at: {trace_file_path_dir}/{trace_file_name}.json")


def bench_latency(
    inferencer: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    og: torch.Tensor,
    active_kv_num: int,
    bench_atten_type: str,
    num_iters: int,
    logger: logging.Logger,
):
    start_event = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_event = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    torch.cuda.synchronize()

    for i in range(num_iters):
        o = run_inference(inferencer, q, k, v, active_kv_num, bench_atten_type)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        start_event[i].record()
        _ = run_backward(o, og, bench_atten_type)
        end_event[i].record()
        torch.cuda.synchronize()

    times = np.array([s.elapsed_time(e) for s, e in zip(start_event, end_event)])
    elapsed_time = np.mean(times) * 1.0e-3
    logger.info(f"Average latency: {elapsed_time:.4f} s")


def benchmark_attention_bwd(
    logger: logging.Logger,
    trace_file_path_dir: str,
    trace_file_name: str = "video_sparse_attn_trace",
    trace_enable: bool = True,
    test_latency: bool = True,
    torch_compile: bool = False,
    num_iters: int = 100,
    batch_size: int = 1,
    seq_len: int = 65536,
    num_heads: int = 48,
    head_dim: int = 64,
    bench_atten_type: str = "FA3",
    topk: float = 0.125,
):
    q, k, v, og = prepare_tensors(
        bench_atten_type, batch_size, num_heads, seq_len, head_dim
    )

    kv_num: int = seq_len // BLOCK_VOLUME
    active_kv_num: int = int(topk * kv_num)

    inferencer = setup_inferencer(bench_atten_type, torch_compile)

    run_warmup(inferencer, q, k, v, og, active_kv_num, bench_atten_type)

    if trace_enable:
        collect_trace(
            inferencer=inferencer,
            q=q,
            k=k,
            v=v,
            og=og,
            active_kv_num=active_kv_num,
            bench_atten_type=bench_atten_type,
            num_iters=num_iters,
            trace_file_path_dir=trace_file_path_dir,
            trace_file_name=trace_file_name,
            logger=logger,
        )

    if test_latency:
        bench_latency(
            inferencer=inferencer,
            q=q,
            k=k,
            v=v,
            og=og,
            active_kv_num=active_kv_num,
            bench_atten_type=bench_atten_type,
            num_iters=num_iters,
            logger=logger,
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Benchmark Video Sparse Attention Kernels"
    )
    parser.add_argument(
        "--bench_atten_type",
        type=str,
        choices=["FA3", "VSA", "VSA_flex"],
        default="VSA",
        help="Which attention variant to benchmark",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--head_dim", type=int, default=64, help="Head dimension")
    parser.add_argument(
        "--topk",
        type=float,
        default=0.125,
        help="Ratio of kv blocks each q block attends to",
    )
    parser.add_argument(
        "--seq_len", type=int, default=65536, help="Sequence length to benchmark"
    )
    parser.add_argument("--test_latency", action="store_true", help="Test latency")
    parser.add_argument(
        "--trace_enable", action="store_true", help="Enable CPU and GPU traces"
    )
    parser.add_argument(
        "--torch_compile", action="store_true", help="Enable torch compile"
    )
    parser.add_argument(
        "--trace_file_path_dir", type=str, default="./", help="trace file path"
    )
    parser.add_argument(
        "--trace_file_name",
        type=str,
        default="video_sparse_attn_trace",
        help="trace file name",
    )
    parser.add_argument(
        "--num_iters", type=int, default=50, help="Number of iterations to run"
    )
    parser.add_argument(
        "--log_file_name",
        type=str,
        default="/home/boxunxu/github_repo/FastVideo-fork/FastVideo/csrc/attn/benchmarks/benchmark_trace_and_latency/benchmark_results.log",
        help="Log file name",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=args.log_file_name,
        filemode="a",
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Benchmark Arguments: {args}")

    benchmark_attention_bwd(
        logger=logger,
        trace_enable=args.trace_enable,
        trace_file_path_dir=args.trace_file_path_dir,
        trace_file_name=args.trace_file_name,
        test_latency=args.test_latency,
        torch_compile=args.torch_compile,
        num_iters=args.num_iters,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        bench_atten_type=args.bench_atten_type,
        topk=args.topk,
    )


if __name__ == "__main__":
    main()
