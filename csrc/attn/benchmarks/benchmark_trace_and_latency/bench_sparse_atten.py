import argparse
import contextlib
import logging

import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity  # pyre-ignore

# Import video sparse attention
from vsa import video_sparse_attn
from xformers.ops import fmha, memory_efficient_attention

# use flash attention v3
fmha._set_use_fa3(True)

# new init
# /home/boxunxu/packages/conda/lib/python3.10/site-packages/vsa-0.0.1-py3.10-linux-x86_64.egg/vsa/__init__.py


# VSA VARIABLES
BLOCK_SIZE = (4, 4, 4)
BLOCK_VOLUME = BLOCK_SIZE[0] * BLOCK_SIZE[1] * BLOCK_SIZE[2]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Benchmark Video Sparse Attention Kernels"
    )
    parser.add_argument(
        "--bench_atten_type",
        type=str,
        default="VSA",
        help="FA3, VSA, VSA_opt1, VSA_opt2, FlexAtten_VSA",
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
        default="benchmark_results.log",
        help="Log file name",
    )

    return parser.parse_args()


def create_qkv_tensors(batch, num_heads, seq_len, head_dim, device="cuda"):
    """Create random input tensors for attention"""

    q = torch.randn(
        batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    k = torch.randn(
        batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    v = torch.randn(
        batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )

    return q, k, v


def create_tiled_qkv_tensors(batch, num_heads, seq_len, head_dim, device="cuda"):
    """Create random input tensors for attention"""

    q = torch.randn(
        batch, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device
    )
    k = torch.randn(
        batch, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device
    )
    v = torch.randn(
        batch, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device
    )

    return q, k, v


def profiler_or_nullcontext(
    enabled: bool,
    with_stack: bool = False,
    trace_file_path_dir: str = "~",
    trace_file_name: str = "video_sparse_attn_trace",
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


def benchmark_attention(  # noqa: C901
    logger: logging.Logger,
    trace_enable: bool = True,
    trace_file_path_dir: str = "~",
    trace_file_name: str = "video_sparse_attn_trace",
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
    def run_inference(inferencer, q, k, v, active_kv_num, bench_atten_type):
        if bench_atten_type == "FA3":
            return inferencer(q, k, v)
        elif bench_atten_type in ["VSA", "VSA_opt1", "VSA_opt2"]:
            return inferencer(
                q,
                k,
                v,
                topk=active_kv_num,
                block_size=BLOCK_SIZE,
                mode=bench_atten_type,
            )
        else:
            raise NotImplementedError("Unknown attention type")

    def prepare_tensors(bench_atten_type, batch_size, num_heads, seq_len, head_dim):
        if bench_atten_type == "FA3":
            return create_qkv_tensors(
                batch_size, num_heads, seq_len, head_dim, device="cuda"
            )
        else:
            return create_tiled_qkv_tensors(
                batch_size, num_heads, seq_len, head_dim, device="cuda"
            )

    def setup_inferencer(bench_atten_type, torch_compile):
        if bench_atten_type == "FA3":
            return (
                torch.compile(memory_efficient_attention)
                if torch_compile
                else memory_efficient_attention
            )
        elif bench_atten_type in ["VSA", "VSA_opt1", "VSA_opt2"]:
            return (
                torch.compile(video_sparse_attn) if torch_compile else video_sparse_attn
            )
        else:
            raise NotImplementedError("Unknown attention type")

    def warmup_stage(inferencer, q, k, v, active_kv_num, bench_atten_type):
        for _ in range(2):
            _ = run_inference(inferencer, q, k, v, active_kv_num, bench_atten_type)

    def trace_execution(inferencer, q, k, v, active_kv_num, bench_atten_type):
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
                _ = run_inference(inferencer, q, k, v, active_kv_num, bench_atten_type)

        logger.info(f"Trace is saved at: {trace_file_path_dir}/{trace_file_name}.json")

    def test_latency_execution(inferencer, q, k, v, active_kv_num, bench_atten_type):
        start_event = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_event = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

        torch.cuda.synchronize()

        for i in range(num_iters):
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            start_event[i].record()
            _ = run_inference(inferencer, q, k, v, active_kv_num, bench_atten_type)
            end_event[i].record()
            torch.cuda.synchronize()

        times = np.array([s.elapsed_time(e) for s, e in zip(start_event, end_event)])
        elapsed_time = np.mean(times) * 1.0e-3
        logger.info(f"Average latency: {elapsed_time:.4f} s")

    def execute_benchmark():
        q, k, v = prepare_tensors(
            bench_atten_type, batch_size, num_heads, seq_len, head_dim
        )

        kv_num = seq_len // BLOCK_VOLUME
        active_kv_num = int(topk * kv_num)

        inferencer = setup_inferencer(bench_atten_type, torch_compile)

        warmup_stage(inferencer, q, k, v, active_kv_num, bench_atten_type)

        if trace_enable:
            trace_execution(inferencer, q, k, v, active_kv_num, bench_atten_type)

        if test_latency:
            test_latency_execution(inferencer, q, k, v, active_kv_num, bench_atten_type)

    execute_benchmark()


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

    benchmark_attention(
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
