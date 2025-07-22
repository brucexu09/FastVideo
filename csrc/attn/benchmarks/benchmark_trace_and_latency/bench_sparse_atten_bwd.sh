#!/bin/bash
# shellcheck shell=bash

# Create a temporary directory for GPU traces
trace_file_path_dir=$(mktemp -d)
echo "Using temporary directory for GPU traces: $trace_file_path_dir"

# (1) Sweep over different attention kernels and sequence lengths

## Setup Parameters
bench_atten_types=("FA3" "VSA" "VSA_flex")
seq_lens=(36864 73728 147456 294912 9216 4096 16384)
num_iters=2
batch_size=1
num_heads=12
head_dim=128
topk=0.125
log_file_name="benchmark_diff_seq.log"

## Generate trace_file_names based on kernel_types and seq_lens
trace_file_names=()
for kernel in "${bench_atten_types[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
        trace_file_names+=("${kernel}_N${seq_len}")
    done
done

## Run benchmarks
for i in "${!bench_atten_types[@]}"; do
  for j in "${!seq_lens[@]}"; do
    python bench_sparse_atten_bwd.py \
      --trace_enable \
      --trace_file_path_dir "$trace_file_path_dir" \
      --trace_file_name "${trace_file_names[$((i * ${#seq_lens[@]} + j))]}" \
      --test_latency \
      --torch_compile \
      --num_iters "$num_iters" \
      --batch_size "$batch_size" \
      --seq_len "${seq_lens[$j]}" \
      --num_heads "$num_heads" \
      --head_dim "$head_dim" \
      --topk "$topk" \
      --bench_atten_type "${bench_atten_types[$i]}" \
      --log_file_name "$log_file_name"
  done
done


# (2) Sweep over different attention kernels and different topk values

## Setup parameters
bench_atten_types=("FA3" "VSA" "VSA_flex")
seq_len=73728
num_iters=2
batch_size=1
num_heads=12
head_dim=128
topk_list=(1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.063 0.047 0.032 0.01)
log_file_name="benchmark_diff_topk.log"

## Generate trace_file_names based on kernel_types and topk_list
trace_file_names=()
for kernel in "${bench_atten_types[@]}"; do
    for topk in "${topk_list[@]}"; do
        trace_file_names+=("${kernel}_K${topk}")
    done
done


## Run benchmarks
for ((i=0; i<${#bench_atten_types[@]}; i++)); do
    for ((j=0; j<${#topk_list[@]}; j++)); do
        CUDA_VISIBLE_DEVICES=4 python bench_sparse_atten_bwd.py \
            --trace_file_path_dir "$trace_file_path_dir" \
            --trace_file_name "${trace_file_names[$((i * ${#topk_list[@]} + j))]}" \
            --trace_enable --torch_compile \
            --test_latency \
            --num_iters "$num_iters" \
            --batch_size "$batch_size" \
            --seq_len "$seq_len" \
            --num_heads "$num_heads" \
            --head_dim "$head_dim" \
            --bench_atten_type "${bench_atten_types[$i]}" \
            --topk "${topk_list[$j]}" \
            --log_file_name "$log_file_name"
    done
done

# python bench_aparse_atten_bwd.py --bench_atten_type "FA3" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 36864 --num_heads 12 --head_dim 128 --topk 0.125
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 36864 --num_heads 12 --head_dim 128 --topk 0.125
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA_flex" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 36864 --num_heads 12 --head_dim 128 --topk 0.125

# python bench_aparse_atten_bwd.py --bench_atten_type "FA3" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.125
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.125
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA_flex" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.125

# python bench_aparse_atten_bwd.py --bench_atten_type "FA3" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 147456 --num_heads 12 --head_dim 128 --topk 0.125
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 147456 --num_heads 12 --head_dim 128 --topk 0.125
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA_flex" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 147456 --num_heads 12 --head_dim 128 --topk 0.125

# python bench_aparse_atten_bwd.py --bench_atten_type "FA3" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 294912 --num_heads 12 --head_dim 128 --topk 0.125
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 294912 --num_heads 12 --head_dim 128 --topk 0.125
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA_flex" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 294912 --num_heads 12 --head_dim 128 --topk 0.125

# python bench_aparse_atten_bwd.py --bench_atten_type "FA3" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 9216 --num_heads 12 --head_dim 128 --topk 0.125
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 9216 --num_heads 12 --head_dim 128 --topk 0.125
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA_flex" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 9216 --num_heads 12 --head_dim 128 --topk 0.125

# python bench_aparse_atten_bwd.py --bench_atten_type "FA3" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 4096 --num_heads 12 --head_dim 128 --topk 0.125
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 4096 --num_heads 12 --head_dim 128 --topk 0.125
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA_flex" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 4096 --num_heads 12 --head_dim 128 --topk 0.125

# python bench_aparse_atten_bwd.py --bench_atten_type "FA3" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 16384 --num_heads 12 --head_dim 128 --topk 0.125
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 16384 --num_heads 12 --head_dim 128 --topk 0.125
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA_flex" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 16384 --num_heads 12 --head_dim 128 --topk 0.125


# python bench_aparse_atten_bwd.py --bench_atten_type "FA3" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.125
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 1
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.9
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.8
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.7
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.5
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.4
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.3
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.2
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.1
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.063
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.047
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.031
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.016


# python bench_aparse_atten_bwd.py --bench_atten_type "VSA" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.6
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA_flex" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.6


# python bench_aparse_atten_bwd.py --bench_atten_type "VSA_flex" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 1
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA_flex" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.9
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA_flex" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.8
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA_flex" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.7
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA_flex" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.5
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA_flex" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.4
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA_flex" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.3
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA_flex" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.2
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA_flex" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.1
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA_flex" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.063
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA_flex" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.047
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA_flex" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.031
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA_flex" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.016
# python bench_aparse_atten_bwd.py --bench_atten_type "VSA_flex" --test_latency --torch_compile --num_iters 200 --batch_size 1 --seq_len 73728 --num_heads 12 --head_dim 128 --topk 0.125
