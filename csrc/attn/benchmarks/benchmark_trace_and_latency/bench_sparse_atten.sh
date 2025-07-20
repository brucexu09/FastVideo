# Sweep over seq_lens to reproduce VSA kernel performance
## Setup parameters
trace_file_names=()
kernel_types=("FA3" "VSA" "VSA_opt1" "VSA_opt2")
seq_lens=(8192 16384 32768 65536 131072)
head_dim=64
num_heads=48
num_iters=100
batch_size=1
topk=0.125
log_file_name="benchmark_diff_seq.log"

## Generate trace_file_names based on kernel_types and seq_lens
for kernel in "${kernel_types[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
        trace_file_names+=("${kernel}_N${seq_len}")
    done
done

## Run bench_sparse_atten.py
for ((i=0; i<${#kernel_types[@]}; i++)); do
    for ((j=0; j<${#seq_lens[@]}; j++)); do
        python bench_sparse_atten.py \
            --trace_file_path_dir "./gpu_traces" \
            --trace_file_name "${trace_file_names[$((i * ${#seq_lens[@]} + j))]}" \
            --trace_enable \
            --test_latency --num_iters "$num_iters" --batch_size "$batch_size" \
            --seq_len "${seq_lens[$j]}" --num_heads "$num_heads" --head_dim "$head_dim" \
            --bench_atten_type "${kernel_types[$i]}" --topk "$topk" \
            --log_file_name "$log_file_name"
    done
done
