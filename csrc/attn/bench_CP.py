
import torch
import torch.nn.functional as F
from xformers.ops import fmha, memory_efficient_attention
from vsa import video_sparse_attn
from flex_attn_vsa import VSA_attention_flex
import numpy as np

def FA3(q, k, v):
    # FA3
    output = memory_efficient_attention(q, k, v, attn_bias=None)
    return output


def VSA(q,k,v):
    # VSA
    output_VSA = video_sparse_attn(q, k, v, topk=topk_blocks, block_size=(4,4,4), compress_attn_weight=0.0)
    return output_VSA


def VSA_flex(q,k,v):
    # VSA-flex
    output_flex = VSA_attention_flex(q,k,v, topk=topk_blocks, block_size=(4,4,4), compress_attn_weight=0.0)
    return output_flex


# setup random seed
torch.manual_seed(42)

# random inputs
batch_size = 1
# seq_len = 36864
# seq_len = 73728
seq_len = 147456
head_num = 12
head_dim = 128
block_size = 64
CP = 2 # context parallelism
q = torch.randn(batch_size, seq_len//CP, head_num, head_dim, requires_grad=True, dtype=torch.bfloat16, device="cuda")
k = torch.randn(batch_size, seq_len, head_num, head_dim, requires_grad=True, dtype=torch.bfloat16, device="cuda")
v = torch.randn(batch_size, seq_len, head_num, head_dim, requires_grad=True, dtype=torch.bfloat16, device="cuda")
topk_blocks = int(0.125 * seq_len/block_size) # select all blocks, causing dense attention
print("topk_blocks:", topk_blocks)
q, k, v = q.permute(0,2,1,3).contiguous(), k.permute(0,2,1,3).contiguous(), v.permute(0,2,1,3).contiguous()
# fwd
for i in range(2):
    # output_FA3 = FA3(q, k, v)
    _ = VSA(q,k,v)
    # _ = VSA_flex(q,k,v)

num_iters = 500
start_event = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
end_event = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
torch.cuda.synchronize()

for i in range(num_iters):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    start_event[i].record()
    # output_FA3 = FA3(q, k, v)
    # # output_VSA = VSA(q.permute(0,2,1,3).contiguous(), k.permute(0,2,1,3).contiguous(), v.permute(0,2,1,3).contiguous()).permute(0,2,1,3)
    _ = VSA(q,k,v)
    # _ = VSA_flex(q,k,v)
    # output_VSA_flex = VSA_flex(q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3)).permute(0,2,1,3)
    end_event[i].record()
    torch.cuda.synchronize()

times = np.array([s.elapsed_time(e) for s, e in zip(start_event, end_event)])
elapsed_time = np.mean(times) * 1.0e-3
print(f"Average latency: {elapsed_time:.4f} s")



# Verify the fwd consistency with FA3
# print("@fwd, consistent between FA3 and FA3 and VSA:", torch.allclose(output_FA3, output_VSA, atol=3e-03, equal_nan=False))
# print("@fwd, consistent between FA3 and FA3 and VSA:", torch.allclose(output_FA3, output_VSA_flex, atol=3e-03, equal_nan=False))
