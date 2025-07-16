
import torch
import torch.nn.functional as F
from xformers.ops import fmha, memory_efficient_attention
from vsa import video_sparse_attn
from flex_attn_vsa import VSA_attention_flex

def FA3(q, k, v):
    # FA3
    output = memory_efficient_attention(q, k, v, attn_bias=None)
    return output

def torch_attn(q, k, v, head_dim):
    # torch_attn
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)

def VSA(q,k,v):
    # VSA
    output_VSA = video_sparse_attn(q, k, v, topk=topk_blocks, block_size=(4,4,4), compress_attn_weight=0.0)
    return output_VSA

def VSA_flex(q,k,v):
    # VSA-flex
    output_flex = VSA_attention_flex(q,k,v, topk=topk_blocks, block_size=(4,4,4), compress_attn_weight=0.0)
    return output_flex

# Verify correctness

# setup random seed
torch.manual_seed(42)

# random inputs
batch_size = 1
seq_len = 4096
head_num = 12
head_dim = 64
block_size = 64
CP = 4 # context parallelism
q = torch.randn(batch_size, seq_len//CP, head_num, head_dim, requires_grad=True, dtype=torch.bfloat16, device="cuda")
k = torch.randn(batch_size, seq_len, head_num, head_dim, requires_grad=True, dtype=torch.bfloat16, device="cuda")
v = torch.randn(batch_size, seq_len, head_num, head_dim, requires_grad=True, dtype=torch.bfloat16, device="cuda")
topk_blocks = int(seq_len/block_size) # select all blocks, causing dense attention
print("topk_blocks:", topk_blocks)

# fwd
output_FA3 = FA3(q, k, v)
output_torch = torch_attn(q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3), head_dim).permute(0,2,1,3)
output_VSA = VSA(q.permute(0,2,1,3).contiguous(), k.permute(0,2,1,3).contiguous(), v.permute(0,2,1,3).contiguous()).permute(0,2,1,3)
output_VSA_flex = VSA_flex(q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3)).permute(0,2,1,3)

# Verify the fwd consistency with FA3
print("@fwd, consistent between FA3 and torch_attn:", torch.allclose(output_FA3, output_torch, atol=3e-03, equal_nan=False))
print("@fwd, consistent between FA3 and FA3 and VSA:", torch.allclose(output_FA3, output_VSA, atol=3e-03, equal_nan=False))
print("@fwd, consistent between FA3 and FA3 and VSA:", torch.allclose(output_FA3, output_VSA_flex, atol=3e-03, equal_nan=False))

# bwd
output_grad = torch.randn_like(output_FA3)
# save grad for FA3
output_FA3.backward(gradient=output_grad)
q_grad_FA3 = q.grad.clone()
k_grad_FA3 = k.grad.clone()
v_grad_FA3 = v.grad.clone()
q.grad = None
k.grad = None
v.grad = None

# save grad for torch attention
output_torch.backward(gradient=output_grad)
q_grad_torch = q.grad.clone()
k_grad_torch = k.grad.clone()
v_grad_torch = v.grad.clone()
q.grad = None
k.grad = None
v.grad = None

# save grad for VSA
output_VSA.backward(gradient=output_grad)
q_grad_VSA = q.grad.clone()
k_grad_VSA = k.grad.clone()
v_grad_VSA = v.grad.clone()
q.grad = None
k.grad = None
v.grad = None

# save grad for VSA_flex
output_VSA_flex.backward(gradient=output_grad)
q_grad_VSA_flex = q.grad.clone()
k_grad_VSA_flex = k.grad.clone()
v_grad_VSA_flex = v.grad.clone()
q.grad = None
k.grad = None
v.grad = None

# Verify the bwd consistency with FA3
print("q_grad@bwd, consistent between FA3 and torch_attn:", torch.allclose(q_grad_FA3, q_grad_torch, atol=5e-3, equal_nan=False))
print("k_grad@bwd, consistent between FA3 and torch_attn:", torch.allclose(k_grad_FA3, k_grad_torch, atol=5e-3, equal_nan=False))
print("v_grad@bwd, consistent between FA3 and torch_attn:", torch.allclose(v_grad_FA3, v_grad_torch, atol=5e-3, equal_nan=False))

print("q_grad@bwd, consistent between FA3 and VSA:", torch.allclose(q_grad_FA3, q_grad_VSA, atol=5e-3, equal_nan=False))
print("k_grad@bwd, consistent between FA3 and VSA:", torch.allclose(k_grad_FA3, k_grad_VSA, atol=5e-3, equal_nan=False))
print("v_grad@bwd, consistent between FA3 and VSA:", torch.allclose(v_grad_FA3, v_grad_VSA, atol=5e-3, equal_nan=False))

print("q_grad@bwd, consistent between FA3 and VSA_flex:", torch.allclose(q_grad_FA3, q_grad_VSA_flex, atol=5e-3, equal_nan=False))
print("k_grad@bwd, consistent between FA3 and VSA_flex:", torch.allclose(k_grad_FA3, k_grad_VSA_flex, atol=5e-3, equal_nan=False))
print("v_grad@bwd, consistent between FA3 and VSA_flex:", torch.allclose(v_grad_FA3, v_grad_VSA_flex, atol=5e-3, equal_nan=False))
