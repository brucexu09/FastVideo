
import torch
import torch.nn.functional as F
from xformers.ops import fmha, memory_efficient_attention
from vsa import video_sparse_attn
# from flex_attn_vsa import VSA_attention_flex

# from logging import getLogger
# from typing import Tuple

# import torch
# from torch.nn.attention.flex_attention import create_block_mask, flex_attention

# logger = getLogger()

# # Compile create_block_mask and flex_attention to manage memory efficiently (avoid OOM)
# cbm_compiled = torch.compile(create_block_mask)
# flex_compiled = torch.compile(flex_attention)


# def block_sparse_attention_flex(
#     query: torch.Tensor,
#     key: torch.Tensor,
#     value: torch.Tensor,
#     mask: torch.Tensor,
#     block_volume: int,
# ) -> torch.Tensor:
#     """
#     FlexAttention-based block sparse attention.

#     Args:
#         query: Input query tensor of shape [batch_size, num_heads, q_seq_len, head_dim]
#         key: Input key tensor of shape [batch_size, num_heads, kv_seq_len, head_dim]
#         value: Input value tensor of shape [batch_size, num_heads, kv_seq_len, head_dim]
#         mask: Block sparsity mask of shape [batch_size, num_heads, q_block_len, kv_block_len]
#         block_volume: Volume of each attention block

#     Returns:
#         Output tensor of shape [batch_size, num_heads, q_seq_len, head_dim]
#     """

#     batch_size, num_heads, q_seq_len, head_dim = query.shape
#     kv_seq_len = key.shape[2]

#     def search_mask(b, h, q_idx, k_idx, block_volume=block_volume, mask=mask):
#         q_blk_idx = q_idx // block_volume
#         k_blk_idx = k_idx // block_volume
#         return (mask[b, h, q_blk_idx, k_blk_idx] > 0.0).to(torch.bool)

#     block_mask = cbm_compiled(
#         search_mask, batch_size, num_heads, q_seq_len, kv_seq_len, device="cuda"
#     )
#     output = flex_compiled(query, key, value, block_mask=block_mask)

#     return output


# def torch_attention(
#     q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Standard scaled dot-product attention.

#     Args:
#         q: Query tensor of shape [batch_size, num_heads, q_seq_len, head_dim]
#         k: Key tensor of shape [batch_size, num_heads, kv_seq_len, head_dim]
#         v: Value tensor of shape [batch_size, num_heads, kv_seq_len, head_dim]

#     Returns:
#         Tuple of:
#             - Output tensor of shape [batch_size, num_heads, q_seq_len, head_dim]
#             - Attention scores of shape [batch_size, num_heads, q_seq_len, kv_seq_len]
#     """
#     QK = torch.matmul(q, k.transpose(-2, -1))
#     attention_scores = QK / q.size(-1) ** 0.5

#     attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
#     output = torch.matmul(attention_probs, v)
#     return output, attention_probs


# def _VSA_flex(
#     query: torch.Tensor,
#     key: torch.Tensor,
#     value: torch.Tensor,
#     topk_r: float = 0.125,
#     block_size: tuple[int, int, int] = (4, 4, 4),
#     compress_attn_weight: float = 0.0,
# ) -> torch.Tensor:
#     """
#     Hybrid Attention Module with 3D block sparsity.

#     This function implements a hybrid attention mechanism that combines coarse-grained
#     block attention with fine-grained sparse attention for efficient computation.

#     Args:
#         query: Input query tensor of shape [batch_size, num_heads, q_seq_len, head_dim]
#         key: Input key tensor of shape [batch_size, num_heads, kv_seq_len, head_dim]
#         value: Input value tensor of shape [batch_size, num_heads, kv_seq_len, head_dim]
#         topk_r: Top-k ratio for selecting row-wise sparse blocks (default: 0.125)
#         block_size: 3D block dimensions as (block_size_t, block_size_h, block_size_w) (default: (4, 4, 4))
#         compress_attn_weight: Weight for the coarse attention component (default: 0.0)

#     Returns:
#         Output tensor of shape [batch_size, num_heads, q_seq_len, head_dim]
#     """
#     # Extract shape information
#     batch_size, num_heads, q_seq_len, head_dim = query.shape
#     _, _, kv_seq_len, _ = key.shape

#     # Calculate block volume
#     block_volume = block_size[0] * block_size[1] * block_size[2]

#     # Compress queries, keys, and values by averaging within blocks
#     q_compress = query.view(
#         batch_size, num_heads, q_seq_len // block_volume, block_volume, head_dim
#     ).mean(dim=3)  # (batch_size, num_heads, q_block_len, head_dim)
#     k_compress = key.view(
#         batch_size, num_heads, kv_seq_len // block_volume, block_volume, head_dim
#     ).mean(dim=3)  # (batch_size, num_heads, kv_block_len, head_dim)
#     v_compress = value.view(
#         batch_size, num_heads, kv_seq_len // block_volume, block_volume, head_dim
#     ).mean(dim=3)  # (batch_size, num_heads, kv_block_len, head_dim)

#     # Compute coarse-grained block attention
#     output_compress, block_attn_score = torch_attention(
#         q_compress,
#         k_compress,
#         v_compress,
#     )  # (batch_size, num_heads, q_block_len, head_dim), (batch_size, num_heads, q_block_len, kv_block_len)

#     # Expand coarse output to original dimensions
#     output_coarse = (
#         output_compress.unsqueeze(3).repeat(1, 1, 1, block_volume, 1).view(query.shape)
#     )  # (batch_size, num_heads, q_seq_len, head_dim)

#     # Row-wise top-k block selection
#     topk = int(topk_r * kv_seq_len // block_volume)
#     _, topk_indices = torch.topk(block_attn_score, topk, dim=-1)
#     block_sparsity_mask = torch.zeros_like(
#         block_attn_score
#     )  # (batch_size, num_heads, q_block_len, kv_block_len)
#     block_sparsity_mask.scatter_(
#         -1, topk_indices, 1
#     )  # (batch_size, num_heads, q_block_len, kv_block_len)

#     # Compute fine-grained sparse attention
#     output_fine = block_sparse_attention_flex(
#         query=query,
#         key=key,
#         value=value,
#         mask=block_sparsity_mask,
#         block_volume=block_volume,
#     )  # (batch_size, num_heads, q_seq_len, head_dim)

#     # Combine coarse and fine outputs
#     output = (
#         compress_attn_weight * output_coarse + output_fine
#     )  # (batch_size, num_heads, q_seq_len, head_dim)

#     return output




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
    output_VSA = video_sparse_attn(q, k, v, topk=int(0.125*topk_blocks), block_size=(4,4,4), compress_attn_weight=0.0)
    return output_VSA

def VSA_flex(q,k,v):
    # VSA-flex
    # output_flex = VSA_attention_flex(q,k,v, topk=topk_blocks, block_size=(4,4,4), compress_attn_weight=0.0)
    output_flex = _VSA_flex(query=q,key=k,value=v, topk_r=0.125, block_size=(4,4,4), compress_attn_weight=0.0)
    return output_flex

# Verify correctness

# setup random seed
torch.manual_seed(42)

# random inputs
batch_size = 1
# seq_len = 8192
seq_len = 16384
head_num = 12
head_dim = 64
block_size = 64
CP = 2 # context parallelism
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
print("@fwd, consistent between FA3 and VSA:", torch.allclose(output_FA3, output_VSA, atol=3e-03, equal_nan=False))
print("@fwd, consistent between FA3 and VSA_flex:", torch.allclose(output_FA3, output_VSA_flex, atol=3e-03, equal_nan=False))
print("@fwd, consistent between VSA and VSA_flex:", torch.allclose(output_VSA, output_VSA_flex, atol=3e-03, equal_nan=False))

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

print("q_grad@bwd, consistent between VSA and VSA_flex:", torch.allclose(q_grad_VSA, q_grad_VSA_flex, atol=5e-3, equal_nan=False))
print("k_grad@bwd, consistent between VSA and VSA_flex:", torch.allclose(k_grad_VSA, k_grad_VSA_flex, atol=5e-3, equal_nan=False))
print("v_grad@bwd, consistent between VSA and VSA_flex:", torch.allclose(v_grad_VSA, v_grad_VSA_flex, atol=5e-3, equal_nan=False))
