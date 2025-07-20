from functools import lru_cache
from logging import getLogger
from typing import Any, cast, List, Optional, Tuple

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# compile create_block_mask and flex_attention
cbm_compiled = torch.compile(create_block_mask)
flex_compiled = torch.compile(flex_attention)


try:
    import xformers
    from xformers.ops import (
        AttentionBias,
        AttentionOp,
        fmha,
        memory_efficient_attention,
    )

    print("using efficient attention")

except (ImportError, ModuleNotFoundError):
    AttentionOp = Any
    AttentionBias = Any
    fmha = None
    print("no efficient attention")


def flex_attention_with_block_mask(query:torch.tensor,
                                    key:torch.tensor,
                                    value:torch.tensor,
                                    mask:torch.tensor,
                                    BS:int) -> torch.Tensor:
    """
    FlexAttention-style sparse block attention.
    q: torch.tensor, [B, H, Nq, d]
    k: torch.tensor, [B, H, Nk, d]
    v: torch.tensor, [B, H, Nv, d]
    mask: torch.tensor, [B, H, nq, nk]
    BS: block size
    """

    B, H, Nq, d = query.shape
    _, _, Nk, _ = key.shape

    def search_mask(b, h, q_idx, k_idx, BS=BS, mask=mask):
        q_blk_idx = q_idx // BS
        k_blk_idx = k_idx // BS
        return (mask[b, h, q_blk_idx, k_blk_idx] > 0.).to(torch.bool) # indicate this is an active block

    block_mask = cbm_compiled(search_mask, B, H, Nq, Nk, device="cuda")
    output = flex_compiled(query, key, value, block_mask=block_mask)

    return output


def torch_attention(q, k, v) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q: torch.tensor, [B, H, Nq, d]
    k: torch.tensor, [B, H, Nk, d]
    v: torch.tensor, [B, H, Nv, d]
    """
    QK = torch.matmul(q, k.transpose(-2, -1))
    QK /= (q.size(-1)**0.5)

    # Causal mask removed since causal is always false
    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, v)
    return output, QK


def VSA_attention_flex(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    p: float = 0.0,
    op: Optional[AttentionOp] = None,
    use_fp8: bool = False,
    topk: int = 1,
    block_size: list[int] = [4, 4, 4],  # (B_T, B_H, B_W)
    compress_attn_weight=0.0,
):
    """
    query: torch.Tensor, [batch_size, num_heads, query_seq_len, head_dim]
    key: torch.Tensor, [batch_size, num_heads, kv_seq_len, head_dim]
    value: torch.Tensor, [batch_size, num_heads, kv_seq_len, head_dim]

    Definitions:
    Hybrid Attention Module with 3D block
    block_size: (block_size_t, block_size_h, block_size_w)
    block_num: (num_blocks_t, num_blocks_h, num_blocks_w)
    """

    # check shape of query, key, value
    batch_size, num_heads, query_seq_len, head_dim = query.shape
    _, _, kv_seq_len, _ = key.shape

    # block volume
    block_volume = block_size[0] * block_size[1] * block_size[2]

    # here, we assume qkv has been rearranged
    q_compress = query.view(
        batch_size, num_heads, query_seq_len // block_volume, block_volume, head_dim
    ).mean(
        dim=3
    )  # comp_q (B, H, n_q, d)
    k_compress = key.view(
        batch_size, num_heads, kv_seq_len // block_volume, block_volume, head_dim
    ).mean(
        dim=3
    )  # comp_k (B, H, n_kv, d)
    v_compress = value.view(
        batch_size, num_heads, kv_seq_len // block_volume, block_volume,  head_dim
    ).mean(
        dim=3
    )  # comp_v (B, H, n_kv, d)

    output_compress, block_attn_score = torch_attention(
        q_compress,
        k_compress,
        v_compress,
    )  # (B H nq d)(B H nq nk)

    output_coarse = (
        output_compress.permute(0, 2, 1, 3)
        .unsqueeze(2)
        .repeat(1, 1, block_volume, 1, 1)
        .view(query.shape)
    ) # B H Nq d

    # print("output_coarse.shape", output_coarse.shape)

    # Topk Selection
    _, topk_indices = torch.topk(block_attn_score, topk, dim=-1)
    masked_A_compress = torch.zeros_like(block_attn_score)  # B H nq nk
    masked_A_compress.scatter_(-1, topk_indices, 1)  # B H nq nk

    output_fine = flex_attention_with_block_mask(
        query=query,
        key=key,
        value=value,
        mask=masked_A_compress,
        BS=block_volume,
    ) # B H Nq d

    # Combine
    # print("output_fine.shape", output_fine.shape)
    output = compress_attn_weight * output_coarse + (1 - compress_attn_weight) * output_fine # B H Nq d

    return output
