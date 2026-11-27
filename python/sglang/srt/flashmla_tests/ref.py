"""Minimal reference sparse-attention decode for flash MLA torch backend.

Implements the actual attention computation that tilelang/triton kernels
would perform, using PyTorch primitives.  Only the Decode (non-prefill)
path is currently needed here.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.flashmla_tests.lib import KVScope, TestcaseForDecode, TestParam


def _gather_blocked_k(
    blocked_k: torch.Tensor,
    indices_in_kvcache: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor],
) -> torch.Tensor:
    """Gather (and optionally mask) the sparse K cache per request.

    Parameters
    ----------
    blocked_k:  ``[num_blocks, block_size, 1, D]`` — per-block K cache.
    indices_in_kvcache: ``[B, max_topk]`` — sparse-index lookup table.
    topk_length: ``[B]`` — number of valid indices per batch entry.

    Returns
    -------
    k: ``[B, max_topk, D]``
    """
    B, Blk, _, D = blocked_k.shape
    blocked_k_2d = blocked_k.reshape(-1, D)  # [num_blocks * Blk, D]
    max_valid_idx = blocked_k_2d.shape[0] - 1

    if indices_in_kvcache is not None:
        # Clamp sentinel values (-1 or out-of-range) to 0 to avoid OOM.
        idx = indices_in_kvcache.clamp(0, max_valid_idx)
        batch_offsets = (torch.arange(B, device=blocked_k.device) * Blk).long()
        flat_indices = (idx + batch_offsets.unsqueeze(1)).flatten()
        k = blocked_k_2d.index_select(0, flat_indices).view(B, -1, D)

        # Zero out entries corresponding to sentinel indices (-1).
        if topk_length is not None:
            max_k = k.shape[1]
            valid_mask = (
                (indices_in_kvcache >= 0)
                & (torch.arange(max_k, device=k.device).unsqueeze(0) < topk_length.unsqueeze(1))
            )
        else:
            valid_mask = indices_in_kvcache >= 0
        k = k * valid_mask.unsqueeze(-1).to(k.dtype)
    else:
        k = blocked_k_2d.view(B, -1, D)

    return k


def _ref_attention_decode(
    q: torch.Tensor,   # [B, H, D]
    k: torch.Tensor,   # [B, S, D]
    sm_scale: float,
    attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Reference softmax attention: ``softmax(q @ k.T * scale) @ k``.

    For MLA (multi-head latent attention) the V head is the same as K.
    """
    q = q.unsqueeze(1) if q.ndim == 3 else q   # [B, 1, D]
    attn = torch.matmul(q, k.transpose(-1, -2)) * sm_scale  # [B, 1, S]
    if attn_sink is not None:
        attn = attn + attn_sink
    attn_weights = F.softmax(attn, dim=-1, dtype=torch.float32).to(k.dtype)
    out = torch.matmul(attn_weights, k)  # [B, 1, D]
    return out.squeeze(1), None           # (output, lse=None)


def ref_sparse_attn_decode(
    p: TestParam,
    t: TestcaseForDecode,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Reference sparse-attention decode — mirrors the tilelang/triton
    attention's numeric behaviour for the unit test that the torch backend
    re-runs."""
    kv = t.kv_scope
    k = _gather_blocked_k(
        kv.blocked_k, kv.indices_in_kvcache, kv.topk_length
    )

    out, lse = _ref_attention_decode(
        q=t.q,
        k=k,
        sm_scale=t.sm_scale or 1.0,
        attn_sink=t.attn_sink,
    )

    # Handle extra KV scope (dual-cache sparsity) if present.
    if t.extra_kv_scope is not None:
        ek = _gather_blocked_k(
            t.extra_kv_scope.blocked_k,
            t.extra_kv_scope.indices_in_kvcache,
            t.extra_kv_scope.topk_length,
        )
        e_out, _ = _ref_attention_decode(
            q=t.q,
            k=ek,
            sm_scale=t.sm_scale or 1.0,
            attn_sink=t.attn_sink,
        )
        out = out + e_out

    return out, lse
