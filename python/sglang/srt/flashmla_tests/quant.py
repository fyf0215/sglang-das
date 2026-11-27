"""Minimal quant utils for flash MLA torch backend.

Wraps the existing ds4 / dsa dequant functions so the ``hip_flash_mla``
torch backend has a functional ``flashmla_tests.quant`` module.
"""

import enum
import torch


class FP8KVCacheLayout(enum.Enum):
    """Layout IDs understood by :func:`dequantize_k_cache`."""

    MODEL1_FP8Sparse = 0
    MODEL1_FP8Dense = 1


def dequantize_k_cache(
    quant_k_cache: torch.Tensor, layout: FP8KVCacheLayout
) -> torch.Tensor:
    """Dequantize an fp8-block-quantized K cache into bf16.

    Format:  [..., dv fp8 nope | (dv/128)*4 float32 scales | rope_bf16 bytes]
    where dv=512 (head dim nope), rope=128 bf16 bytes (64 values).
    The hard-coded 656-byte assert in the dsa impl doesn't account for
    model variants with larger token payloads; compute dv/rope/tile
    dynamically from the actual tensor shape.
    """
    original_ndim = quant_k_cache.ndim
    if original_ndim == 3:
        quant_k_cache = quant_k_cache.unsqueeze(1)
    N, B, _, Q = quant_k_cache.shape  # blocks, block_size, 1, bytes_per_token

    tile_size = 128
    # The rope region occupies the remainder after nope+scales.
    num_tiles = 4  # 512 / 128
    dv = num_tiles * tile_size  # 512
    scale_bytes = num_tiles * 4   # 16
    rope_elements = Q - dv - scale_bytes  # remaining bytes → bf16 count / 2
    assert rope_elements % 2 == 0, f"rope bytes ({rope_elements}) not even → dim_quant={Q}"
    rope_dim = rope_elements // 2  # number of bf16 values

    quant_k_cache = quant_k_cache.view(N, B, -1)
    assert quant_k_cache.shape[-1] == Q

    # Split into nope (fp8), scales (fp32), rope (bf16)
    input_nope = quant_k_cache[..., :dv]
    input_scale = quant_k_cache[..., dv : dv + scale_bytes].view(torch.float32)
    input_rope = quant_k_cache[..., dv + scale_bytes :].view(torch.bfloat16)

    result = torch.empty(
        (N, B, dv + rope_dim), dtype=torch.bfloat16, device=quant_k_cache.device
    )
    result[..., dv:] = input_rope

    for tile_idx in range(num_tiles):
        cur_nope = input_nope[
            ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
        ].to(torch.float32)
        cur_scales = input_scale[..., tile_idx].unsqueeze(-1)
        result[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
            cur_nope * cur_scales
        )

    if original_ndim == 3:
        return result.squeeze(1)
    return result.view(N, B, 1, -1)
