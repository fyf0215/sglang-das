"""Minimal test-infrastructure data-classes for flash MLA torch backend."""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class ExtraTestParamForDecode:
    b: int
    is_varlen: str
    have_zero_seqlen_k: str
    extra_s_k: str
    extra_topk: str
    extra_block_size: str
    have_extra_topk_length: str


@dataclass
class TestParam:
    s_q: int
    s_kv: str
    topk: str
    h_q: int
    h_kv: int
    d_qk: int
    d_v: int
    decode: ExtraTestParamForDecode
    seed: int
    check_correctness: bool
    is_all_indices_invalid: bool
    num_runs: int
    have_attn_sink: bool
    have_topk_length: bool


@dataclass
class KVScope:
    t: str
    cache_seqlens: str
    block_table: str
    blocked_k: torch.Tensor
    blocked_k_quantized: torch.Tensor
    abs_indices: str
    indices_in_kvcache: Optional[torch.Tensor]
    topk_length: Optional[torch.Tensor]


@dataclass
class TestcaseForDecode:
    p: str
    q: torch.Tensor
    attn_sink: Optional[torch.Tensor]
    sm_scale: Optional[float]
    kv_scope: KVScope
    extra_kv_scope: Optional[KVScope]
