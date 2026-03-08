import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl

from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from ssd.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    key_offsets = idx * key_stride + offs
    value_offsets = idx * value_stride + offs
    key = tl.load(key_ptr + key_offsets, mask=mask)
    value = tl.load(value_ptr + value_offsets, mask=mask)
    cache_offsets = slot.to(tl.int64) * D + offs
    tl.store(k_cache_ptr + cache_offsets, key, mask=mask)
    tl.store(v_cache_ptr + cache_offsets, value, mask=mask)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    block_d = 1 << (D - 1).bit_length()
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        D=D,
        BLOCK_D=block_d,
    )

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        draft: bool = False,
        speculate: bool = False,
        draft_async: bool = False,
        use_eagle: bool = False,
        F: int = 1,
        K: int = 1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.draft = draft
        self.speculate = speculate
        self.draft_async = draft_async
        self.use_eagle = use_eagle
        self.prefill_wrappers = {}
        self.F = F # async_fan_out
        self.K = K # speculate_k
        self.only_prefill_wrapper = None

    def _expand_kv_heads(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_kv_heads == self.num_heads:
            return x
        repeat = self.num_heads // self.num_kv_heads
        return x.repeat_interleave(repeat, dim=1)

    def _gather_paged_kv(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        context_len: int,
    ) -> torch.Tensor:
        if context_len == 0:
            return cache.new_empty((0, self.num_kv_heads, self.head_dim))
        blocks = []
        remaining = context_len
        for block_id in block_table.tolist():
            if block_id < 0 or remaining <= 0:
                break
            take = min(remaining, cache.shape[1])
            blocks.append(cache[block_id, :take])
            remaining -= take
        return torch.cat(blocks, dim=0)

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        k = self._expand_kv_heads(k)
        v = self._expand_kv_heads(v)
        q_t = q.transpose(0, 1).unsqueeze(0)
        k_t = k.transpose(0, 1).unsqueeze(0)
        v_t = v.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            is_causal=True,
            scale=self.scale,
        )
        return out.squeeze(0).transpose(0, 1)

    def _prefill_fallback(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context,
    ) -> torch.Tensor:
        outs = []
        batch_size = context.cu_seqlens_q.numel() - 1
        for i in range(batch_size):
            q0 = int(context.cu_seqlens_q[i].item())
            q1 = int(context.cu_seqlens_q[i + 1].item())
            q_seq = q[q0:q1]
            if context.block_tables is None:
                k0 = int(context.cu_seqlens_k[i].item())
                k1 = int(context.cu_seqlens_k[i + 1].item())
                k_seq = k[k0:k1]
                v_seq = v[k0:k1]
            else:
                context_len = int(context.cu_seqlens_k[i + 1].item() - context.cu_seqlens_k[i].item())
                k_seq = self._gather_paged_kv(self.k_cache, context.block_tables[i], context_len)
                v_seq = self._gather_paged_kv(self.v_cache, context.block_tables[i], context_len)
            outs.append(self._sdpa(q_seq, k_seq, v_seq))
        return torch.cat(outs, dim=0)

    def _decode_fallback(self, q: torch.Tensor, context) -> torch.Tensor:
        outs = []
        if context.cu_seqlens_q is None:
            batch_size = q.shape[0]
            q_splits = [(i, i + 1) for i in range(batch_size)]
        else:
            batch_size = context.cu_seqlens_q.numel() - 1
            q_splits = [
                (int(context.cu_seqlens_q[i].item()), int(context.cu_seqlens_q[i + 1].item()))
                for i in range(batch_size)
            ]
        for i, (q0, q1) in enumerate(q_splits):
            q_seq = q[q0:q1]
            context_len = int(context.context_lens[i].item())
            k_seq = self._gather_paged_kv(self.k_cache, context.block_tables[i], context_len)
            v_seq = self._gather_paged_kv(self.v_cache, context.block_tables[i], context_len)
            outs.append(self._sdpa(q_seq, k_seq, v_seq))
        return torch.cat(outs, dim=0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        flash_kwargs = {}
        if q.is_cuda and torch.cuda.get_device_capability(q.device)[0] >= 12:
            flash_kwargs["ver"] = 4

        k_cache, v_cache = self.k_cache, self.v_cache

        context = get_context()
        if self.k_cache.numel() and self.v_cache.numel():
            store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:
                k, v = k_cache, v_cache

            k, v = k.view(-1, self.num_kv_heads, self.head_dim), v.view(-1, self.num_kv_heads, self.head_dim)
            try:
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, **flash_kwargs)
            except (AssertionError, RuntimeError):
                o = self._prefill_fallback(q, k, v, context)
        else:
            # verify/glue decode: multi-query with cu_seqlens_q (K+1 or variable per seq)
            verify_or_glue = (
                self.speculate and context.cu_seqlens_q is not None
            )
            decode = not verify_or_glue
            tree_decode = (
                decode and self.speculate and self.draft and self.draft_async
                and not context.is_jit
            )

            if verify_or_glue:
                assert context.context_lens is not None
                try:
                    o = flash_attn_with_kvcache(q, k_cache, v_cache,
                                            cache_seqlens=context.context_lens, page_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True,
                                            cu_seqlens_q=context.cu_seqlens_q, max_seqlen_q=context.max_seqlen_q,
                                            **flash_kwargs)
                except (AssertionError, RuntimeError):
                    o = self._decode_fallback(q, context)

            elif tree_decode:
                if self.only_prefill_wrapper is not None:
                    prefill_wrapper = self.only_prefill_wrapper
                else:
                    mq_len = self.F * (self.K+1)
                    bs = q.shape[0] // mq_len
                    wrapper_bs = None
                    for available_bs in sorted(self.prefill_wrappers.keys()):
                        if available_bs >= bs:
                            wrapper_bs = available_bs
                            break
                    prefill_wrapper = self.prefill_wrappers[wrapper_bs]
                o = prefill_wrapper.run(q, (self.k_cache, self.v_cache))
            else: # single query decode
                q = q.unsqueeze(1)
                try:
                    o = flash_attn_with_kvcache(q, k_cache, v_cache,
                                                cache_seqlens=context.context_lens, page_table=context.block_tables,
                                                softmax_scale=self.scale, causal=True,
                                                **flash_kwargs,
                                                )
                except (AssertionError, RuntimeError):
                    o = self._decode_fallback(q.squeeze(1), context)

        o = o.view(-1, self.num_heads * self.head_dim)
        return o
