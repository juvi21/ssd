import torch
from einops import rearrange
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate

from ssd.layers.attention import Attention
from ssd.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from ssd.layers.layernorm import RMSDNorm
from ssd.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from ssd.layers.rotary_embedding import get_rope
from ssd.utils.context import get_context


class KimiRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(orig_dtype)


class KimiBlockSparseMLP(nn.Module):
    def __init__(
        self,
        config,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ):
        super().__init__()
        self.hidden_dim = config.hidden_size if hidden_size is None else hidden_size
        self.ffn_dim = config.intermediate_size if intermediate_size is None else intermediate_size
        self.w1 = ColumnParallelLinear(
            self.hidden_dim,
            self.ffn_dim,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.w2 = RowParallelLinear(
            self.ffn_dim,
            self.hidden_dim,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.w3 = ColumnParallelLinear(
            self.hidden_dim,
            self.ffn_dim,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.act_fn = F.silu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states))


class KimiMLP(nn.Module):
    def __init__(
        self,
        config,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class KimiMoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_token
        self.num_experts = config.num_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.moe_router_activation_func = config.moe_router_activation_func
        self.num_expert_group = getattr(config, "num_expert_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)
        self.moe_renormalize = config.moe_renormalize
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.num_experts, self.gating_dim)))
        self.weight.weight_loader = self.weight_loader
        self.e_score_correction_bias = nn.Parameter(torch.empty(self.num_experts))
        self.e_score_correction_bias.weight_loader = self.weight_loader
        self.reset_parameters()

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden)
        num_tokens = hidden_states.shape[0]
        logits = F.linear(hidden_states.float(), self.weight.float(), None)
        if self.moe_router_activation_func == "sigmoid":
            scores = logits.sigmoid()
        elif self.moe_router_activation_func == "softmax":
            scores = logits.softmax(dim=1)
        else:
            raise NotImplementedError(
                f"Unsupported Kimi MoE router activation: {self.moe_router_activation_func}",
            )

        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        group_scores = scores_for_choice.view(
            num_tokens,
            self.num_expert_group,
            -1,
        ).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(
            group_scores,
            k=self.topk_group,
            dim=-1,
            sorted=False,
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(
            num_tokens,
            self.num_expert_group,
            self.num_experts // self.num_expert_group,
        ).reshape(num_tokens, -1)
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)
        if self.top_k > 1 and self.moe_renormalize:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight


class KimiSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.experts = nn.ModuleList([
            KimiBlockSparseMLP(
                config,
                intermediate_size=config.moe_intermediate_size,
                tp_group=tp_group,
                tp_size=tp_size,
            )
            for _ in range(config.num_experts)
        ])
        self.gate = KimiMoEGate(config)
        if config.num_shared_experts is not None:
            self.shared_experts = KimiMLP(
                config=config,
                intermediate_size=config.moe_intermediate_size * config.num_shared_experts,
                tp_group=tp_group,
                tp_size=tp_size,
            )
        else:
            self.shared_experts = None

    @torch.no_grad()
    def moe_infer(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0).cpu().numpy()
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = hidden_states[idxs // topk_ids.shape[1]]

        outputs = []
        start_idx = 0
        for expert_idx, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            outputs.append(self.experts[expert_idx](sorted_tokens[start_idx:end_idx]))
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)
        reordered = torch.empty_like(outs)
        reordered[idxs] = outs
        return reordered.view(*topk_ids.shape, -1).type(topk_weight.dtype).mul_(
            topk_weight.unsqueeze(dim=-1),
        ).sum(dim=1).type(reordered.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        flat_hidden = hidden_states.view(-1, hidden_states.shape[-1])
        output = self.moe_infer(flat_hidden, topk_idx, topk_weight).view(*orig_shape)
        if self.shared_experts is not None:
            output = output + self.shared_experts(identity)
        return output


class KimiMLAAttention(nn.Module):
    def __init__(
        self,
        config,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ):
        super().__init__()
        self.config = config
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.hidden_size = config.hidden_size
        self.kv_lora_rank = config.kv_lora_rank
        self.scaling = self.q_head_dim ** -0.5

        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_heads * self.q_head_dim,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.total_num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=getattr(config, "max_position_embeddings", getattr(config, "model_max_length", 4096)),
            base=config.rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.attn = Attention(
            self.num_heads,
            self.q_head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.size(0)
        q_states = self.q_proj(hidden_states).view(num_tokens, self.num_heads, self.q_head_dim)
        q_pass, q_rot = torch.split(
            q_states,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_states = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(
            num_tokens,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_pass, value_states = torch.split(
            kv_states,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        q_rot_flat = q_rot.reshape(num_tokens, -1)
        k_rot_flat = k_rot.unsqueeze(1).expand(-1, self.num_heads, -1).reshape(num_tokens, -1)
        q_rot_out, k_rot_out = self.rotary_emb(positions, q_rot_flat, k_rot_flat)
        q_rot = q_rot_out.view(num_tokens, self.num_heads, self.qk_rope_head_dim)
        k_rot = k_rot_out.view(num_tokens, self.num_heads, self.qk_rope_head_dim)

        query_states = torch.cat((q_pass, q_rot), dim=-1).reshape(num_tokens, -1)
        key_states = torch.cat((k_pass, k_rot), dim=-1).reshape(num_tokens, -1)
        if self.v_head_dim != self.q_head_dim:
            value_states = F.pad(value_states, (0, self.q_head_dim - self.v_head_dim))
        attn_output = self.attn(
            query_states,
            key_states,
            value_states.reshape(num_tokens, -1),
        )
        attn_output = attn_output.view(num_tokens, self.num_heads, self.q_head_dim)[..., :self.v_head_dim]
        return self.o_proj(attn_output.reshape(num_tokens, -1))


class KimiDeltaAttention(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.hidden_size = config.hidden_size
        self.conv_size = config.linear_attn_config["short_conv_kernel_size"]
        self.total_num_heads = config.linear_attn_config["num_heads"]
        self.num_heads = self.total_num_heads // tp_size
        self.head_dim = config.linear_attn_config["head_dim"]
        self.projection_k_size = self.total_num_heads * self.head_dim
        self.projection_size = self.projection_k_size

        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.projection_k_size,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.projection_k_size,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.projection_size,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.q_conv1d = ShortConvolution(
            hidden_size=self.num_heads * self.head_dim,
            kernel_size=self.conv_size,
            activation="silu",
        )
        self.q_conv1d.weight.weight_loader = self.conv_weight_loader
        self.k_conv1d = ShortConvolution(
            hidden_size=self.num_heads * self.head_dim,
            kernel_size=self.conv_size,
            activation="silu",
        )
        self.k_conv1d.weight.weight_loader = self.conv_weight_loader
        self.v_conv1d = ShortConvolution(
            hidden_size=self.num_heads * self.head_dim,
            kernel_size=self.conv_size,
            activation="silu",
        )
        self.v_conv1d.weight.weight_loader = self.conv_weight_loader
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)).view(1, 1, -1, 1),
        )
        self.A_log.weight_loader = self.a_log_loader
        self.f_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            self.projection_size,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.dt_bias = nn.Parameter(torch.empty(self.num_heads * self.head_dim, dtype=torch.float32))
        self.dt_bias.weight_loader = self.dt_bias_loader
        self.b_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_heads,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.g_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            self.projection_size,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.o_norm = FusedRMSNormGated(
            self.head_dim,
            eps=config.rms_norm_eps,
            activation="sigmoid",
        )
        self.o_proj = RowParallelLinear(
            self.projection_size,
            self.hidden_size,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.conv_states: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.recurrent_states: dict[int, torch.Tensor] = {}

    def conv_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        shard_size = param.data.size(0)
        start_idx = (dist.get_rank(group=self.tp_group) if self.tp_size > 1 else 0) * shard_size
        param.data.copy_(loaded_weight.narrow(0, start_idx, shard_size))

    def a_log_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        shard_size = param.data.size(2)
        start_idx = (dist.get_rank(group=self.tp_group) if self.tp_size > 1 else 0) * shard_size
        param.data.copy_(loaded_weight.narrow(2, start_idx, shard_size))

    def dt_bias_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        shard_size = param.data.size(0)
        start_idx = (dist.get_rank(group=self.tp_group) if self.tp_size > 1 else 0) * shard_size
        param.data.copy_(loaded_weight.narrow(0, start_idx, shard_size))

    def _get_seq_ids(self) -> tuple[int, ...]:
        context = get_context()
        if context.seq_ids is None:
            raise RuntimeError("Kimi KDA requires seq_ids in the runtime context.")
        return context.seq_ids

    def _stack_states(
        self,
        seq_ids: tuple[int, ...],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not seq_ids:
            return None, None, None, None
        if seq_ids[0] not in self.recurrent_states:
            return None, None, None, None
        missing = [seq_id for seq_id in seq_ids if seq_id not in self.recurrent_states]
        if missing:
            raise RuntimeError(f"Missing Kimi KDA state for sequences: {missing}")
        recurrent_state = torch.stack([self.recurrent_states[seq_id] for seq_id in seq_ids], dim=0)
        conv_state_q = torch.stack([self.conv_states[seq_id][0] for seq_id in seq_ids], dim=0)
        conv_state_k = torch.stack([self.conv_states[seq_id][1] for seq_id in seq_ids], dim=0)
        conv_state_v = torch.stack([self.conv_states[seq_id][2] for seq_id in seq_ids], dim=0)
        return recurrent_state, conv_state_q, conv_state_k, conv_state_v

    def _update_states(
        self,
        seq_ids: tuple[int, ...],
        recurrent_state: torch.Tensor | None,
        conv_state_q: torch.Tensor | None,
        conv_state_k: torch.Tensor | None,
        conv_state_v: torch.Tensor | None,
    ):
        if recurrent_state is None:
            return
        for idx, seq_id in enumerate(seq_ids):
            self.recurrent_states[seq_id] = recurrent_state[idx].detach()
            self.conv_states[seq_id] = (
                conv_state_q[idx].detach(),
                conv_state_k[idx].detach(),
                conv_state_v[idx].detach(),
            )

    def reset_seq_states(self, seq_ids: list[int]):
        for seq_id in seq_ids:
            self.recurrent_states.pop(seq_id, None)
            self.conv_states.pop(seq_id, None)

    def cleanup_seq_states(self, seq_ids: list[int]):
        self.reset_seq_states(seq_ids)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        seq_ids = self._get_seq_ids()
        batch_size = len(seq_ids)
        if context.is_prefill:
            hidden_states = hidden_states.unsqueeze(0)
            cu_seqlens = context.cu_seqlens_q.to(torch.long) if context.cu_seqlens_q is not None else None
            recurrent_state = None
            conv_state_q = conv_state_k = conv_state_v = None
            max_q_len = context.max_seqlen_q if context.cu_seqlens_q is not None else hidden_states.shape[1]
        elif context.cu_seqlens_q is not None:
            q_len = hidden_states.size(0) // batch_size
            hidden_states = hidden_states.view(batch_size, q_len, -1)
            cu_seqlens = None
            recurrent_state, conv_state_q, conv_state_k, conv_state_v = self._stack_states(seq_ids)
            max_q_len = context.max_seqlen_q if context.max_seqlen_q > 0 else q_len
        else:
            hidden_states = hidden_states.view(batch_size, 1, -1)
            cu_seqlens = None
            recurrent_state, conv_state_q, conv_state_k, conv_state_v = self._stack_states(seq_ids)
            max_q_len = 1

        q, conv_state_q = self.q_conv1d(
            x=self.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
        k, conv_state_k = self.k_conv1d(
            x=self.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
        v, conv_state_v = self.v_conv1d(
            x=self.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
        g = self.f_b_proj(self.f_a_proj(hidden_states))
        g = rearrange(g, "... (h d) -> ... h d", h=self.num_heads, d=self.head_dim)
        g = fused_kda_gate(g, self.A_log, dt_bias=self.dt_bias, output_dtype=hidden_states.dtype)
        beta = self.b_proj(hidden_states).float().sigmoid()

        q = rearrange(q, "... (h d) -> ... h d", h=self.num_heads, d=self.head_dim)
        k = rearrange(k, "... (h d) -> ... h d", h=self.num_heads, d=self.head_dim)
        v = rearrange(v, "... (h d) -> ... h d", h=self.num_heads, d=self.head_dim)

        if max_q_len <= 64:
            output, recurrent_state = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            output, recurrent_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )

        self._update_states(
            seq_ids,
            recurrent_state,
            conv_state_q,
            conv_state_k,
            conv_state_v,
        )

        gated = self.g_b_proj(self.g_a_proj(hidden_states))
        gated = rearrange(gated, "... (h d) -> ... h d", h=self.num_heads, d=self.head_dim)
        output = self.o_norm(output, gated)
        output = self.o_proj(rearrange(output, "b t h d -> (b t) (h d)"))
        return output


class KimiDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ):
        super().__init__()
        self.is_linear_attn = config.is_kda_layer(layer_idx)
        if self.is_linear_attn:
            self.self_attn = KimiDeltaAttention(
                config=config,
                layer_idx=layer_idx,
                tp_group=tp_group,
                tp_size=tp_size,
            )
        else:
            self.self_attn = KimiMLAAttention(
                config=config,
                tp_group=tp_group,
                tp_size=tp_size,
            )
        self.has_moe = (
            config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % getattr(config, "moe_layer_freq", 1) == 0
        )
        if self.has_moe:
            self.block_sparse_moe = KimiSparseMoeBlock(
                config=config,
                tp_group=tp_group,
                tp_size=tp_size,
            )
        else:
            self.mlp = KimiMLP(
                config=config,
                tp_group=tp_group,
                tp_size=tp_size,
            )
        self.input_layernorm = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.is_linear_attn:
            hidden_states = self.self_attn(hidden_states)
        else:
            hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self.has_moe:
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def reset_seq_states(self, seq_ids: list[int]):
        if self.is_linear_attn:
            self.self_attn.reset_seq_states(seq_ids)

    def cleanup_seq_states(self, seq_ids: list[int]):
        if self.is_linear_attn:
            self.self_attn.cleanup_seq_states(seq_ids)


class KimiLinearModel(nn.Module):
    def __init__(
        self,
        config,
        draft: bool = False,
        speculate: bool = False,
        spec_k: int = 1,
        async_fan_out: int = 1,
        draft_async: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            draft_async=draft_async,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.layers = nn.ModuleList([
            KimiDecoderLayer(
                config,
                layer_idx=i,
                tp_group=tp_group,
                tp_size=tp_size,
            )
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)

    def prepare_for_run(self, seqs, is_prefill: bool):
        if not is_prefill:
            return
        reset_ids = [seq.seq_id for seq in seqs if seq.num_cached_tokens == 0]
        if reset_ids:
            self.reset_seq_states(reset_ids)

    def reset_seq_states(self, seq_ids: list[int]):
        for layer in self.layers:
            layer.reset_seq_states(seq_ids)

    def cleanup_seq_states(self, seq_ids: list[int]):
        for layer in self.layers:
            layer.cleanup_seq_states(seq_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class KimiLinearForCausalLM(nn.Module):
    packed_modules_mapping = {}

    def __init__(
        self,
        config,
        draft: bool = False,
        speculate: bool = False,
        use_eagle: bool = False,
        spec_k: int = 1,
        async_fan_out: int = 1,
        draft_async: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ):
        super().__init__()
        assert not use_eagle, "Kimi does not support Eagle draft mode."
        assert not (tp_group is None and tp_size > 1), (
            "KimiLinearForCausalLM requires a TP group when tp_size > 1."
        )
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.model = KimiLinearModel(
            config,
            draft=draft,
            speculate=speculate,
            spec_k=spec_k,
            async_fan_out=async_fan_out,
            draft_async=draft_async,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            draft_async=draft_async,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def prepare_for_run(self, seqs, is_prefill: bool):
        self.model.prepare_for_run(seqs, is_prefill)

    def cleanup_seq_states(self, seq_ids: list[int]):
        self.model.cleanup_seq_states(seq_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor, last_only: bool = True) -> torch.Tensor:
        return self.lm_head(hidden_states, last_only=last_only)
