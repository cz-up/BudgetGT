import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from gt_sp.checkpointing import batchnorm_safe_checkpoint_context_fn
from gt_sp.gt_layer import DistributedAttentionNodeLevel
from gt_sp.initialize import (
    sequence_parallel_is_initialized,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
)
from torch_scatter import scatter

try:
    from gt_sp.multi_tier import _MultiTierResourceManager, apply_tier
except ModuleNotFoundError as exc:
    if exc.name != "gt_sp.multi_tier":
        raise
    _MultiTierResourceManager = None

    def apply_tier(layer, mode: str, x, **kwargs):
        if mode == "retain":
            return layer(x, **kwargs)
        if mode in ("keep_mha", "ffn_only"):
            x = layer.forward_attn_only(x, **kwargs)
            return layer.forward_ffn_checkpointed(x)

        def _full(h):
            return layer(h, **kwargs)

        return checkpoint(
            _full,
            x,
            use_reentrant=False,
            context_fn=lambda: batchnorm_safe_checkpoint_context_fn(layer),
        )


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        

class CoreAttention(nn.Module):
    """
    Core attn 
    """
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(CoreAttention, self).__init__()

        # SP group: Per attention head and per partition values.
        seq_parallel_world_size = 1
        if sequence_parallel_is_initialized():
            seq_parallel_world_size = get_sequence_parallel_world_size()
        world_size = seq_parallel_world_size 

        self.hidden_size_per_partition = hidden_size // world_size
        self.hidden_size_per_attention_head =  hidden_size // num_heads
        self.num_attention_heads_per_partition = num_heads // world_size

        self.scale = math.sqrt(self.hidden_size_per_attention_head)
        self.num_heads = num_heads
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.attention_dropout_rate = attention_dropout_rate



    def sparse_attention_bias(self, q, k, v, edge_index, attn_bias):
        # q, k, v: [b, s, np, hn]  e: [total_edges, n, hn], edge_index: [2, total_edges], attn_bias: [b, n, s+1, s+1]
        batch_size, node_num = k.size(0), k.size(1)
        num_heads = q.size(2)
        if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2 and edge_index.size(0) == 3:
            edge_index = edge_index[:2]
        
        # Reshaping into [total_s, np, hn] to
        # get projections for multi-head attention
        # kqv: [total_s, np, hn],  e: [total_edges, np, hn]
        q = q.view(-1, num_heads, self.hidden_size_per_attention_head)
        k = k.view(-1, num_heads, self.hidden_size_per_attention_head)
        v = v.view(-1, num_heads, self.hidden_size_per_attention_head)

        # -> [total_edges, np, hn]
        src = k[edge_index[0].to(torch.long)]
        dest = q[edge_index[1].to(torch.long)]
        score = torch.mul(src, dest)  # element-wise multiplication

        # Scale scores by sqrt(d)
        score = score / self.scale

        # Use available edge features to modify the scores for edges
        # -> [total_edges, np, 1]
        score = score.sum(-1, keepdim=True).clamp(-5, 5)

        # [b, np, s+1, s+1] -> [b, s+1, s+1, np] -> [b, s+1, b, s+1, np]
        if attn_bias is not None:
            attn_bias = attn_bias.permute(0, 2, 3, 1).contiguous().unsqueeze(2).repeat(1, 1, batch_size, 1, 1)
            attn_bias = attn_bias.view(batch_size * node_num, batch_size * node_num, num_heads)
            attn_bias = attn_bias.repeat(1, 1, 1, num_heads)

            score = score + \
                    attn_bias[edge_index[0].to(torch.long), edge_index[1].to(torch.long), :].unsqueeze(2)

        # softmax -> [total_edges, np, 1]
        score = torch.exp(score)

        # Normalize first (per-destination softmax), then apply att_dropout —
        # matching torch.nn.MultiheadAttention(dropout=attn_dropout), which
        # drops weights after softmax normalization (same scheme as GraphGPS).
        # Note the original GT (Dwivedi & Bresson) has no attention-prob
        # dropout; attention_dropout_rate=0 reproduces it exactly.
        Z = score.new_zeros(v.size(0), num_heads, 1)
        scatter(score, edge_index[1], dim=0, out=Z, reduce='add')
        norm_score = score / (Z[edge_index[1].to(torch.long)] + 1e-6)
        norm_score = self.att_dropout(norm_score)

        # Apply normalized attention weights to source values, aggregate at dst
        # -> [total_s, np, hn]
        msg = v[edge_index[0].to(torch.long)] * norm_score
        wV = torch.zeros_like(v)
        scatter(msg, edge_index[1], dim=0, out=wV, reduce='add')

        return wV


    def forward(self, q, k, v, attn_bias=None, edge_index=None, attn_type=None):
        # ===================================
        # Raw attention scores. [b, np, s+1, s+1]
        # ===================================
        # q, k, v: [b, s+p, np, hn], edge_index: [2, total_edges], attn_bias: [b, n, s+p, s+p]
        batch_size, s_len = q.size(0), q.size(1)
        attn_type = "sparse" if attn_type is None else str(attn_type).lower()
        if attn_type != "sparse":
            raise ValueError(
                f"GT is restricted to graph-neighborhood sparse attention to match the original implementation; "
                f"got attn_type={attn_type!r}."
            )
        x = self.sparse_attention_bias(q, k, v, edge_index, attn_bias)
        
        # [b, s+p, hp]
        x = x.view(batch_size, s_len, -1)

        return x


class MultiHeadAttention(nn.Module):
    """Distributed multi-headed attention.

    """
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads # hn
        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)

        local_attn = CoreAttention(
            hidden_size, attention_dropout_rate, num_heads)
        if sequence_parallel_is_initialized():
            self.dist_attn = DistributedAttentionNodeLevel(
                local_attn,
                get_sequence_parallel_group(),
            )
        else:
            self.dist_attn = local_attn


    def forward(self, x, attn_bias=None, edge_index=None, attn_type=None):
        # x: [b, s/p+1, h], attn_bias: [b, n_head, s+1, s+1]
        orig_q_size = x.size()
        # =====================
        # Query, Key, and Value
        # =====================

        # q, k, v: [b, s/p+1, h] -> [b, s/p+1, n_head, hn]
        batch_size = x.size(0) # number of sequences to train a time 
        q = self.linear_q(x).view(batch_size, -1, self.num_heads, self.att_size)
        k = self.linear_k(x).view(batch_size, -1, self.num_heads, self.att_size) 
        v = self.linear_v(x).view(batch_size, -1, self.num_heads, self.att_size)
        # print(f'rank {get_sequence_parallel_rank()} q: {q[:, 0, :, :]}')
        # exit(0)
        

        # ==================================
        # core attention computation
        # ==================================
        x = self.dist_attn(q, k, v, attn_bias, edge_index, attn_type)

        # =================
        # linear
        # =================

        # [b, s/p+1, h]

        assert x.size() == orig_q_size
        return x



class EncoderLayer(nn.Module):
    def __init__(
        self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads,
    ):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads,
        )
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.O = nn.Linear(hidden_size, hidden_size)

        # Original GT (Dwivedi & Bresson) defaults to BatchNorm
        # (batch_norm=True, layer_norm=False in the reference configs).
        self.norm1 = nn.BatchNorm1d(hidden_size)

        self.FFN_layer1 = nn.Linear(hidden_size, hidden_size * 2)
        self.FFN_layer2 = nn.Linear(hidden_size*2, hidden_size)
        self.norm2 = nn.BatchNorm1d(hidden_size)

    def _bn(self, norm, x):
        """Apply BatchNorm1d to (B, N, C) by temporarily flattening the batch dim."""
        b, n, c = x.shape
        return norm(x.view(b * n, c)).view(b, n, c)

    def forward(self, x, attn_bias=None, edge_index=None, attn_type=None):
        y = self.self_attention(x, attn_bias, edge_index=edge_index, attn_type=attn_type)
        y = self.self_attention_dropout(y)
        y = self.O(y)
        x = self._bn(self.norm1, x + y)

        y = self.FFN_layer1(x)
        y = F.relu(y)
        y = self.self_attention_dropout(y)
        y = self.FFN_layer2(y)
        x = self._bn(self.norm2, x + y)
        return x

    def forward_attn_only(self, x, attn_bias=None, edge_index=None, attn_type=None, **kwargs):
        """MHA sub-block (used by apply_tier keep_mha path).

        Returns norm1(x + attn_out); FFN takes this normalized residual as input.
        """
        y = self.self_attention(x, attn_bias, edge_index=edge_index, attn_type=attn_type)
        y = self.self_attention_dropout(y)
        y = self.O(y)
        return self._bn(self.norm1, x + y)

    def forward_ffn_checkpointed(self, x_ln):
        """FFN sub-block with gradient checkpointing.

        x_ln is norm1(x + attn_out) from forward_attn_only.
        """
        def _ffn(h):
            y = self.FFN_layer1(h)
            y = F.relu(y)
            y = self.self_attention_dropout(y)
            y = self.FFN_layer2(y)
            return self._bn(self.norm2, h + y)
        return checkpoint(
            _ffn,
            x_ln,
            use_reentrant=False,
            context_fn=lambda: batchnorm_safe_checkpoint_context_fn(self.norm2),
        )


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y



class GT(nn.Module):
    """GT for node-level task.
    No global token.

    """
    def __init__(
        self,
        n_layers,
        num_heads,
        input_dim,
        hidden_dim,
        output_dim,
        attn_bias_dim,
        dropout_rate,
        input_dropout_rate,
        attention_dropout_rate,
        ffn_dim,
        num_global_node,
    ):
        super().__init__()
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(input_dropout_rate)
        

        encoders = [
            EncoderLayer(
                hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads,
            )
            for _ in range(n_layers)
        ]
        self.layers = nn.ModuleList(encoders)
        self.n_layers = n_layers

        self.MLP_layer = MLPReadout(hidden_dim, output_dim)   # 1 out dim since regression problem
        self.apply(lambda module: init_params(module, n_layers=n_layers))

        self.activation_checkpoint = False
        self.activation_checkpoint_mode = "none"
        self._comm_ckpt = None  # _MultiTierResourceManager | None

    def set_activation_checkpoint(self, enabled: bool = True, mode: str | None = None, deferred: bool = False) -> None:
        mode_aliases = {
            None: "none",
            "none": "none",
            "all": "layer",
            "layer": "layer",
            "ffn_only": "ffn_only",
            "multi_tier": "multi_tier",
        }
        resolved_mode = mode_aliases.get(mode)
        if resolved_mode is None:
            raise ValueError(
                f"activation_checkpoint_mode must be 'all', 'ffn_only', or 'multi_tier', got {mode!r}"
            )
        enabled = bool(enabled) and resolved_mode != "none"
        self.activation_checkpoint = enabled
        self.activation_checkpoint_mode = resolved_mode if enabled else "none"
        if self.activation_checkpoint_mode == "multi_tier":
            if _MultiTierResourceManager is None:
                raise RuntimeError(
                    "activation_checkpoint_mode='multi_tier' requires gt_sp/multi_tier.py, "
                    "but that module is not available in this environment."
                )
            self._comm_ckpt = _MultiTierResourceManager(len(self.layers), deferred=deferred)
        else:
            self._comm_ckpt = None

    def comm_aware_notify_budget_frozen(self, reuse_deferred_baseline: bool = False) -> None:
        if self._comm_ckpt is not None:
            self._comm_ckpt.notify_budget_frozen(reuse_deferred_baseline=reuse_deferred_baseline)

    def comm_aware_notify_step_end(self, device, t_bwd: float = None, t_fwd: float = None) -> None:
        if self._comm_ckpt is not None:
            self._comm_ckpt.notify_step_end(device, t_bwd=t_bwd, t_fwd=t_fwd)

    def forward(self, x, attn_bias, edge_index, perturb=None, attn_type=None):
        # x → [bs=1, s/p, x_d]
        x = x.unsqueeze(0)

        # [bs, s/p, x_d] → [bs, s/p, h]
        node_feature = self.node_encoder(x)

        node_feature -= perturb if perturb is not None else 0
        output = self.input_dropout(node_feature)

        use_ckpt = self.activation_checkpoint and self.training and torch.is_grad_enabled()
        ckpt_mode = getattr(self, "activation_checkpoint_mode", "none")

        if use_ckpt and ckpt_mode == "multi_tier" and self._comm_ckpt is not None:
            self._comm_ckpt.plan(output.device)

        for i, enc_layer in enumerate(self.layers):
            if not use_ckpt:
                output = enc_layer(output, edge_index=edge_index, attn_type=attn_type)
                continue
            if ckpt_mode == "multi_tier" and self._comm_ckpt is not None:
                layer_mode = self._comm_ckpt.mode(i)
            else:
                layer_mode = ckpt_mode
            output = apply_tier(enc_layer, layer_mode, output, edge_index=edge_index, attn_type=attn_type)

        if use_ckpt and ckpt_mode == "multi_tier" and self._comm_ckpt is not None:
            self._comm_ckpt.record_post_forward_memory(output.device)

        # Output part
        output = self.MLP_layer(output[0, :, :])

        return F.log_softmax(output, dim=1)


