# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from typing import Any, Tuple
from torch import Tensor
from torch.nn import Module
import torch.distributed as dist

from gt_sp.comm_profiler import profile_call
from gt_sp.initialize import (
    get_sequence_parallel_world_size,
)
from gt_sp.utils import (
    extend_global_token0,
    copy_global_token0,
)


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def _run_all_to_all(group: dist.ProcessGroup, input: Tensor, scatter_idx: int, gather_idx: int, tag: str) -> Tensor:
        seq_world_size = get_sequence_parallel_world_size()

        input_list = [t.contiguous() for t in torch.tensor_split(input, seq_world_size, scatter_idx)]
        output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
        payload_bytes = int(input.numel() * input.element_size())

        def _collective():
            torch.distributed.all_to_all(output_list, input_list, group=group)
            return torch.cat(output_list, dim=gather_idx).contiguous()

        return profile_call(tag, _collective, device=input.device, payload_bytes=payload_bytes)

    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor, scatter_idx: int, gather_idx: int) -> Tensor:

        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        return _SeqAllToAll._run_all_to_all(group, input, scatter_idx, gather_idx, "seq_all_to_all_fwd")

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        grad_input = _SeqAllToAll._run_all_to_all(
            ctx.group,
            grad_output[0],
            ctx.gather_idx,
            ctx.scatter_idx,
            "seq_all_to_all_bwd",
        )
        return (None, grad_input, None, None)




class DistributedAttentionNoMerge(torch.nn.Module):
    """Distributed attention for node-level tasks (no global-token merge)."""

    def __init__(
        self,
        local_attention: Module,
        sequence_process_group: dist.ProcessGroup,
        scatter_idx: int = 2,  # head
        gather_idx: int = 1,  # s
    ) -> None:

        super(DistributedAttentionNoMerge, self).__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_bias: Tensor, edge_index: Tensor, attn_type, *args: Any) -> Tensor:
        """ forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        query_layer = _SeqAllToAll.apply(self.spg, query, self.scatter_idx, self.gather_idx)
        key_layer = _SeqAllToAll.apply(self.spg, key, self.scatter_idx, self.gather_idx)
        value_layer = _SeqAllToAll.apply(self.spg, value, self.scatter_idx, self.gather_idx)

        if attn_bias is not None:
            attn_bias_layer = _SeqAllToAll.apply(self.spg, attn_bias, 3, 1)
            attn_bias_layer = extend_global_token0(attn_bias_layer, extend_dim=2)
        else:
            attn_bias_layer = attn_bias

        context_layer = self.local_attn(
            query_layer, key_layer, value_layer,
            attn_bias_layer, edge_index, attn_type, *args,
        )

        context_layer = copy_global_token0(context_layer, extend_dim=1)
        output = _SeqAllToAll.apply(self.spg, context_layer, self.gather_idx, self.scatter_idx)
        return output
