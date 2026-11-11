"""Helpers that keep activation checkpointing transparent to BatchNorm state."""

import contextlib

import torch
from torch.nn.modules.batchnorm import _BatchNorm


@contextlib.contextmanager
def preserve_batchnorm_running_stats(module):
    """Restore BatchNorm buffers after a checkpoint recomputation.

    Non-reentrant activation checkpointing reruns the wrapped forward during
    backward.  BatchNorm must remain in training mode for that recomputation so
    the recomputed activations and gradients use the same batch statistics as
    the original forward.  The second pass must not, however, permanently
    advance running_mean, running_var, or num_batches_tracked.

    The snapshot is taken when this context is entered (i.e. immediately before
    recomputation), so restoring it leaves exactly the single update performed
    by the original forward.  SyncBatchNorm is covered because it subclasses
    _BatchNorm; its distributed statistics/gradients are still computed during
    recomputation, while only the persistent buffers are rolled back.
    """

    snapshots = []
    for child in module.modules():
        if not isinstance(child, _BatchNorm) or not child.track_running_stats:
            continue
        snapshots.append(
            (
                child,
                None if child.running_mean is None else child.running_mean.detach().clone(),
                None if child.running_var is None else child.running_var.detach().clone(),
                (
                    None
                    if child.num_batches_tracked is None
                    else child.num_batches_tracked.detach().clone()
                ),
            )
        )

    try:
        yield
    finally:
        with torch.no_grad():
            for child, running_mean, running_var, num_batches_tracked in snapshots:
                if running_mean is not None and child.running_mean is not None:
                    child.running_mean.copy_(running_mean)
                if running_var is not None and child.running_var is not None:
                    child.running_var.copy_(running_var)
                if num_batches_tracked is not None and child.num_batches_tracked is not None:
                    child.num_batches_tracked.copy_(num_batches_tracked)


def batchnorm_safe_checkpoint_context_fn(module):
    """Return forward/recompute contexts for ``checkpoint(context_fn=...)``."""

    return contextlib.nullcontext(), preserve_batchnorm_running_stats(module)
