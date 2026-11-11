import torch
import torch.distributed as dist
import numpy as np
import random
import contextlib
import os
import sys
import time
from typing import List, Optional
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.utils import coalesce, remove_self_loops, subgraph, add_self_loops
from gt_sp.initialize import (
    sequence_parallel_is_initialized,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_src_rank,
    get_sequence_length_per_rank,
    set_global_token_indices,
    get_global_token_indices,
    get_global_token_num,
    set_last_batch_global_token_indices,
    last_batch_flag,
    get_last_batch_flag,
)

_RANDOM_WALK_GRAPH_CACHE = {}


def clear_random_walk_graph_cache(device_type: str = None, data_ptr: int = None) -> int:
    """Evict cached random-walk graph structures after graph storage changes.

    With no filters, evicts everything.  ``device_type`` ("cpu"/"cuda")
    restricts eviction to CSR entries resident on that device type (cache
    key[3]); ``data_ptr`` restricts to entries built from a specific source
    edge_index tensor (cache key[0]) — used to drop entries keyed to a
    temporary GPU topology cache before that tensor is freed.
    Returns the number of CSR entries evicted.
    """
    if device_type is None and data_ptr is None:
        n = len(_RANDOM_WALK_GRAPH_CACHE)
        _RANDOM_WALK_GRAPH_CACHE.clear()
        return n
    evict = [
        k
        for k in _RANDOM_WALK_GRAPH_CACHE
        if (device_type is None or k[3] == device_type)
        and (data_ptr is None or k[0] == int(data_ptr))
    ]
    for k in evict:
        _RANDOM_WALK_GRAPH_CACHE.pop(k)
    return len(evict)


def _compute_local_spd_bias(edge_index: Tensor, n_nodes: int, max_dist: int) -> Tensor:
    """Compute one-hot shortest-path-distance attention bias for a local subgraph (CPU).

    Args:
        edge_index: [2, E] tensor with local 0-indexed node ids (before fix_edge_index).
        n_nodes:    number of nodes in the subgraph.
        max_dist:   max hop distance; distances > max_dist are clamped to the last bin.
                    Output one-hot dimension = max_dist + 1.

    Returns:
        Tensor of shape [n_nodes, n_nodes, max_dist+1], float32, on CPU.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    d = max_dist + 1

    if n_nodes == 0:
        return torch.zeros(0, 0, d, dtype=torch.float32)

    if edge_index.numel() == 0:
        bias = torch.zeros(n_nodes, n_nodes, d, dtype=torch.float32)
        bias[:, :, -1] = 1.0
        for i in range(n_nodes):
            bias[i, i, -1] = 0.0
            bias[i, i, 0] = 1.0
        return bias

    ei = edge_index.cpu().numpy()
    rows = np.concatenate([ei[0], ei[1]])
    cols = np.concatenate([ei[1], ei[0]])
    data_arr = np.ones(len(rows), dtype=np.float32)
    # Guard against out-of-range indices (e.g. if edge_index already has virtual nodes)
    mask = (rows >= 0) & (rows < n_nodes) & (cols >= 0) & (cols < n_nodes)
    rows, cols, data_arr = rows[mask], cols[mask], data_arr[mask]

    adj = csr_matrix((data_arr, (rows, cols)), shape=(n_nodes, n_nodes))
    dist_mat = shortest_path(adj, method='D', directed=False, unweighted=True)  # [n, n], inf → unreachable

    # inf → max_dist (last bin), then clamp to [0, max_dist]
    dist_mat = np.where(np.isinf(dist_mat), float(max_dist), dist_mat)
    dist_mat = np.clip(dist_mat, 0, max_dist)
    dist_int = torch.from_numpy(dist_mat.astype(np.int64))  # [n, n]

    bias = torch.zeros(n_nodes, n_nodes, d, dtype=torch.float32)
    bias.scatter_(2, dist_int.unsqueeze(2), 1.0)
    return bias


def fix_edge_index(x, num_node):
    # Add new edges of virtual nodes
    virt_edges = []

    num_virtual_tokens = 1
    arange_nodes = torch.arange(num_node, device=x.device, dtype=x.dtype)
    for idx in range(num_virtual_tokens):
        virt_edge_index = torch.cat([(arange_nodes + (1 + idx)).view(1, -1), # virtual node index = 0
                                        (x.new_zeros([num_node]) + idx).view(1, -1)], dim=0)
        virt_edges.append(virt_edge_index)

        virt_edge_index = torch.cat([(x.new_zeros([num_node]) + idx).view(1, -1), 
                                    (arange_nodes + (1 + idx)).view(1, -1)], dim=0)
        virt_edges.append(virt_edge_index)

    extra_virt_edges = torch.cat(virt_edges, dim=1)
    x = torch.cat([(x + 1), extra_virt_edges], dim=1) # virtual node index = 0, other nodes start from 1
    return x


def resolve_edge_build_device(args, default_device) -> torch.device:
    """Resolve the device used for edge construction and sampling."""
    spec = getattr(args, "edge_build_device", "gpu")
    # "gpu" builds edges on the training device (the GPU); "same" is the
    # deprecated alias kept for backward compatibility with older runs/scripts.
    if spec in (None, "", "gpu", "same"):
        return torch.device(default_device)

    try:
        build_device = torch.device(spec)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(
            f"Invalid --edge_build_device={spec!r}; expected gpu/cpu/cuda/cuda:N."
        ) from exc

    if build_device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--edge_build_device requests CUDA, but CUDA is not available.")
        if build_device.index is None:
            default_dev = torch.device(default_device)
            if default_dev.type == "cuda" and default_dev.index is not None:
                return torch.device(f"cuda:{default_dev.index}")
            return torch.device(f"cuda:{torch.cuda.current_device()}")

    return build_device


def _rw_walk_transient_bytes(num_starts: int, walk_length: int) -> int:
    """GPU bytes held by one _run_random_walk call with ``num_starts`` walks.

    The ``[num_starts, L+1]`` int64 output plus, for the step in flight, the
    int64 temporaries cur / lo / deg / offset / deg-1 / index / next / where,
    the float64 uniforms on the device, and the ``deg > 0`` bool mask:
    9 x 8 + 1 = 73 bytes per walk.  Steps run sequentially, so only one step's
    temporaries are live at a time.  No term depends on E.
    """
    n = max(0, int(num_starts))
    return int(n * (int(walk_length) + 1) * 8 + n * 73)


def _estimate_rw_working_set_bytes(
    num_nodes: int,
    num_edges: int,
    walks_per_node: int,
    walk_length: int,
    num_heads: int = 1,
) -> int:
    """First-principles estimate of the GPU working set for one call to
    compute_hop_buckets_random_walk() on a graph of (N, E).

    This is the COLD estimate: the CSR is not cached on the GPU yet, so
    _get_random_walk_graph builds it with SparseTensor(...).coalesce(), whose
    sort pipeline is the only O(E) transient.  Once the CSR is cached, a walk
    costs only _rw_walk_transient_bytes (see compute_hop_buckets_random_walk).
    Each line item below corresponds to a specific allocation in either the
    coalesce sort or this module's walk / bucketing code.
    No multiplicative padding — every byte is traceable to a documented
    operation. The 1-byte boolean tensors are counted explicitly rather
    than approximated as int64.

    Sources:
      - SparseTensor.coalesce():    row * N + col key, argsort, reorder,
                                    then rowptr via bincount + cumsum.
      - thrust::sort auxiliary:     CUB's DeviceRadixSort::SortKeys requires
                                    sizeof(KeyT) × num_items of temporary
                                    storage (https://nvlabs.github.io/cub/
                                    structcub_1_1_device_radix_sort.html).
      - compute_hop_buckets loop:   per-d mask + index_select + stack.

    The PyTorch caching allocator's alignment overhead (≤ 512 B per
    allocation) is dwarfed by the per-tensor sizes here and is omitted.
    """
    E = max(0, int(num_edges))
    N = max(1, int(num_nodes))
    W = max(1, int(walks_per_node))
    L = max(1, int(walk_length))

    # ---------- CSR build: SparseTensor(row, col).coalesce() ----------
    # All int64 (8 bytes/element) unless noted.
    #
    # (1) row * num_nodes                          : intermediate, shape (E,)
    sort_step1 = E * 8
    # (2) (row * num_nodes) + col                  : intermediate, shape (E,)
    sort_step2 = E * 8
    # (3) torch.argsort(...) permutation output    : shape (E,)
    argsort_perm = E * 8
    # (4) CUB radix-sort auxiliary storage         : sizeof(int64) × E
    radix_aux = E * 8
    # (5) row[perm], col[perm] (final reordered)   : 2 × shape (E,)
    perm_outputs = 2 * E * 8
    # (6) deg = scatter_add_ on zeros(N)           : shape (N,)
    deg = N * 8
    # (7) rowptr = zeros(N+1) + cumsum             : shape (N+1,)
    rowptr_bytes = (N + 1) * 8
    # (8) walks tensor + one step's temporaries (_run_random_walk)
    walks_bytes = _rw_walk_transient_bytes(N * W, L)

    # ---------- compute_hop_buckets_random_walk per-d loop ----------
    # Live transients during one iteration of d ∈ [1..L] (sequential, so
    # the live peak is for a single d, not the sum):
    #   - valid bool mask    : N*W bytes (1 byte per element)
    #   - kv_nodes[valid]    : ≤ N*W * 8 bytes (worst case all valid)
    #   - query_nodes[valid] : ≤ N*W * 8 bytes
    #   - torch.stack(...)   : 2 × N*W * 8 bytes (the edges tensor)
    valid_mask = N * W * 1
    kv_select = N * W * 8
    query_select = N * W * 8
    stack_edges = 2 * N * W * 8
    per_d_peak = valid_mask + kv_select + query_select + stack_edges

    # ---------- buckets cumulative (across all d) ----------
    # Each hop appends up to N*W edges to the bucket; final bucket holds
    # the sum of all hops' contributions (kept alive until function return).
    bucket_cumulative = 2 * N * W * L * 8

    total = (
        sort_step1 + sort_step2 + argsort_perm + radix_aux
        + perm_outputs + deg + rowptr_bytes + walks_bytes
        + per_d_peak + bucket_cumulative
    )
    return int(total)


# Module-level cache for rw-device decisions to suppress per-rank / per-call
# log spam. Key: (device_type, decision_label); value: previous (estimate_mib,
# free_mib). We only re-print when the decision label flips, not on every
# call with identical inputs.
_RW_DEVICE_DECISION_CACHE: dict = {}

def _clear_rw_device_cache() -> None:
    """Clear the RW-device decision cache so the next call re-evaluates GPU memory."""
    _RW_DEVICE_DECISION_CACHE.clear()


# Per-process GPU memory cap registered by set_rw_gpu_memory_cap() when
# torch.cuda.set_per_process_memory_fraction() is in effect.  0 means no cap.
# _gpu_can_fit_rw uses this to bound the usable budget: the driver-reported
# free (mem_get_info) is the physical free and does NOT account for the
# per-process cap, so without this correction _gpu_can_fit_rw can return True
# while the actual allocation later fails (cap - reserved is too small for
# the requested contiguous block, even though physical free looks ample).
_GPU_PROCESS_MEMORY_CAP_BYTES: int = 0


def set_rw_gpu_memory_cap(cap_bytes: int) -> None:
    """Register the per-process GPU memory cap set via set_per_process_memory_fraction.

    Should be called once from _apply_multi_tier_gpu_memory_limit() after the
    fraction is set.  Passing 0 disables the cap-awareness correction.
    """
    global _GPU_PROCESS_MEMORY_CAP_BYTES
    _GPU_PROCESS_MEMORY_CAP_BYTES = max(0, int(cap_bytes))


# Planner override for the random-walk build device. The auto-decision in
# _gpu_can_fit_rw is intentionally conservative under a per-process HBM cap
# (it clamps usable memory to cap - reserved), so on large graphs it forces
# the rw build to CPU even when the transient build peak would actually fit
# the budget. When the Multi-Tier planner has selected a rw-on-GPU plan (its
# transient peak was profiled feasible under the cap), it flips this flag so
# the rw build stays on GPU. The planner sets it on EVERY (re-)plan — True only
# when the active plan is rw-on-GPU, False otherwise — so the flag can never
# leak a stale True across plan changes.
_FORCE_RW_ON_GPU: bool = False


def set_force_rw_on_gpu(value: bool) -> None:
    """Force (or release) keeping the random-walk build on GPU.

    Set by _MultiTierResourceManager whenever it (re-)selects an active plan.
    A reactive OOM in the rw builder still falls back to CPU, so an over-
    optimistic True degrades gracefully rather than crashing.
    """
    global _FORCE_RW_ON_GPU
    _FORCE_RW_ON_GPU = bool(value)


# Set by multi_tier runs. A GPU random-walk CSR that holds its own copy of col
# (E x 8 bytes, 2 GiB on amazon) stays cached through every later model step,
# so it is persistent HBM that the planner has to budget. While this is set,
# the rw builder creates such a CSR only when the active plan chose rw-on-GPU
# (_FORCE_RW_ON_GPU) and walks on the CPU otherwise -- including before the
# planner has run, so DEFERRED and calibration steps measure the model without
# it. A CSR whose col aliases a GPU-resident sorted topology (gpu_persist)
# costs only rowptr and is exempt.
_RW_GPU_CSR_REQUIRES_PLAN: bool = False


def set_rw_gpu_csr_requires_plan(value: bool) -> None:
    global _RW_GPU_CSR_REQUIRES_PLAN
    _RW_GPU_CSR_REQUIRES_PLAN = bool(value)


# Run-level registry of (device_str, E) for which a real-edge GPU sampling build
# was observed to OOM at runtime and fall back to CPU. Unlike
# _RW_DEVICE_DECISION_CACHE (a per-probe log-suppression cache cleared by
# _clear_rw_device_cache before every policy profile), this set is STICKY for the
# whole run: once a key is present, real-edge sampling for that key skips the GPU
# attempt entirely (no per-epoch retry / OOM spam), and the multi_tier planner is
# told the GPU edge policy is infeasible so it re-plans (e.g. gpu_ephemeral ->
# gpu_persist -> cpu prefetch). The OOM driving this is allocator fragmentation,
# which is not statically predictable — even the mem_get_info-based _gpu_can_fit
# guard is fooled — so we record the observed ground truth instead of estimating.
_RW_REALEDGE_GPU_INFEASIBLE: set = set()


def _rw_realedge_key(device, num_edges: int):
    return (str(device), int(num_edges))


def mark_rw_realedge_gpu_infeasible(device, num_edges: int) -> None:
    """Record that real-edge GPU sampling for (device, num_edges) OOM'd at runtime."""
    _RW_REALEDGE_GPU_INFEASIBLE.add(_rw_realedge_key(device, num_edges))


def rw_realedge_gpu_infeasible(device, num_edges: int) -> bool:
    """Return True if real-edge GPU sampling for (device, num_edges) is known infeasible."""
    return _rw_realedge_key(device, num_edges) in _RW_REALEDGE_GPU_INFEASIBLE


def snapshot_rw_device_state():
    """Copy every piece of run-level rw/real-edge device state that a profiling
    probe can mutate: the forced rw-on-GPU flag, the sticky real-edge
    GPU-infeasible set, and the device-decision cache (which also holds the
    sticky "rw_gpu_oom" records). Pair with restore_rw_device_state()."""
    return (
        bool(_FORCE_RW_ON_GPU),
        set(_RW_REALEDGE_GPU_INFEASIBLE),
        dict(_RW_DEVICE_DECISION_CACHE),
    )


def restore_rw_device_state(snapshot) -> bool:
    """Restore a snapshot_rw_device_state() snapshot in place.

    Returns True if, since the snapshot, a GPU build fell back to CPU -- i.e. a
    new real-edge GPU-infeasible mark or a new sticky rw-GPU-OOM record
    appeared. A probe that hit such a fallback did not measure the GPU path.
    """
    force, infeasible, decisions = snapshot
    fell_back = bool(_RW_REALEDGE_GPU_INFEASIBLE - infeasible) or any(
        isinstance(k, tuple) and k and k[0] == "rw_gpu_oom" and k not in decisions
        for k in _RW_DEVICE_DECISION_CACHE
    )
    set_force_rw_on_gpu(force)
    _RW_REALEDGE_GPU_INFEASIBLE.clear()
    _RW_REALEDGE_GPU_INFEASIBLE.update(infeasible)
    _RW_DEVICE_DECISION_CACHE.clear()
    _RW_DEVICE_DECISION_CACHE.update(decisions)
    return fell_back


def rw_realedge_gpu_infeasible_any() -> bool:
    """Return True if any real-edge GPU sampling fallback was recorded this run."""
    return len(_RW_REALEDGE_GPU_INFEASIBLE) > 0


def _expandable_segments_enabled() -> bool:
    """Return True if PYTORCH_CUDA_ALLOC_CONF enables expandable_segments."""
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    return "expandable_segments:True" in conf or "expandable_segments:true" in conf


def _rw_log_is_rank0() -> bool:
    """Rank-0 detection for modules that don't carry args. Falls back to
    True when torch.distributed isn't initialised (single-process runs).
    """
    try:
        import torch.distributed as _dist
        if not _dist.is_available() or not _dist.is_initialized():
            return True
        return _dist.get_rank() == 0
    except Exception:
        return True


def _gpu_concurrent_overhead_bytes(device) -> int:
    """Per-process CUDA library overhead.

    Fixed-ish startup cost for cuBLAS, cuDNN, curand, kernel cache, CUDA
    stream metadata, and optional libs (flash-attention etc.). Independent
    of dataset and only weakly dependent on CUDA / cuDNN version.

    Line-item breakdown (NOT a multiplier):
      - cuBLAS handle + workspace : ~50 MiB / stream × ≤ 4 streams ≤ 200 MiB
        (https://docs.nvidia.com/cuda/cublas/#cublas-context)
      - cuDNN handle + default workspace : up to ~256 MiB
        (https://docs.nvidia.com/deeplearning/cudnn/api/index.html)
      - curand / kernel cache / stream metadata : ~50 MiB combined
      - Headroom for optional libs / future CUDA-version upgrades : ~50-100 MiB
    Bundled conservative ceiling: 512 MiB.

    Note: PyTorch caching allocator fragmentation is *intentionally not*
    subtracted here. ``reserved - allocated`` is memory the allocator
    holds and can hand out to new tensors from its internal pool — it's
    NOT off-limits to RW build. Treating it as overhead would
    double-count and force needless CPU fallbacks at training time
    (when caching allocator naturally holds large reserved blocks across
    step boundaries). If the cached blocks happen to be too fragmented
    to satisfy a large contiguous request, the reactive OOM handler in
    the caller takes over.

    Returns 0 for non-CUDA devices.
    """
    try:
        torch_device = torch.device(device)
    except (TypeError, RuntimeError):
        return 0
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return 0
    return 512 * 1024 * 1024


def _rw_gpu_usable_bytes(device) -> int:
    """Bytes usable for an RW build on *device*, after reserved-slack and
    per-process-cap corrections. Returns ``-1`` for non-CUDA / unavailable
    devices (i.e. "no GPU memory limit applies").

    This is the SINGLE source of truth for "usable GPU memory for RW": both the
    fit decision (:func:`_gpu_can_fit_rw`) and the rank-0 device-decision logs
    read it, so a logged number can never disagree with the decision. Previously
    the log re-read the *uncapped* physical free, producing nonsensical lines
    like "estimate 12535 MiB > usable 30215 MiB" while the decision was actually
    made against the much smaller per-process-cap headroom.
    """
    try:
        torch_device = torch.device(device)
    except (TypeError, RuntimeError):
        return -1
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return -1
    try:
        free, _ = torch.cuda.mem_get_info(torch_device)
    except (RuntimeError, AttributeError):
        return -1
    # Add the caching allocator's reserved-but-unallocated slack only when the
    # driver still has non-trivial physical free to grow into (>= 256 MiB);
    # otherwise fragmented slack cannot satisfy a large contiguous request.
    _MIN_DRIVER_FREE_FOR_SLACK = 256 * 1024 * 1024  # 256 MiB
    if _expandable_segments_enabled() and int(free) >= _MIN_DRIVER_FREE_FOR_SLACK:
        try:
            reserved_slack = max(
                0,
                torch.cuda.memory_reserved(torch_device) - torch.cuda.memory_allocated(torch_device),
            )
            free = free + reserved_slack
        except Exception:
            pass
    # A per-process memory cap (e.g. the planner's HBM limit) is invisible to
    # mem_get_info, so clamp free to the cap headroom (cap - reserved); new
    # allocations beyond the pool's free blocks must come from there.
    if _GPU_PROCESS_MEMORY_CAP_BYTES > 0:
        try:
            reserved = torch.cuda.memory_reserved(torch_device)
            cap_headroom = max(0, _GPU_PROCESS_MEMORY_CAP_BYTES - reserved)
            free = min(int(free), cap_headroom)
        except Exception:
            pass
    overhead = _gpu_concurrent_overhead_bytes(torch_device)
    return max(0, int(free) - overhead)


def _gpu_can_fit_rw(device, need_bytes: int) -> bool:
    """Return True iff *device* has room for a need_bytes RW build.

    Returns True for non-CUDA / unavailable devices, letting the reactive
    try/except in callers catch any residual fragmentation OOM.
    """
    # Planner override: a rw-on-GPU plan was selected (its transient peak was
    # profiled feasible under the HBM cap), so keep rw on GPU regardless of the
    # conservative cap-clamped headroom. A real OOM is still caught reactively.
    if _FORCE_RW_ON_GPU:
        try:
            if torch.device(device).type == "cuda" and torch.cuda.is_available():
                return True
        except (TypeError, RuntimeError):
            pass
    usable = _rw_gpu_usable_bytes(device)
    if usable < 0:
        return True
    return int(need_bytes) <= usable


def _safe_rw_device(rw_dev, args, num_nodes: int, num_edges: int):
    """Decide between GPU and CPU for a single build_head_hop_edges call.

    If the requested rw_dev is CUDA but the estimated RW working set would
    leave too little free memory for the rest of the step, fall back to CPU
    proactively (and log on rank 0). This avoids the partial-allocation-then-
    OOM pattern that fragments the caching allocator before the reactive
    `except RuntimeError` path can take over.
    """
    requested = torch.device(rw_dev)
    if requested.type != "cuda":
        return requested
    estimate = _estimate_rw_working_set_bytes(
        num_nodes=num_nodes,
        num_edges=num_edges,
        walks_per_node=int(getattr(args, "walks_per_node", 2)),
        walk_length=int(getattr(args, "walk_length", 4)),
        num_heads=int(getattr(args, "num_heads", 1)),
    )
    if _gpu_can_fit_rw(requested, estimate):
        return requested
    rank = int(getattr(args, "rank", 0)) if args is not None else 0
    usable_mib = _rw_gpu_usable_bytes(requested) / (1024 ** 2)
    cache_key = ("_safe_rw_device", str(requested), int(num_edges), "cpu")
    prev = _RW_DEVICE_DECISION_CACHE.get(cache_key)
    if prev is None and rank == 0:
        print(
            f"[rw-device] estimated working set {estimate/(1024**2):.0f} MiB "
            f"exceeds the usable GPU budget {usable_mib:.0f} MiB "
            f"(physical free minus reserved and the planner HBM cap, "
            f"num_edges={num_edges:,}); forcing CPU build to avoid OOM. "
            f"To reduce false CPU fallbacks, "
            f"set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. "
            f"(Further matching calls suppressed.)"
        )
    _RW_DEVICE_DECISION_CACHE[cache_key] = (estimate, usable_mib)
    return torch.device("cpu")


def get_seed_batch_size(args) -> int:
    seed_batch_size = getattr(args, "seed_batch_size", None)
    if seed_batch_size is None:
        seed_batch_size = getattr(args, "seq_len")
    return max(1, int(seed_batch_size))


@contextlib.contextmanager
def fixed_random_seed(seed: int):
    """Temporarily set RNG seeds and restore states afterward."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@contextlib.contextmanager
def fixed_random_seed_cpu(seed: int):
    """Temporarily set CPU-side RNG seeds without touching CUDA RNG state."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed would also reseed every CUDA generator (and only the
    # CPU state is restored below); seed the CPU default generator directly.
    torch.default_generator.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)


def adjust_edge_index_nomerge(edge_index, sub_seq_len):
    new_index = edge_index.clone()
    mask = edge_index > 0
    new_index[mask] = edge_index[mask] + ((edge_index[mask].float() - 1) // sub_seq_len).to(torch.int64)

    return new_index


def pad_y(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = torch.full((padlen, ), -100, dtype=x.dtype, device=x.device)
        new_x[:xlen] = x
        x = new_x
    # x = torch.cat([x, torch.full((addlen, ), -100, dtype=x.dtype, device=x.device)], dim=0)
    return x


def pad_2d(x, padlen):
    # xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    # ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
    # ts = torch.nn.utils.rnn.pad_sequence(ts, batch_first=True, padding_value=0)
    # indexes = torch.stack(indexes, dim=0)

    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x


def pad_attn_bias(x, padlen):
    seq_parallel_world_size = get_sequence_parallel_world_size()
    xlen = x.size(0)
    new_x = x.new_zeros([padlen, padlen, x.size(2)], dtype=x.dtype)
    new_x[:xlen, :xlen, :] = x
    x = new_x
    return x


def random_split_idx(data_y, frac_train, frac_valid, frac_test, seed):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    random.seed(seed)
    all_idx = np.arange(data_y.shape[0])
    random.shuffle(all_idx)
    train_idx = all_idx[:int(frac_train * data_y.shape[0])]
    val_idx = all_idx[int(frac_train * data_y.shape[0]):int((frac_train+frac_valid) * data_y.shape[0])]
    test_idx = all_idx[int((frac_train+frac_valid) * data_y.shape[0]):]
    split_idx = {'train': torch.tensor(train_idx),
                'valid': torch.tensor(val_idx),
                'test': torch.tensor(test_idx)}
    # print(f"Train nodes: {len(train_idx)}, Valid nodes: {len(val_idx)}, Test nodes: {len(test_idx)}")
    return split_idx


def gen_sub_edge_index(edge_index, idx_batch, N):
    """
    Get sub edge_index according to given sequence nodes
        
    Arguments:
        edge_index (Tensor): original edge_index of the whole graph
        idx_batch (Tensor): training node indexes of a batch
        N (Int): number of nodes in the whole graph
    """
    adj = edge_index
    edge_index_i, _ = subgraph(idx_batch, adj, num_nodes=N, relabel_nodes=True)

    return edge_index_i


def get_batch(args, x, y, idx_batch, adjs, rest_split_sizes, device):
    """Generate a local subsequence in sequence parallel
    """

    # For sequence parallel
    seq_parallel_world_size = get_sequence_parallel_world_size() if sequence_parallel_is_initialized() else 1
    seq_parallel_world_rank = get_sequence_parallel_rank() if sequence_parallel_is_initialized() else 0
    seq_length = args.seq_len

    assert seq_length % seq_parallel_world_size == 0
    sub_seq_length = seq_length // seq_parallel_world_size
    sub_seq_start = seq_parallel_world_rank * sub_seq_length
    sub_seq_end = (seq_parallel_world_rank + 1) * sub_seq_length

    x_i = x[idx_batch]
    y_i = y[idx_batch]
    attn_bias = torch.cat([torch.tensor(i[idx_batch.cpu(), :][:, idx_batch.cpu()].toarray(), dtype=torch.float32).unsqueeze(0) for i in adjs])
    attn_bias = attn_bias.permute(1, 2, 0) # [s, s, d]

    if idx_batch.shape[0] < seq_length:
        
        assert rest_split_sizes is not None, 'split_sizes should not be None'
        x_i_list = [t for t in torch.split(x_i, rest_split_sizes, dim=0)]
        y_i_list = [t for t in torch.split(y_i, rest_split_sizes, dim=0)]
        attn_bias_list = [t for t in torch.split(attn_bias, rest_split_sizes, dim=0)]

        padlen = max(rest_split_sizes)
        x_i_list_pad = []
        y_i_list_pad = []
        attn_bias_list_pad = []
        for i in range(len(x_i_list)):
            x_i_list_pad.append(pad_2d(x_i_list[i], padlen))
            y_i_list_pad.append(pad_y(y_i_list[i], padlen))
            attn_bias_list_pad.append(pad_attn_bias(attn_bias_list[i], padlen))
        last_batch_flag(True)
        
        return x_i_list_pad[seq_parallel_world_rank].to(device), y_i_list_pad[seq_parallel_world_rank].to(device), attn_bias_list_pad[seq_parallel_world_rank].to(device)
    
    else:
        x_i = x_i[sub_seq_start:sub_seq_end, :].to(device)
        y_i = y_i[sub_seq_start:sub_seq_end].to(device)
        attn_bias = attn_bias[sub_seq_start:sub_seq_end, :, :].to(device) 
        last_batch_flag(False)
        
        return x_i, y_i, attn_bias
    

def _apply_to_edge_index(edge_index, fn):
    if isinstance(edge_index, list):
        return [fn(e) if e is not None else None for e in edge_index]
    if edge_index is None:
        return None
    return fn(edge_index)


def _merge_edge_index_list(edge_list: List[Tensor]) -> Tensor:
    """Merge a list of edge_index tensors into one unique edge_index."""
    tensors = [e for e in edge_list if e is not None and e.numel() > 0]
    if not tensors:
        return edge_list[0] if edge_list else None
    if len(tensors) == 1:
        return tensors[0]
    device = tensors[0].device
    edge_index = torch.cat(tensors, dim=1).to(torch.long)
    num_nodes = int(max(edge_index[0].max().item(), edge_index[1].max().item())) + 1
    return coalesce(edge_index, num_nodes=num_nodes).to(device)


def _is_strictly_sorted_coo(edge_index: Tensor, num_nodes: int, chunk: int = 1 << 24) -> bool:
    """True iff edge_index is strictly increasing in (row, col) -- i.e. already
    coalesced, so SparseTensor(...).coalesce() returns it without a copy.
    Checked in chunks so the transient stays at a few chunk-sized int64 tensors."""
    E = int(edge_index.size(1))
    if E <= 1:
        return True
    row, col = edge_index[0], edge_index[1]
    N = int(num_nodes)
    prev = None
    for s0 in range(0, E, chunk):
        s1 = min(E, s0 + chunk)
        key = row[s0:s1].to(torch.int64) * N + col[s0:s1].to(torch.int64)
        if prev is not None and not bool(key[0] > prev):
            return False
        if key.numel() > 1 and not bool((key[1:] > key[:-1]).all()):
            return False
        prev = key[-1]
        del key
    return True


_SORTED_COO_CACHE: dict = {}


def _rw_gpu_csr_needs_col_copy(edge_index: Tensor, num_nodes: int, device) -> bool:
    """True iff a CSR for edge_index built on *device* holds its own copy of
    col, i.e. unless edge_index already lives there strictly sorted, in which
    case _get_random_walk_graph aliases col to edge_index[1]."""
    if edge_index.device != torch.device(device):
        return True
    key = (int(edge_index.data_ptr()), int(edge_index.size(1)), int(num_nodes))
    if key not in _SORTED_COO_CACHE:
        _SORTED_COO_CACHE[key] = _is_strictly_sorted_coo(edge_index, num_nodes)
    return not _SORTED_COO_CACHE[key]


def _get_random_walk_graph(edge_index: Tensor, num_nodes: int, device) -> tuple[Tensor, Tensor, Tensor]:
    """Return (placeholder, rowptr, col) for the given edge_index on *device*.

    Only rowptr and col (CSR) are cached — the COO edge_index_dev is NOT
    stored because col is identical to the coalesced edge_index[1] and row
    can be reconstructed from rowptr on demand (saves ~2×E×8 bytes per cache
    entry, ≈ 4 GiB on amazon).  Callers receive a zero-shape placeholder
    tensor for ``edge_index_dev`` so that ``.new_zeros()`` and device-pinning
    still work; ``_run_random_walk`` uses only rowptr/col for the fast CSR
    path and reconstructs row from rowptr for the rare COO fallback.

    If a CPU CSR is already cached and *device* is CUDA, rowptr and col are
    H2D-transferred instead of rebuilt from scratch (avoids the 6×E×8-byte
    sort/argsort pipeline).
    """
    dev = torch.device(device)
    key = (int(edge_index.data_ptr()), int(edge_index.size(1)), int(num_nodes), dev.type, dev.index)
    cached = _RANDOM_WALK_GRAPH_CACHE.get(key)
    if cached is not None:
        return cached

    # GPU target: if CPU CSR already built, H2D-transfer rowptr+col only --
    # unless the source edge_index already lives on this GPU (gpu_persist
    # topology) and is strictly sorted by (row, col). Then coalesce returns the
    # input, so building here makes col a view of edge_index[1] and the GPU CSR
    # costs only rowptr; the H2D path would add a second E x 8-byte copy of col
    # (2 GiB on amazon) that stays resident through the model step. An unsorted
    # edge list would instead pay a full on-GPU sort, so it keeps the H2D path.
    # Same edge list either way, so rowptr/col -- and the walks -- are identical.
    if dev.type == "cuda" and not (edge_index.device == dev and _is_strictly_sorted_coo(edge_index, num_nodes)):
        cpu_key = (int(edge_index.data_ptr()), int(edge_index.size(1)), int(num_nodes), "cpu", None)
        cpu_cached = _RANDOM_WALK_GRAPH_CACHE.get(cpu_key)
        if cpu_cached is not None:
            _, rowptr_cpu, col_cpu = cpu_cached
            rowptr = rowptr_cpu.to(dev)
            col = col_cpu.to(dev)
            placeholder = torch.empty((2, 0), dtype=edge_index.dtype, device=dev)
            cached = (placeholder, rowptr, col)
            _RANDOM_WALK_GRAPH_CACHE[key] = cached
            return cached

    # Build CSR from scratch on the target device.
    edge_index_dev = edge_index if edge_index.device == dev else edge_index.to(dev)
    adj = SparseTensor(
        row=edge_index_dev[0],
        col=edge_index_dev[1],
        sparse_sizes=(num_nodes, num_nodes),
    ).coalesce()
    rowptr, col, _ = adj.csr()
    # Store a placeholder instead of edge_index_dev — see docstring.
    placeholder = torch.empty((2, 0), dtype=edge_index.dtype, device=dev)
    cached = (placeholder, rowptr, col)
    _RANDOM_WALK_GRAPH_CACHE[key] = cached
    return cached


def _map_global_nodes_to_local(nodes_global: Tensor, sampled_nodes: Tensor) -> Tensor:
    sampled_nodes = sampled_nodes.to(torch.long)
    nodes_global = nodes_global.to(torch.long)
    if nodes_global.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=sampled_nodes.device)
    sorted_nodes, order = torch.sort(sampled_nodes)
    sorted_local = torch.arange(sampled_nodes.numel(), dtype=torch.long, device=sampled_nodes.device)[order]
    pos = torch.searchsorted(sorted_nodes, nodes_global)
    pos = torch.clamp(pos, max=max(int(sorted_nodes.numel()) - 1, 0))
    return sorted_local[pos]


def sample_seed_rw_subgraph(
    edge_index: Tensor,
    seed_nodes: Tensor,
    num_nodes: int,
    device,
    walk_length: int,
    walks_per_node: int,
) -> tuple[Tensor, Tensor]:
    """Sample a seed-centric subgraph by random walks on the full graph.

    The returned sampled nodes keep seeds first, followed by visited context nodes.
    Edge semantics follow attention message flow: src provides k/v, dst is the query.
    We keep both walk trajectory edges (curr -> prev) and direct seed-query edges
    (visited -> seed).
    """
    seed_nodes = seed_nodes.to(torch.long).view(-1).cpu()
    if seed_nodes.numel() == 0:
        return seed_nodes, edge_index.new_zeros((2, 0), dtype=torch.long).cpu()

    walk_length = max(1, int(walk_length))
    walks_per_node = max(1, int(walks_per_node))
    rw_device = torch.device(device)
    edge_index_dev, rowptr, col = _get_random_walk_graph(edge_index, num_nodes, rw_device)

    seed_nodes_dev = seed_nodes.to(rw_device)
    starts = seed_nodes_dev.repeat_interleave(walks_per_node)
    walks = _run_random_walk(edge_index_dev, rowptr, col, starts, walk_length)
    starts_walk = walks[:, 0]

    visited_nodes = [seed_nodes_dev]
    edge_parts: List[Tensor] = []
    for d in range(1, walk_length + 1):
        curr = walks[:, d]
        prev = walks[:, d - 1]
        valid = (curr >= 0) & (prev >= 0)
        if not valid.any():
            continue
        curr_valid = curr[valid]
        prev_valid = prev[valid]
        seed_valid = starts_walk[valid]
        edge_parts.append(torch.stack([curr_valid, prev_valid], dim=0))
        edge_parts.append(torch.stack([curr_valid, seed_valid], dim=0))
        visited_nodes.append(curr_valid)

    if len(visited_nodes) > 1:
        context_nodes = torch.unique(torch.cat(visited_nodes[1:], dim=0))
        context_nodes = context_nodes[~torch.isin(context_nodes, seed_nodes_dev)]
        sampled_nodes = torch.cat([seed_nodes_dev, context_nodes], dim=0)
    else:
        sampled_nodes = seed_nodes_dev

    if edge_parts:
        sampled_edges_global = _merge_edge_index_list(edge_parts)
        edge_src_local = _map_global_nodes_to_local(sampled_edges_global[0], sampled_nodes)
        edge_dst_local = _map_global_nodes_to_local(sampled_edges_global[1], sampled_nodes)
        sampled_edge_index = torch.stack([edge_src_local, edge_dst_local], dim=0)
    else:
        sampled_edge_index = edge_index.new_zeros((2, 0), dtype=torch.long, device=rw_device)

    return sampled_nodes.cpu(), sampled_edge_index.cpu()


def compute_hop_buckets_random_walk(
    edge_index: Tensor,
    num_nodes: int,
    num_buckets: int,
    device: str = "cpu",
    walk_length: int = 4,
    walks_per_node: int = 2,
    min_hop: int = 1,
    seed: Optional[int] = None,
) -> List[Tensor]:
    """
    使用随机游走生成 hop buckets，bucket i 对应 hop i+1，
    最后一个 bucket 合并 >= num_buckets 的 hop.
    设置 min_hop=2 可跳过 1-hop 真实邻居，使 RW 池只包含远距节点。
    seed 非 None 时游走只由 seed 决定，与全局 RNG 状态和其他线程无关。
    """
    num_buckets = max(1, int(num_buckets))
    walk_length = max(1, int(walk_length))
    walks_per_node = max(1, int(walks_per_node))
    min_hop = max(1, int(min_hop))
    if edge_index.numel() == 0:
        return [edge_index.new_zeros((2, 0), dtype=edge_index.dtype) for _ in range(num_buckets)]

    inferred_nodes = int(edge_index.max().item()) + 1
    num_nodes = max(num_nodes, inferred_nodes)
    rw_device = torch.device(device)

    # Proactive GPU-OOM guard. The only O(E) GPU transient is the one-time
    # CSR build (SparseTensor coalesce sort, ~6 × E × 8 bytes; amazon ≈ 264M
    # edges → ~12 GiB); a walk on a cached CSR is O(N·W).  Falling back to
    # CPU when even that does not fit saves all caller paths
    # — multi_tier edge-policy profiling, probe, and training step — from
    # the partial-allocation-then-OOM pattern that fragments the allocator.
    #
    # Safety margin: reserve 10% of currently-free GPU for non-RW transients
    # (model forward, autograd graph, NCCL buffers). Logs are gated on
    # rank 0 and deduplicated against the previous decision so a stable
    # CPU-fallback regime does not flood stdout.
    if rw_device.type == "cuda":
        E_cur = int(edge_index.size(1))
        # Sticky OOM record: if a prior GPU walk OOM'd for this exact graph,
        # skip GPU entirely rather than re-evaluating every epoch.  The cold
        # estimate (~12.5 GiB for amazon) passes _gpu_can_fit_rw on a 30 GiB
        # card even at a 16 GiB cap, so the reactive OOM-then-fallback loop
        # would repeat forever without this sticky bypass.
        _oom_sticky_key = ("rw_gpu_oom", int(edge_index.data_ptr()), E_cur, int(num_nodes))
        if _oom_sticky_key in _RW_DEVICE_DECISION_CACHE:
            rw_device = torch.device("cpu")
        elif (
            _RW_GPU_CSR_REQUIRES_PLAN
            and not _FORCE_RW_ON_GPU
            and _rw_gpu_csr_needs_col_copy(edge_index, num_nodes, rw_device)
        ):
            # Unplanned persistent HBM (see _RW_GPU_CSR_REQUIRES_PLAN): walk on
            # the CPU, and drop a full-copy GPU CSR that an earlier rw-on-GPU
            # plan left behind, since the current plan counts that memory free.
            clear_random_walk_graph_cache(device_type="cuda", data_ptr=edge_index.data_ptr())
            cache_key = ("compute_hop_buckets_random_walk", str(rw_device), E_cur, "cpu-unplanned")
            if _RW_DEVICE_DECISION_CACHE.get(cache_key) is None and _rw_log_is_rank0():
                print(
                    f"[rw-device] compute_hop_buckets_random_walk: a GPU CSR would keep "
                    f"a {E_cur * 8 / (1024 ** 2):.0f} MiB copy of col resident through "
                    f"the model step and no rw-on-GPU plan is active; building "
                    f"random-walk edges on CPU. (Further matching calls suppressed.)"
                )
            _RW_DEVICE_DECISION_CACHE[cache_key] = True
            rw_device = torch.device("cpu")
        else:
            # Three-level estimate: cold (CSR built on GPU) → CPU CSR cached
            # (H2D of rowptr/col) → GPU CSR cached (walk temporaries only).
            _cuda_csr_key = (int(edge_index.data_ptr()), E_cur, num_nodes, rw_device.type, rw_device.index)
            _cpu_csr_key = (int(edge_index.data_ptr()), E_cur, num_nodes, "cpu", None)
            N_cur = max(1, int(num_nodes))
            W = max(1, int(walks_per_node))
            L = max(1, int(walk_length))
            # Three cases:
            #   GPU CSR cached  → rowptr/col already live on the GPU.
            #   CPU CSR cached  → rowptr/col are H2D'd ((N+1)×8 + E×8).
            #   Cold            → SparseTensor coalesce sort dominates.
            walk_and_buckets = (
                _rw_walk_transient_bytes(N_cur * W, L)
                + N_cur * W * 33              # per-d peak (mask+2×select+stack)
                + 2 * N_cur * W * L * 8       # bucket_cumulative
            )
            if _cuda_csr_key in _RANDOM_WALK_GRAPH_CACHE:
                estimate = int(walk_and_buckets)
            elif _cpu_csr_key in _RANDOM_WALK_GRAPH_CACHE:
                estimate = int((N_cur + 1) * 8 + E_cur * 8 + walk_and_buckets)
            else:
                estimate = _estimate_rw_working_set_bytes(
                    num_nodes=num_nodes,
                    num_edges=E_cur,
                    walks_per_node=walks_per_node,
                    walk_length=walk_length,
                )
            if not _gpu_can_fit_rw(rw_device, estimate):
                usable_mib = _rw_gpu_usable_bytes(rw_device) / (1024 ** 2)
                cache_key = ("compute_hop_buckets_random_walk", str(rw_device), E_cur, "cpu")
                prev = _RW_DEVICE_DECISION_CACHE.get(cache_key)
                if prev is None and _rw_log_is_rank0():
                    print(
                        f"[rw-device] compute_hop_buckets_random_walk: estimated "
                        f"working set {estimate/(1024**2):.0f} MiB exceeds the usable "
                        f"GPU budget {usable_mib:.0f} MiB (physical free minus reserved "
                        f"and the planner HBM cap), num_edges={E_cur:,}; forcing CPU "
                        f"build. NOTE: this overrides only the RW build device; topology "
                        f"policies such as gpu_persist may still cache edge_index_global "
                        f"on GPU. (Further matching calls suppressed.)"
                    )
                _RW_DEVICE_DECISION_CACHE[cache_key] = (estimate, usable_mib)
                rw_device = torch.device("cpu")
            else:
                # Symmetric log: the working set fits, so the random-walk build
                # stays on the GPU. Without this, only the CPU-offload case was
                # logged, making it look like rw was never confirmed on GPU.
                cache_key_gpu = ("compute_hop_buckets_random_walk", str(rw_device), E_cur, "gpu")
                if _RW_DEVICE_DECISION_CACHE.get(cache_key_gpu) is None and _rw_log_is_rank0():
                    print(
                        f"[rw-device] compute_hop_buckets_random_walk: estimated "
                        f"working set {estimate/(1024**2):.0f} MiB fits the GPU; "
                        f"building random-walk edges on {rw_device}. "
                        f"(Further matching calls suppressed.)"
                    )
                _RW_DEVICE_DECISION_CACHE[cache_key_gpu] = (estimate, -1.0)

    edge_index_dev, rowptr, col = _get_random_walk_graph(edge_index, num_nodes, rw_device)

    starts = torch.arange(num_nodes, device=rw_device, dtype=torch.long)
    starts = starts.repeat_interleave(walks_per_node)
    # Without an explicit seed, _run_random_walk draws one from the CPU RNG;
    # snapshot it so a CPU retry after a GPU failure replays the same walks.
    _walk_rng_state = torch.random.get_rng_state()
    try:
        walks = _run_random_walk(edge_index_dev, rowptr, col, starts, walk_length, seed=seed)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as _gpu_walk_exc:
        if rw_device.type != "cuda":
            raise
        # GPU walk failed (OOM).  Evict the stale GPU CSR entry, clear the
        # device decision cache so the next epoch re-evaluates, and retry the
        # whole walk on CPU.
        _bad_key = (
            int(edge_index.data_ptr()), int(edge_index.size(1)),
            int(num_nodes), rw_device.type, rw_device.index,
        )
        _RANDOM_WALK_GRAPH_CACHE.pop(_bad_key, None)
        # Store a sticky CPU-forced record so all subsequent epochs skip the
        # GPU attempt directly, without re-evaluating the estimate every time.
        # The cold estimate (~12.5 GiB on amazon) always passes _gpu_can_fit_rw
        # on a large-physical-memory card even under a 16 GiB cap, so without
        # this sticky entry the OOM → CPU-fallback loop would repeat forever.
        _oom_sticky_key = ("rw_gpu_oom", int(edge_index.data_ptr()), int(edge_index.size(1)), int(num_nodes))
        _RW_DEVICE_DECISION_CACHE[_oom_sticky_key] = True
        if _rw_log_is_rank0():
            print(
                f"[rw-device] GPU walk failed ({type(_gpu_walk_exc).__name__}); "
                f"evicting GPU CSR cache and retrying on CPU. "
                f"GPU walk suppressed for subsequent epochs (num_edges={int(edge_index.size(1)):,})."
            )
        rw_device = torch.device("cpu")
        edge_index_dev, rowptr, col = _get_random_walk_graph(edge_index, num_nodes, rw_device)
        starts = starts.to(rw_device)
        torch.random.set_rng_state(_walk_rng_state)
        walks = _run_random_walk(edge_index_dev, rowptr, col, starts, walk_length, seed=seed)

    buckets = [edge_index_dev.new_zeros((2, 0), dtype=edge_index_dev.dtype) for _ in range(num_buckets)]
    query_nodes = walks[:, 0]
    for d in range(min_hop, walk_length + 1):
        kv_nodes = walks[:, d]
        valid = kv_nodes >= 0
        if d > 1:
            prev = walks[:, d - 1]
            valid = valid & (kv_nodes != prev)
        if not valid.any():
            continue
        query_nodes_valid = query_nodes[valid]
        kv_nodes_valid = kv_nodes[valid]
        # edge_index[0] provides key/value nodes and edge_index[1] is the updated query node.
        edges = torch.stack([kv_nodes_valid, query_nodes_valid], dim=0)
        bucket_idx = min(d - 1, num_buckets - 1)
        buckets[bucket_idx] = torch.cat([buckets[bucket_idx], edges], dim=1)
    return buckets


def build_head_hop_edges(
    edge_index: Tensor,
    num_nodes: int,
    num_heads: int,
    num_groups: int,
    device: str = "cpu",
    walk_length: int = 4,
    walks_per_node: int = 2,
    min_hop: int = 1,
    seed: Optional[int] = None,
):
    """
    构建 per-head edge_index 列表。
    当前实现不再按 head 分桶：所有 head 共享同一份随机游走子图。
    返回单个共享 edge_index，避免调用方把同一张图复制多份后再去重。
    min_hop=2 可使 RW 池只包含 2-hop 以上的远距节点。
    """
    _ = num_heads
    _ = num_groups  # kept for backward-compatible call signature
    buckets = compute_hop_buckets_random_walk(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_buckets=1,
        device=device,
        walk_length=walk_length,
        walks_per_node=walks_per_node,
        min_hop=min_hop,
        seed=seed,
    )
    return buckets[0] if buckets else edge_index.new_zeros((2, 0), dtype=edge_index.dtype)


def compute_khop_pool_dgl_neighbor(
    edge_index: Tensor,
    num_nodes: int,
    fanout: List[int],
    device: str = "cpu",
    min_hop: int = 1,
    seed: int = None,
) -> Tensor:
    """DGL neighbor-sampler based k-hop edge pool (alternative to random walk).

    For every node ``q`` (treated as a query) this performs a fanout-bounded
    breadth expansion along *out*-edges (the same direction random walks follow):
    at hop ``d`` it samples up to ``fanout[d-1]`` successors of each current
    frontier node, tracking which query ``q`` every frontier node descends from.
    For hops ``>= min_hop`` it emits edges ``(neighbour -> q)`` so query ``q``
    attends to its sampled hop-``d`` nodes.

    fanout entry ``-1`` means "all neighbours" at that hop (exact k-hop; may OOM
    on dense graphs — that is expected/accepted for this baseline).

    Returns a ``(2, E)`` LongTensor with the SAME orientation as the RW pool:
    row 0 = key/value node, row 1 = query node.  No per-query budget truncation
    is applied here (downstream ``_resolve_rw_edges_for_state`` leaves the pool
    untouched unless random-block sampling is enabled).
    """
    import dgl

    # Seed DGL's sampler RNG so all SP ranks (which call this independently on
    # the rank-local build path) draw bit-identical samples. Same seed + same
    # graph + same OMP thread count per rank => identical edges, no broadcast.
    if seed is not None:
        dgl.seed(int(seed))
        if torch.device(device).type == "cpu":
            # CPU build: seed only the CPU default generator so callers'
            # CPU-only seed contexts stay free of CUDA RNG side effects.
            torch.default_generator.manual_seed(int(seed))
        else:
            torch.manual_seed(int(seed))

    rw_device = torch.device(device)
    k = len(fanout)
    if edge_index.numel() == 0 or k == 0:
        return edge_index.new_zeros((2, 0), dtype=torch.long)

    inferred_nodes = int(edge_index.max().item()) + 1
    N = max(int(num_nodes), inferred_nodes)
    src = edge_index[0].to(rw_device).long()
    dst = edge_index[1].to(rw_device).long()
    g = dgl.graph((src, dst), num_nodes=N, device=rw_device)

    # Frontier carried as two parallel tensors so provenance (origin query) is
    # tracked across hops without materialising per-query subgraphs.
    origin = torch.arange(N, device=rw_device, dtype=torch.long)   # which query
    current = torch.arange(N, device=rw_device, dtype=torch.long)  # current node
    src_parts: List[Tensor] = []
    dst_parts: List[Tensor] = []

    for d in range(1, k + 1):
        if current.numel() == 0:
            break
        f = int(fanout[d - 1])
        uniq = torch.unique(current)
        # Sample out-edges (node -> successor) for the unique frontier nodes.
        sg = dgl.sampling.sample_neighbors(g, uniq, f, edge_dir="out")
        su, sv = sg.edges()
        su = su.long()
        sv = sv.long()
        if su.numel() == 0:
            break
        # Group sampled successors by their source node id.
        deg = torch.bincount(su, minlength=N)            # successors sampled per node
        order = torch.argsort(su, stable=True)
        sv_sorted = sv[order]
        node_offset = torch.zeros(N + 1, dtype=torch.long, device=rw_device)
        node_offset[1:] = torch.cumsum(deg, dim=0)
        # Expand each frontier entry to all sampled successors of its node.
        cnt = deg[current]                               # successors for each entry
        keep = cnt > 0
        if not bool(keep.any()):
            break
        origin_k = origin[keep]
        current_k = current[keep]
        cnt_k = cnt[keep]
        total = int(cnt_k.sum().item())
        # Ragged gather: entry i contributes sv_sorted[off[c]:off[c]+cnt] for c=current_k[i].
        seg_start = node_offset[current_k].repeat_interleave(cnt_k)
        entry_start = torch.zeros_like(cnt_k)
        if cnt_k.numel() > 1:
            entry_start[1:] = torch.cumsum(cnt_k, dim=0)[:-1]
        within = torch.arange(total, device=rw_device) - entry_start.repeat_interleave(cnt_k)
        gather_idx = seg_start + within
        new_nodes = sv_sorted[gather_idx]                # successor per expanded entry
        new_origin = origin_k.repeat_interleave(cnt_k)

        if d >= min_hop:
            nz = new_nodes != new_origin                 # drop trivial self edges
            src_parts.append(new_nodes[nz])              # key/value
            dst_parts.append(new_origin[nz])             # query
        origin = new_origin
        current = new_nodes

    if not src_parts:
        return edge_index.new_zeros((2, 0), dtype=torch.long)
    return torch.stack([torch.cat(src_parts), torch.cat(dst_parts)], dim=0).to(torch.long)



def _run_random_walk(
    edge_index: Tensor,
    rowptr: Tensor,
    col: Tensor,
    starts: Tensor,
    walk_length: int,
    seed: Optional[int] = None,
) -> Tensor:
    """Uniform random walks over the CSR graph ``(rowptr, col)``.

    Returns ``[len(starts), walk_length + 1]`` with ``walks[:, 0] == starts``.
    A walk that reaches a node with out-degree 0 stays on it for the remaining
    steps (the dead-end convention the DGL and torch_cluster paths used).

    Determinism: all uniforms come from a private CPU generator, so the walks
    are independent of OMP thread count and of the device the walk runs on
    (uniforms are drawn on the CPU and moved to ``rowptr.device``; a CPU and a
    GPU build with the same seed give identical walks).  The generator's seed
    is the first draw of a generator seeded with ``seed`` -- the same value
    the default generator yields right after ``manual_seed(seed)``, so passing
    ``seed=edge_seed`` reproduces the walks built inside
    ``_fixed_torch_cpu_seed(edge_seed)`` bit for bit.  With ``seed=None`` the
    draw comes from the process-global CPU RNG instead, which other threads
    can reseed concurrently (the prefetch worker and eval both do), so only an
    explicit seed is thread-safe.  The DGL sampler this replaces was not
    reproducible under a fixed seed with more than one thread
    (docs/experiment_log.md, E31), and torch_cluster >= 1.6 re-sorts all E
    edges on every call.

    Working set is O(len(starts)) per step; nothing here is O(E).
    ``edge_index`` is unused and kept for call-site compatibility.
    """
    _ = edge_index
    device = rowptr.device
    num_edges = int(col.numel())
    seed_gen = None
    if seed is not None:
        seed_gen = torch.Generator(device="cpu")
        seed_gen.manual_seed(int(seed))
    walk_seed = int(torch.randint(0, 2**62, (1,), dtype=torch.int64, generator=seed_gen).item())
    gen = torch.Generator(device="cpu")
    gen.manual_seed(walk_seed)

    cur = starts.to(device=device, dtype=torch.long)
    walks = torch.empty((cur.numel(), int(walk_length) + 1), dtype=torch.long, device=device)
    walks[:, 0] = cur
    for step in range(1, int(walk_length) + 1):
        lo = rowptr[cur]
        deg = rowptr[cur + 1] - lo
        u = torch.rand(cur.numel(), generator=gen, dtype=torch.float64)
        u = u.to(device=device, non_blocking=True)
        # floor(u * deg) is uniform on [0, deg); the clamp only guards the
        # float64 rounding edge case u * deg == deg.
        offset = torch.minimum((u * deg).long(), (deg - 1).clamp_(min=0))
        if num_edges > 0:
            nxt = col[(lo + offset).clamp_(max=num_edges - 1)]
            cur = torch.where(deg > 0, nxt, cur)
        walks[:, step] = cur
    return walks


def get_batch_blockize(
    args,
    x,
    y,
    idx_batch,
    rest_split_sizes,
    edge_index,
    N,
    device=None,
    seed_label_mask=None,
    force_induced_edges: bool = False,
    apply_graphormer_virtual_edges: bool = True,
):
    """
    Generate a local subsequence in sequence parallel and slice the corresponding edge_index.
    """
    seq_parallel_world_size = get_sequence_parallel_world_size() if sequence_parallel_is_initialized() else 1
    seq_parallel_world_rank = get_sequence_parallel_rank() if sequence_parallel_is_initialized() else 0
    seq_length = args.seq_len
    batch_mode = getattr(args, "batch_subgraph_mode", "induced")
    model_name = str(getattr(args, "model", "")).lower()
    dynamic_split_sizes = None
    dev = None
    sampled_nodes = None
    induced_edge_index_i_raw = None
    rw_edge_index_i_raw = None
    if device is not None:
        dev = torch.device(device) if not isinstance(device, torch.device) else device

    if force_induced_edges:
        x_i = x[idx_batch]
        y_i = y[idx_batch]
        induced_edge_index_i_raw = gen_sub_edge_index(edge_index, idx_batch, N)
    elif batch_mode == "seed_rw":
        if seq_parallel_world_size > 1:
            src_rank = get_sequence_parallel_src_rank()
            group = get_sequence_parallel_group()
        else:
            src_rank = None
            group = None

        do_sample = (seq_parallel_world_size <= 1) or (dist.get_rank() == src_rank)
        if do_sample:
            sampled_nodes, rw_edge_index_i_raw = sample_seed_rw_subgraph(
                edge_index=edge_index,
                seed_nodes=idx_batch,
                num_nodes=N,
                device=resolve_edge_build_device(args, device if device is not None else edge_index.device),
                walk_length=getattr(args, "walk_length", 4),
                walks_per_node=getattr(args, "walks_per_node", 2),
            )
        else:
            sampled_nodes = None
            rw_edge_index_i_raw = None

        if seq_parallel_world_size > 1:
            assert dev is not None, "seed_rw broadcasting requires a concrete device in sequence parallel mode."
            broad_device = dev
            if do_sample:
                sizes_broad = torch.tensor(
                    [
                        int(sampled_nodes.size(0)),
                        int(rw_edge_index_i_raw.size(1)),
                    ],
                    device=broad_device,
                    dtype=torch.long,
                )
            else:
                sizes_broad = torch.empty(2, device=broad_device, dtype=torch.long)
            dist.broadcast(sizes_broad, src_rank, group=group)

            sampled_node_count = int(sizes_broad[0].item())
            sampled_edge_count = int(sizes_broad[1].item())
            if do_sample:
                sampled_nodes_broad = sampled_nodes.to(broad_device)
                edge_index_broad = rw_edge_index_i_raw.to(broad_device)
            else:
                sampled_nodes_broad = torch.empty(sampled_node_count, device=broad_device, dtype=torch.long)
                edge_index_broad = torch.empty((2, sampled_edge_count), device=broad_device, dtype=torch.long)
            dist.broadcast(sampled_nodes_broad, src_rank, group=group)
            dist.broadcast(edge_index_broad, src_rank, group=group)
            sampled_nodes = sampled_nodes_broad.cpu()
            rw_edge_index_i_raw = edge_index_broad.cpu()

        x_i = x[sampled_nodes]
        y_i = y.new_full((sampled_nodes.size(0),) + tuple(y.shape[1:]), -100)
        seed_labels = y[idx_batch].clone()
        if seed_label_mask is not None:
            seed_mask_cpu = seed_label_mask.to(torch.bool).cpu()
            seed_labels[~seed_mask_cpu] = -100
        y_i[:idx_batch.shape[0]] = seed_labels
        induced_edge_index_i_raw = gen_sub_edge_index(edge_index, sampled_nodes, N)

        split_sizes = [t.shape[0] for t in torch.tensor_split(torch.zeros(sampled_nodes.size(0)), seq_parallel_world_size, dim=0)]
        dynamic_split_sizes = split_sizes
        sub_real_seq_len = max(split_sizes) + args.num_global_node
        set_last_batch_global_token_indices(
            list(range(0, seq_parallel_world_size * sub_real_seq_len, sub_real_seq_len))
        )
    else:
        x_i = x[idx_batch]  # [s, x_d]
        y_i = y[idx_batch]  # [s]
        # Keep raw (0-indexed, before fix_edge_index) subgraph for local_spd and RW sampling.
        induced_edge_index_i_raw = gen_sub_edge_index(edge_index, idx_batch, N)

    # --- attn_bias (optional, controlled by args.attn_bias_mode) ---
    _attn_bias_mode = getattr(args, 'attn_bias_mode', 'none')
    full_attn_bias = None  # [n_real, n_real, d] on CPU (before SP split)
    if _attn_bias_mode == 'local_spd':
        _n_real = x_i.shape[0]
        _max_dist = getattr(args, 'attn_bias_max_dist', 5)
        if _n_real > 16384:
            import warnings
            warnings.warn(
                f"[local_spd] seq_len={_n_real} is large; BFS for {_n_real}x{_n_real} may be slow. "
                "Consider using --attn_bias_mode none or reducing --seq_len."
            )
        full_attn_bias = _compute_local_spd_bias(
            induced_edge_index_i_raw.cpu(), n_nodes=_n_real, max_dist=_max_dist,
        )  # [n_real, n_real, d]

    attn_bias = None
    if force_induced_edges:
        edge_index_i_heads = induced_edge_index_i_raw
    elif model_name == "gt":
        edge_parts = [induced_edge_index_i_raw]
        if batch_mode == "seed_rw":
            if rw_edge_index_i_raw is not None and rw_edge_index_i_raw.numel() > 0:
                edge_parts.append(rw_edge_index_i_raw)
            edge_index_i_heads = _merge_edge_index_list(edge_parts)
        else:
            rw_base = induced_edge_index_i_raw
            rw_dev = resolve_edge_build_device(args, dev if dev is not None else rw_base.device)
            # Proactive GPU-OOM guard: if the estimated RW working set won't
            # fit in the current free GPU pool with a 15% safety margin,
            # build on CPU instead of OOM'ing and fragmenting the allocator.
            rw_dev = _safe_rw_device(
                rw_dev, args, num_nodes=x_i.size(0),
                num_edges=int(rw_base.size(1)) if rw_base is not None else 0,
            )
            if rw_base.device != rw_dev:
                rw_base = rw_base.to(rw_dev)
            try:
                rw_edge_index_i_raw = build_head_hop_edges(
                    edge_index=rw_base,
                    num_nodes=x_i.size(0),
                    num_heads=args.num_heads,
                    num_groups=1,
                    device=rw_base.device,
                    walk_length=getattr(args, "walk_length", 4),
                    walks_per_node=getattr(args, "walks_per_node", 2),
                )
            except torch.cuda.OutOfMemoryError:
                # Pre-check under-estimated; fall back to CPU.
                if int(getattr(args, "rank", 0)) == 0:
                    print("[rw-device] pre-check passed but build still OOM'd; retrying on CPU.")
                rw_base = rw_base.to("cpu")
                torch.cuda.empty_cache()
                rw_edge_index_i_raw = build_head_hop_edges(
                    edge_index=rw_base,
                    num_nodes=x_i.size(0),
                    num_heads=args.num_heads,
                    num_groups=1,
                    device=rw_base.device,
                    walk_length=getattr(args, "walk_length", 4),
                    walks_per_node=getattr(args, "walks_per_node", 2),
                )
            if rw_edge_index_i_raw is not None and rw_edge_index_i_raw.numel() > 0:
                edge_parts.append(rw_edge_index_i_raw)
            edge_index_i_heads = _merge_edge_index_list(edge_parts)
    elif batch_mode == "seed_rw":
        edge_index_i_heads = rw_edge_index_i_raw
    else:
        rw_base = induced_edge_index_i_raw
        rw_dev = resolve_edge_build_device(args, dev if dev is not None else rw_base.device)
        # Proactive GPU-OOM guard — see comment above.
        rw_dev = _safe_rw_device(
            rw_dev, args, num_nodes=x_i.size(0),
            num_edges=int(rw_base.size(1)) if rw_base is not None else 0,
        )
        if rw_base.device != rw_dev:
            rw_base = rw_base.to(rw_dev)
        try:
            edge_index_i_heads = build_head_hop_edges(
                edge_index=rw_base,
                num_nodes=x_i.size(0),
                num_heads=args.num_heads,
                num_groups=1,
                device=rw_base.device,
                walk_length=getattr(args, "walk_length", 4),
                walks_per_node=getattr(args, "walks_per_node", 2),
            )
        except torch.cuda.OutOfMemoryError:
            if int(getattr(args, "rank", 0)) == 0:
                print("[rw-device] pre-check passed but build still OOM'd; retrying on CPU.")
            rw_base = rw_base.to("cpu")
            torch.cuda.empty_cache()
            edge_index_i_heads = build_head_hop_edges(
                edge_index=rw_base,
                num_nodes=x_i.size(0),
                num_heads=args.num_heads,
                num_groups=1,
                device=rw_base.device,
                walk_length=getattr(args, "walk_length", 4),
                walks_per_node=getattr(args, "walks_per_node", 2),
            )

    if args.model == "graphormer" and apply_graphormer_virtual_edges:
        edge_index_i_heads = _apply_to_edge_index(
            edge_index_i_heads,
            lambda e: fix_edge_index(e, x_i.shape[0]),
        )

    # Current head-hop implementation shares the same subgraph across heads.
    # Merge to one tensor early to avoid per-head broadcast/memory overhead.
    if isinstance(edge_index_i_heads, list):
        edge_index_i_heads = _merge_edge_index_list(edge_index_i_heads)

    if dev is not None and edge_index_i_heads.device != dev:
        edge_index_i_heads = edge_index_i_heads.to(dev)
    if seq_parallel_world_size > 1:
        src_rank = get_sequence_parallel_src_rank()
        group = get_sequence_parallel_group()
        if dist.get_rank() == src_rank:
            size_broad = torch.tensor([edge_index_i_heads.size(1)], device=edge_index_i_heads.device, dtype=torch.long)
        else:
            size_broad = torch.empty(1, device=edge_index_i_heads.device, dtype=torch.long)
        dist.broadcast(size_broad, src_rank, group=group)
        if dist.get_rank() != src_rank:
            edge_index_i_heads = torch.empty((2, int(size_broad.item())), device=edge_index_i_heads.device, dtype=edge_index_i_heads.dtype)
        dist.broadcast(edge_index_i_heads, src_rank, group=group)

    if force_induced_edges:
        split_sizes = rest_split_sizes if idx_batch.shape[0] < seq_length else None
    elif batch_mode == "seed_rw":
        split_sizes = dynamic_split_sizes
    else:
        split_sizes = rest_split_sizes if idx_batch.shape[0] < seq_length else None

    if split_sizes is not None:
        seq_length = max(split_sizes) * seq_parallel_world_size
        x_i = pad_2d(x_i, seq_length)
        y_i = pad_y(y_i, seq_length)

        afterpad_split_sizes = [max(split_sizes)] * seq_parallel_world_size
        x_i_list = [t for t in torch.split(x_i, afterpad_split_sizes, dim=0)]
        y_i_list = [t for t in torch.split(y_i, afterpad_split_sizes, dim=0)]

        if args.model == "graphormer" and apply_graphormer_virtual_edges:
            edge_index_i_heads = _apply_to_edge_index(
                edge_index_i_heads,
                lambda e: adjust_edge_index_nomerge(e, max(split_sizes)),
            )

        x_i = x_i_list[seq_parallel_world_rank]
        y_i = y_i_list[seq_parallel_world_rank]
        sub_seq_start = seq_parallel_world_rank * max(split_sizes)
        sub_seq_end = sub_seq_start + x_i.size(0)

        if full_attn_bias is not None:
            # Pad [n_real, n_real, d] → [seq_length, seq_length, d] (zeros for pad nodes)
            # Use direct padding instead of pad_attn_bias() which requires SP group init
            n_real, _, _d = full_attn_bias.shape
            if n_real < seq_length:
                padded = full_attn_bias.new_zeros([seq_length, seq_length, _d])
                padded[:n_real, :n_real, :] = full_attn_bias
                attn_bias = padded
            else:
                attn_bias = full_attn_bias
            # Slice rows for this rank: [sub_s, full_padded_s, d]
            attn_bias = attn_bias[sub_seq_start:sub_seq_end, :, :]
            if dev is not None:
                attn_bias = attn_bias.to(dev)

        last_batch_flag(True)
    else:
        assert seq_length % seq_parallel_world_size == 0
        sub_seq_length = seq_length // seq_parallel_world_size
        sub_seq_start = seq_parallel_world_rank * sub_seq_length
        sub_seq_end = (seq_parallel_world_rank + 1) * sub_seq_length

        x_i = x_i[sub_seq_start:sub_seq_end, :]
        y_i = y_i[sub_seq_start:sub_seq_end]
        if args.model == "graphormer" and apply_graphormer_virtual_edges:
            edge_index_i_heads = _apply_to_edge_index(
                edge_index_i_heads,
                lambda e: adjust_edge_index_nomerge(e, sub_seq_length),
            )

        if full_attn_bias is not None:
            # Slice rows for this rank: [sub_s, full_s, d]
            attn_bias = full_attn_bias[sub_seq_start:sub_seq_end, :, :]
            if dev is not None:
                attn_bias = attn_bias.to(dev)

        last_batch_flag(False)

    return (x_i, y_i, edge_index_i_heads, attn_bias)


    


def extend_global_token0(x_layer: Tensor, extend_dim: int) -> Tensor:
    """Extend global token embedding to each rank

    Arguments:
        x_layer (Tensor): input tensor

    Returns:
        * output (Tensor): output tensor
    """
    # TODO consider multiple global tokens case
    # x_layer: [b, s+1, hp] 
    global_token_num = get_global_token_num()
    seq_world_size = get_sequence_parallel_world_size()
    
    assert (x_layer.size(extend_dim) - 1) % seq_world_size == 0
    split_sizes = [1] + [(x_layer.size(extend_dim) - 1) // seq_world_size] * seq_world_size

    # Split.
    tensor_list = torch.split(x_layer, split_sizes, dim=extend_dim)    
    output_list = [torch.cat([tensor_list[0], tensor_list[i]], dim=extend_dim).contiguous() for i in range(1, seq_world_size+1)]
    
    return torch.cat(output_list, dim=extend_dim).contiguous()


def copy_global_token0(x_layer: Tensor, extend_dim: int) -> Tensor:
    """copy global token embedding to each rank

    Arguments:
        x_layer (Tensor): input tensor

    Returns:
        * output (Tensor): output tensor
    """
    # x_layer: [b, s+p, hp] 
    # global_token_num = get_global_token_num()
    seq_world_size = get_sequence_parallel_world_size()

    if not get_last_batch_flag():
        global_token_indices = get_global_token_indices(last_batch=False)
    else:
        global_token_indices = get_global_token_indices(last_batch=True)

    assert x_layer.size(extend_dim) % seq_world_size == 0
    x_layer[:, torch.LongTensor(global_token_indices), :] = x_layer[:, 0, :].unsqueeze(1).repeat(1, len(global_token_indices), 1)
    
    return x_layer.contiguous()


    # sampler = SubsetRandomSampler(flatten_idx.tolist())
    # dataloader = DataLoader(
    #             dataset_train,
    #             batch_size=args.batch_size,
    #             num_workers=args.num_workers,
    #             pin_memory=True,
    #             sampler=sampler,
    #             collate_fn=partial(
    #                 collator,
    #                 max_node=get_dataset(args.dataset)["max_node"],
    #                 multi_hop_max_dist=args.multi_hop_max_dist,
    #                 spatial_pos_max=args.spatial_pos_max,
    #             )
    # )
    # # batch = next(iter(dataloader))
    # # print(f"rank {rank} {batch.y}")
    # # for batch in dataloader:
    # #     print(f"rank {rank} {batch.y}")
    # # exit(0)
    # return next(iter(dataloader))


