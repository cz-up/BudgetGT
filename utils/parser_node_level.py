import argparse


def add_node_common_args(parser, defaults=None):
    defaults = defaults or {}

    parser.add_argument("--dataset_dir", type=str, default=defaults.get("dataset_dir", "./dataset/"))
    parser.add_argument("--dataset", type=str, default=defaults.get("dataset", "ogbn-arxiv"))
    parser.add_argument(
        "--split_id",
        type=int,
        default=defaults.get("split_id", 0),
        help="split index for datasets providing multiple fixed/default splits",
    )

    parser.add_argument("--model", type=str, default=defaults.get("model", "graphormer"),
                        choices=["graphormer", "gt", "exphormer", "graphgps"])
    parser.add_argument(
        "--gt_lap_pe_dim",
        type=int,
        default=defaults.get("gt_lap_pe_dim", 0),
        help=(
            "[GT] Laplacian positional encoding dimension (0 = disabled, the historical "
            "behaviour). The original GT (Dwivedi & Bresson) adds a linear projection of "
            "the top-k Laplacian eigenvectors to the node embedding; concatenating the "
            "eigenvectors to the input features before the node encoder is mathematically "
            "equivalent. Eigenvectors are computed once on rank 0 and cached under the "
            "dataset directory. Per-epoch random sign flips (the paper's augmentation) "
            "are applied during training and removed for probe/eval."
        ),
    )
    parser.add_argument(
        "--expander_degree",
        type=int,
        default=defaults.get("expander_degree", 0),
        help=(
            "[Exphormer] Degree of the random d-regular expander graph added as extra "
            "attention edges (0 = disabled). Paper uses 3. Expander edges are generated "
            "once at startup (fixed seed) and merged into the attention edge pool each epoch. "
            "Edge type (0=real/RW, 1=expander) is carried in edge_index row-2 for the "
            "Exphormer model's edge-feature-modulated attention."
        ),
    )
    parser.add_argument("--n_layers", type=int, default=defaults.get("n_layers", 4))
    parser.add_argument(
        "--pre_mp_layers",
        type=int,
        default=defaults.get("pre_mp_layers", 0),
        help="GeneralLayer blocks (Linear(no bias)->BN->Dropout->ReLU) between the node "
             "encoder and the attention stack, matching the native backbones' "
             "gnn.layers_pre_mp.  Native Exphormer/GPS use 1; default 0 preserves the "
             "structure every existing run in logs/ was trained with.  "
             "Only --model exphormer and --model graphgps implement it.",
    )
    parser.add_argument("--num_heads", type=int, default=defaults.get("num_heads", 8))
    parser.add_argument("--hidden_dim", type=int, default=defaults.get("hidden_dim", 64))
    parser.add_argument("--ffn_dim", type=int, default=defaults.get("ffn_dim", 256))
    parser.add_argument("--attn_bias_dim", type=int, default=defaults.get("attn_bias_dim", 1))
    parser.add_argument("--dropout_rate", type=float, default=defaults.get("dropout_rate", 0.3))
    parser.add_argument("--input_dropout_rate", type=float, default=defaults.get("input_dropout_rate", 0.1))
    parser.add_argument(
        "--attention_dropout_rate",
        type=float,
        default=defaults.get("attention_dropout_rate"),
        help=(
            "Dropout on post-softmax attention weights. Default resolves per "
            "backbone: 0 for gt/exphormer (their original implementations apply "
            "no attention-prob dropout — Exphormer's official configs carry "
            "attn_dropout=0.5 but the Exphormer layer never consumes it), "
            "0.5 for graphormer/graphgps (original semantics consume it). "
            "An explicit value always wins."
        ),
    )
    parser.add_argument("--num_global_node", type=int, default=defaults.get("num_global_node", 1))
    parser.add_argument(
        "--attn_type",
        type=str,
        default=defaults.get("attn_type", "sparse"),
        help="attention type: sparse/full/flash/performer",
    )

    parser.add_argument("--weight_decay", type=float, default=defaults.get("weight_decay", 0.01))
    parser.add_argument(
        "--warmup_updates",
        type=int,
        default=defaults.get("warmup_updates", 10),
        help="warmup steps for optimizer learning rate scheduling",
    )
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 300))
    parser.add_argument("--eval_every", type=int, default=defaults.get("eval_every", 5))
    parser.add_argument("--peak_lr", type=float, default=defaults.get("peak_lr", 2e-4))
    parser.add_argument("--end_lr", type=float, default=defaults.get("end_lr", 1e-9))
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default=defaults.get("lr_schedule", "polynomial"),
        choices=["polynomial", "cosine_with_warmup"],
        help="LR scheduler: 'polynomial' (linear decay) or 'cosine_with_warmup' (matches Exphormer)",
    )
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--save_model", action="store_true", default=False, help="whether to save model")
    parser.add_argument("--model_dir", type=str, default=defaults.get("model_dir", "./model_ckpt/"))

    parser.add_argument(
        "--walk_length",
        type=int,
        default=defaults.get("walk_length", 4),
        help="head hop random-walk length",
    )
    parser.add_argument(
        "--walks_per_node",
        type=int,
        default=defaults.get("walks_per_node", 2),
        help="head hop random walks per node",
    )
    parser.add_argument(
        "--rw_edge_mode",
        type=str,
        default=defaults.get("rw_edge_mode", "random_walk"),
        choices=["random_walk", "dgl_neighbor"],
        help="how to build the head-hop edge pool: 'random_walk' (default) or "
             "'dgl_neighbor' (DGL fanout-based k-hop neighbour sampling). "
             "dgl_neighbor requires --walks_per_node > 0 to enable the rw branch; "
             "walk_length/walks_per_node are otherwise ignored in this mode.",
    )
    parser.add_argument(
        "--fanout",
        type=str,
        default=defaults.get("fanout", "10,5"),
        help="comma-separated per-hop fanout for --rw_edge_mode dgl_neighbor, "
             "e.g. '10,5' (2-hop). -1 at a hop means all neighbours (exact k-hop, "
             "may OOM on dense graphs). Ignored for random_walk mode.",
    )
    parser.add_argument(
        "--edge_build_device",
        type=str,
        default=defaults.get("edge_build_device", "gpu"),
        help="device for edge construction/sampling: gpu|cpu|cuda|cuda:N "
             "('same' is a deprecated alias for 'gpu', i.e. the training device)",
    )
    parser.add_argument("--rank", type=int, default=defaults.get("rank"))
    parser.add_argument("--local-rank", "--local_rank", type=int, default=defaults.get("local_rank"))
    parser.add_argument("--world-size", type=int, default=defaults.get("world_size"))
    parser.add_argument(
        "--distributed-backend",
        default=defaults.get("distributed_backend", "nccl"),
        choices=["nccl", "gloo", "ccl"],
        help="Which backend to use for distributed training.",
    )
    parser.add_argument(
        "--distributed-timeout-minutes",
        type=int,
        default=defaults.get("distributed_timeout_minutes", 10),
        help="Timeout minutes for torch.distributed.",
    )
    parser.add_argument(
        "--sequence-parallel-size",
        type=int,
        default=defaults.get("sequence_parallel_size", 4),
        help="Enable DeepSpeed's sequence parallel.",
    )


def add_node_batch_sp_args(parser, defaults=None):
    defaults = defaults or {}

    parser.add_argument(
        "--batch_subgraph_mode",
        type=str,
        default=defaults.get("batch_subgraph_mode", "induced"),
        choices=["induced", "seed_rw"],
        help="batch subgraph construction: induced subgraph or seed-based full-graph random walk",
    )
    parser.add_argument(
        "--seed_batch_size",
        type=int,
        default=defaults.get("seed_batch_size"),
        help="number of seed/query nodes per batch when --batch_subgraph_mode seed_rw; defaults to seq_len",
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=defaults.get("seq_len", 256000),
        help="total sequence length here",
    )

    parser.add_argument("--full_attn_hop_stats", action="store_true", default=False, help="enable full attention hop-mass stats in val")
    parser.add_argument("--full_attn_hop_mass", type=float, default=0.95, help="target hop mass ratio (e.g., 0.95)")
    parser.add_argument("--full_attn_hop_max_queries", type=int, default=defaults.get("full_attn_hop_max_queries", 64), help="max queries per batch for hop-mass stats")
    parser.add_argument(
        "--full_attn_hop_query_sampling",
        type=str,
        default=defaults.get("full_attn_hop_query_sampling", "random"),
        choices=["random", "prefix"],
        help="query selection for full-attention hop-mass stats: random samples without replacement or the old prefix-based order",
    )
    parser.add_argument("--full_attn_hop_max_batches", type=int, default=defaults.get("full_attn_hop_max_batches", 1), help="max batches per layer to collect hop-mass stats")
    parser.add_argument("--full_attn_hop_max_hop", type=int, default=defaults.get("full_attn_hop_max_hop", 64), help="max hop to explore when collecting hop-mass stats (<=0 for no limit)")
    parser.add_argument(
        "--full_attn_hop_stats_dir",
        type=str,
        default=defaults.get("full_attn_hop_stats_dir", "./plot/hop_stats"),
        help="output root directory for per-epoch full-attention hop-mass statistics",
    )
    parser.add_argument(
        "--full_attn_hop_stats_tag",
        type=str,
        default=defaults.get("full_attn_hop_stats_tag", ""),
        help="optional suffix appended to the hop-mass stats directory name",
    )
    parser.add_argument(
        "--attn_bias_mode",
        type=str,
        default=defaults.get("attn_bias_mode", "none"),
        choices=["none", "local_spd"],
        help="Attention bias encoding mode: none (disabled) | local_spd (per-batch BFS)",
    )
    parser.add_argument("--attn_bias_max_dist", type=int, default=defaults.get("attn_bias_max_dist", 5), help="[local_spd] max hop distance; distances > this are clamped. one-hot dim = max_dist+1")


def add_node_fullgraph_sp_args(parser, defaults=None):
    defaults = defaults or {}

    parser.add_argument(
        "--force_random_split",
        action="store_true",
        default=defaults.get("force_random_split", False),
        help="skip default split files and always use random 60/20/20 split (for fair comparison with baselines)",
    )
    parser.add_argument(
        "--node_sample_ratio",
        type=float,
        default=defaults.get("node_sample_ratio", 1.0),
        help=(
            "uniformly sample this fraction of graph vertices before full-graph SP training "
            "and keep the induced subgraph; 1.0 uses the full graph"
        ),
    )
    parser.add_argument(
        "--activation_checkpoint_mode",
        type=str,
        default=defaults.get("activation_checkpoint_mode"),
        choices=["all", "ffn_only", "multi_tier"],
        help=(
            "activation checkpoint mode: "
            "'all' checkpoints the full EncoderLayer; "
            "'ffn_only' checkpoints only the FFN block, keeping all MHA activations to "
            "eliminate A2A recomputation from backward (faster backward, higher peak memory); "
            "'multi_tier' uses a four-tier scheduler (recompute / offload to pinned CPU / "
            "ffn-only checkpoint / retain on GPU) with fail-fast-recover fallback: starts "
            "uniformly at recompute while edge budget is still exploring, and switches to "
            "CPU offload on GPU OOM."
        ),
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default=defaults.get("amp_dtype", "none"),
        choices=["none", "bf16", "fp16"],
        help="mixed precision dtype for model forward: none|bf16|fp16",
    )
    parser.add_argument(
        "--profile_sp_comm",
        action="store_true",
        default=defaults.get("profile_sp_comm", False),
        help=(
            "profile SeqAllToAll and full-graph edge broadcast communication time; "
            "also enable CUDA-synchronized rw/fwd/bwd/opt step timers for accurate wall time"
        ),
    )
    parser.add_argument(
        "--multi_tier_gpu_memory_limit_mib",
        type=int,
        default=defaults.get("multi_tier_gpu_memory_limit_mib", 0),
        help=(
            "per-rank GPU memory cap in MiB for --activation_checkpoint_mode multi_tier; "
            "<=0 uses the physical GPU size. The cap is used by the multi_tier planner "
            "and applied to PyTorch's CUDA allocator when running on CUDA."
        ),
    )
    parser.add_argument(
        "--force_multi_tier_plan",
        type=str,
        default=defaults.get("force_multi_tier_plan", ""),
        help=(
            "ablation override: bypass the multi_tier planner and force a specific "
            "edge_policy + tier configuration. Format: '<edge_policy>:<tier_config>'. "
            "edge_policy ∈ {gpu_persist, gpu_ephemeral, cpu_rank_local_prefetch, "
            "cpu_broadcast_prefetch}; tier_config ∈ {recompute, keep_mha=N, retain=N} "
            "where N counts layers from the back. "
            "Example: 'cpu_rank_local_prefetch:keep_mha=2' or 'gpu_persist:recompute'. "
            "Empty (default) means let the planner choose."
        ),
    )
    parser.add_argument(
        "--disable_edge_prefetch",
        action="store_true",
        default=defaults.get("disable_edge_prefetch", False),
        help="ablation: never overlap edge construction with GPU compute; build the next epoch's edges in the foreground (serial). Isolates the value of overlap-aware prefetching against an otherwise identical build path.",
    )
    parser.add_argument(
        "--max_total_edges_per_query",
        type=int,
        default=defaults.get("max_total_edges_per_query", 0),
        help="total per-query edge budget for random real/RW block sampling, used with --adaptive_edge_budget or the --fixed_*_edges_per_query overrides (<=0 disables). Automatically split if adaptive budget is disabled.",
    )
    parser.add_argument(
        "--adaptive_edge_budget",
        action="store_true",
        default=defaults.get("adaptive_edge_budget", False),
        help="enable probe-based greedy allocation between real-edge and RW edge budgets in full-graph sparse attention",
    )
    parser.add_argument(
        "--adaptive_edge_budget_block_size",
        type=int,
        default=defaults.get("adaptive_edge_budget_block_size", 0),
        help="advanced override; <=0 uses the default per-query edge block size",
    )
    parser.add_argument(
        "--adaptive_edge_budget_min_gain",
        type=float,
        default=defaults.get("adaptive_edge_budget_min_gain", 0.0),
        help="advanced override; minimum relative probe-loss gain treated as a significant signal (<=0 uses the default 3e-5); gains below it neither steer growth direction nor commit a budget move",
    )
    parser.add_argument(
        "--adaptive_edge_budget_patience",
        type=int,
        default=defaults.get("adaptive_edge_budget_patience", 0),
        help="advanced override (sensitivity study); rounds without a winner before the split freezes (P), <=0 uses the default 3",
    )
    parser.add_argument(
        "--adaptive_edge_budget_max_confirm",
        type=int,
        default=defaults.get("adaptive_edge_budget_max_confirm", 0),
        help="advanced override (sensitivity study); cap on the consecutive wins a commit needs (K_max), <=0 uses the default 3",
    )
    parser.add_argument(
        "--include_real_edges",
        type=int,
        default=defaults.get("include_real_edges", 0),
        help="legacy full-graph flag for including all real edges (used by the all-real baseline); with an adaptive or fixed edge budget and --max_total_edges_per_query > 0, real-edge blocks are enabled automatically",
    )
    parser.add_argument(
        "--to_bidirected",
        action="store_true",
        default=defaults.get("to_bidirected", False),
    )
    parser.add_argument(
        "--force_edge_broadcast",
        action="store_true",
        default=defaults.get("force_edge_broadcast", False),
        help="force the legacy full-graph path where the SP source rank builds sampled edges and broadcasts them to the other SP ranks even when deterministic seeded local construction is available",
    )
    parser.add_argument(
        "--fixed_real_edges_per_query",
        type=int,
        default=defaults.get("fixed_real_edges_per_query"),
        help="fixed full-graph sampling override; set together with --fixed_rw_edges_per_query to train with a manually specified adaptive-style edge budget",
    )
    parser.add_argument(
        "--fixed_rw_edges_per_query",
        type=int,
        default=defaults.get("fixed_rw_edges_per_query"),
        help="fixed full-graph sampling override; set together with --fixed_real_edges_per_query to train with a manually specified adaptive-style edge budget",
    )
    parser.add_argument(
        "--fixed_walk_length",
        type=int,
        default=defaults.get("fixed_walk_length"),
        help="optional fixed random-walk length paired with --fixed_real_edges_per_query/--fixed_rw_edges_per_query; defaults to --walk_length when omitted",
    )

def _normalize_attention_dropout_args(args):
    """Resolve the per-backbone attention_dropout_rate default (None sentinel).

    gt/exphormer: 0 — the original GT (Dwivedi & Bresson) and Exphormer sparse
    attention implementations apply no dropout to attention weights (the
    attn_dropout field in Exphormer's official configs is never consumed by
    the Exphormer layer), and TorchGT's sparse path likewise has none.
    graphormer/graphgps: 0.5 — their original implementations consume it.
    An explicit --attention_dropout_rate always takes precedence.
    """
    if getattr(args, "attention_dropout_rate", None) is None:
        model = str(getattr(args, "model", "")).lower()
        args.attention_dropout_rate = 0.0 if model in ("gt", "exphormer") else 0.5


def normalize_main_node_sp_args(args):
    _normalize_checkpoint_args(args)
    _normalize_attention_dropout_args(args)
    if str(getattr(args, "model", "")).lower() == "gt":
        args.attn_type = "sparse"
    return args


def normalize_main_node_fullgraph_sp_args(args):
    _normalize_checkpoint_args(args)
    _normalize_attention_dropout_args(args)
    _normalize_fixed_edge_budget_args(args)
    _normalize_multi_tier_gpu_memory_limit_args(args)
    node_sample_ratio = float(getattr(args, "node_sample_ratio", 1.0))
    if not (0.0 < node_sample_ratio <= 1.0):
        raise ValueError("--node_sample_ratio must be in (0, 1].")
    args.node_sample_ratio = node_sample_ratio
    if str(getattr(args, "model", "")).lower() in ("gt", "exphormer"):
        args.attn_type = "sparse"
    # graphgps supports both sparse (default) and full attention; preserve user's --attn_type
    if str(getattr(args, "model", "")).lower() == "graphgps":
        if not getattr(args, "attn_type", None):
            args.attn_type = "sparse"
    if args.model != "graphormer":
        args.num_global_node = 0
    elif args.num_global_node not in (0, 1):
        raise ValueError("main_node_fullgraph_sp.py currently supports at most one Graphormer-style virtual node.")
    if str(getattr(args, "dataset", "")).lower() == "ogbn-arxiv":
        args.to_bidirected = True
    return args


def _normalize_checkpoint_args(args):
    checkpoint_mode = getattr(args, "activation_checkpoint_mode", None)
    if checkpoint_mode is None:
        return
    checkpoint_mode = str(checkpoint_mode).lower()
    if checkpoint_mode not in {"all", "ffn_only", "multi_tier"}:
        raise ValueError(f"Unsupported activation_checkpoint_mode: {checkpoint_mode}")
    args.activation_checkpoint_mode = checkpoint_mode


def _normalize_multi_tier_gpu_memory_limit_args(args):
    limit_mib = int(getattr(args, "multi_tier_gpu_memory_limit_mib", 0) or 0)
    if limit_mib < 0:
        limit_mib = 0
    args.multi_tier_gpu_memory_limit_mib = limit_mib


def _normalize_fixed_edge_budget_args(args):
    fixed_real = getattr(args, "fixed_real_edges_per_query", None)
    fixed_rw = getattr(args, "fixed_rw_edges_per_query", None)
    fixed_walk = getattr(args, "fixed_walk_length", None)

    if fixed_real is None and fixed_rw is None:
        if fixed_walk is not None:
            raise ValueError("--fixed_walk_length requires --fixed_real_edges_per_query and --fixed_rw_edges_per_query.")
        return
    if fixed_real is None or fixed_rw is None:
        raise ValueError("--fixed_real_edges_per_query and --fixed_rw_edges_per_query must be set together.")
    if int(fixed_real) < 0 or int(fixed_rw) < 0:
        raise ValueError("Fixed edge budgets must be >= 0.")
    args.fixed_real_edges_per_query = int(fixed_real)
    args.fixed_rw_edges_per_query = int(fixed_rw)
    if args.fixed_rw_edges_per_query > 0 and int(getattr(args, "walks_per_node", 0)) <= 0:
        raise ValueError("--fixed_rw_edges_per_query > 0 requires --walks_per_node > 0.")
    if fixed_walk is not None:
        if int(fixed_walk) <= 0:
            raise ValueError("--fixed_walk_length must be > 0 when provided.")
        args.fixed_walk_length = int(fixed_walk)
