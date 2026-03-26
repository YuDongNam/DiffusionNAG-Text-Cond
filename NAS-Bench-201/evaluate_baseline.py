"""
evaluate_baseline.py
====================
Unified Evaluation Script for baseline_diffusion_nag.py

Evaluates the SDEdit mutation results using either a local 'dry_run' mode
or a 'full' production mode with real NASBench-201 datasets and strict weight loading.
"""

import sys
import os
import json
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Internal project imports
from baseline_diffusion_nag import (
    TextConditionedCATE, get_default_config, load_cate_checkpoint,
    VESDE, sdedit_mutation, data_scaler, data_inverse_scaler,
    collate_fn, NADTripletDataset, DummyTripletDataset, build_vocab_from_jsonl,
    build_attention_mask_from_adj, parse_nad_graph_string,
    padding_mask_from_onehot, apply_padding_to_attention_mask,
    load_or_create_dataset_splits, IndexView
)
import random
import numpy as np

# ============================================================================
# Reproducibility
# ============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# Core Functions
# ============================================================================
def discretize_generated_x(x_continuous):
    B, N, V = x_continuous.shape
    x_discrete = torch.zeros_like(x_continuous)
    indices = torch.argmax(x_continuous, dim=-1)
    
    # Removed hardcoded Node 0 and Node 7 limits to dynamically support variable-length NAD DAGs.
    x_discrete.scatter_(-1, indices.unsqueeze(-1), 1.0)
    return x_discrete


def is_valid_NAD_x(x_discrete, start_idx, end_idx):
    """
    Dynamically checks validity for variable-length NAD DAGs.
    x_discrete is a one-hot [N, V] numpy array.
    """
    if len(x_discrete.shape) == 2:
        types = np.argmax(x_discrete, axis=-1)
    else:
        types = x_discrete
        
    # 1. Start node must be at index 0
    if types[0] != start_idx:
        return False, "NO_START_NODE"
    
    # 2. Find the end node dynamically
    end_indices = np.where(types == end_idx)[0]
    if len(end_indices) != 1:  
        return False, "MULTIPLE_OR_NO_END_NODES"
    
    end_pos = end_indices[0]
    if end_pos == 0:
        return False, "END_NODE_AT_ZERO"
        
    # 3. Intermediate nodes cannot be start or end
    interm_types = types[1:end_pos]
    if start_idx in interm_types or end_idx in interm_types:
        return False, "INTERM_START_END"
        
    return True, "NO_ERROR"


def dynamic_decode_NAD_string(x_discrete, ops_decoder, start_idx, end_idx):
    """
    Decodes the graph dynamically up to the actual end node.
    Returns a pipe-separated string representing the variable-length topology.
    """
    is_valid, _ = is_valid_NAD_x(x_discrete, start_idx=start_idx, end_idx=end_idx)
    if not is_valid:
        return None
        
    types = np.argmax(x_discrete, axis=-1)
    end_pos = np.where(types == end_idx)[0][0]
    arch_str = "|".join([ops_decoder[t] for t in types[:end_pos+1]])
    return arch_str


def evaluate_metrics(generated_x, ops_decoder, training_graphs_list, start_idx, end_idx):
    B = generated_x.shape[0]
    valid_arch_strings = []
    
    # 1. Validity Check (Dynamic NAD Bypassing)
    for i in range(B):
        x = generated_x[i].cpu().numpy()
        is_valid, _ = is_valid_NAD_x(x, start_idx=start_idx, end_idx=end_idx)
        if is_valid:
            arch_str = dynamic_decode_NAD_string(x, ops_decoder, start_idx=start_idx, end_idx=end_idx)
            if arch_str is not None:
                valid_arch_strings.append(arch_str)
                
    valid_count = len(valid_arch_strings)
    validity = valid_count / B if B > 0 else 0.0
    
    # 2. Uniqueness Check
    unique_arch_strings = list(set(valid_arch_strings))
    unique_count = len(unique_arch_strings)
    uniqueness = unique_count / valid_count if valid_count > 0 else 0.0
    
    # 3. Novelty Check
    train_set_hashed = set(training_graphs_list)
    novel_archs = [arch for arch in unique_arch_strings if arch not in train_set_hashed]
    novel_count = len(novel_archs)
    novelty = novel_count / unique_count if unique_count > 0 else 0.0
    
    return validity, uniqueness, novelty, valid_count, unique_count, novel_count, unique_arch_strings


# ============================================================================
# Novelty baseline from JSONL train split
# ============================================================================
def _ops_to_seq_string(ops, end_op="output"):
    """Convert parsed op list into the same string form used by evaluation.

    We stop at the first occurrence of `end_op` (inclusive). If it doesn't exist,
    we return the full op list joined.
    """
    try:
        end_pos = ops.index(end_op)
        ops = ops[: end_pos + 1]
    except ValueError:
        pass
    return "|".join(ops)


def build_seen_graphs_from_train_dataset(
    train_dataset,
    *,
    vocab,
    input_idx,
    output_idx,
    pad_idx,
):
    """Build seen_graphs_set from the ACTUAL tensors the model saw in training.

    Requirement compliance:
    - Constructed from X_parent and X_child returned by the train dataset.
    - Uses the same decode funnel: find first output token index dynamically.
    """
    seen = set()
    for i in range(len(train_dataset)):
        X_parent, _, X_child, _, _ = train_dataset[i]
        for X in (X_parent, X_child):
            types = torch.argmax(X, dim=-1).tolist()  # [N]
            # enforce start token presence at 0 if possible; still decode by EOS funnel
            try:
                end_pos = types.index(output_idx)
                types = types[: end_pos + 1]
            except ValueError:
                # no explicit output -> trim trailing PAD
                while types and types[-1] == pad_idx:
                    types.pop()
            seq = "|".join([vocab[t] for t in types])
            seen.add(seq)
    return seen


# ============================================================================
# Main Execution Loop
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate Baseline Diffusion NAG")
    parser.add_argument('--mode', type=str, choices=['dry_run', 'jsonl', 'full'], default='jsonl',
                        help="dry_run: synthetic tiny dataset; jsonl: run on whole set; full: locked split file")
    parser.add_argument('--jsonl_path', type=str, default='NAD_triplet_dataset.jsonl',
                        help="Path to NAD triplet JSONL dataset.")
    parser.add_argument('--max_node', type=int, default=110,
                        help="Max number of nodes (pad/truncate not allowed; raises if exceeded).")
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'], default='auto',
                        help="Device to run inference on.")
    parser.add_argument('--batch_size', type=int, default=None,
                        help="Batch size. Defaults to 64.")
    parser.add_argument('--t_edit', type=float, default=0.5,
                        help="SDEdit edit ratio. The fraction of total timesteps to use for noise corruption (default: 0.5).")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility. (default: 42)")
    parser.add_argument('--vocab_scan_limit', type=int, default=None,
                        help="Optional limit on number of JSONL lines to scan when building vocab.")
    parser.add_argument('--dataset_limit', type=int, default=None,
                        help="Optional limit on number of JSONL lines to load for evaluation.")
    parser.add_argument('--split_path', type=str, default="results/dataset_splits.json",
                        help="Unified split file path used in full mode.")
    parser.add_argument('--split_seed', type=int, default=42,
                        help="Seed used only if split file does not exist yet.")
    parser.add_argument('--split_train_ratio', type=float, default=0.8,
                        help="Train ratio used only if split file does not exist yet.")
    parser.add_argument('--split_val_ratio', type=float, default=0.1,
                        help="Val ratio used only if split file does not exist yet.")
    parser.add_argument('--eval_split', type=str, choices=["test", "val"], default="test",
                        help="Which split to evaluate on in full mode.")
    args = parser.parse_args()

    # Apply Random Seed
    set_seed(args.seed)

    print("=" * 70)
    print(f"  EVALUATION BASELINE — MODE: {args.mode.upper()}")
    print("=" * 70)

    # ----------------------------------------------------------------
    # 1. Setup & Device
    # ----------------------------------------------------------------
    if args.device == 'auto':
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = torch.device(args.device)
    print(f"[INFO] Using Device: {DEVICE}")

    BATCH_SIZE = args.batch_size if args.batch_size is not None else 64
    print(f"[INFO] Batch Size: {BATCH_SIZE}")

    D_TEXT = 512
    D_HIDDEN = 256
    
    if args.mode in ("jsonl", "full"):
        # Build op vocab from JSONL, then build a config matching that vocab size.
        vocab, op_to_idx = build_vocab_from_jsonl(args.jsonl_path, max_items=args.vocab_scan_limit)
        print(f"[INFO] Built node-op vocab: V={len(vocab)} (includes PAD)")
        if "input" not in op_to_idx or "output" not in op_to_idx:
            raise ValueError("Vocab must contain both 'input' and 'output' ops for validity checks.")
    else:
        vocab = ["PAD", "input", "output"]
        op_to_idx = {op: i for i, op in enumerate(vocab)}
        print(f"[INFO] Using dry_run vocab: V={len(vocab)}")

    START_IDX = op_to_idx["input"]
    END_IDX = op_to_idx["output"]

    config = get_default_config(max_node=args.max_node, n_vocab=len(vocab), data_name='NAD_JSONL' if args.mode == "jsonl" else "DRY_RUN")
    config.device = DEVICE
    N_NODES = config.data.max_node
    N_VOCAB = config.data.n_vocab

    # ----------------------------------------------------------------
    # 2. Strict Model & Checkpoint Loading
    # ----------------------------------------------------------------
    print("[INFO] Constructing Network...")
    model = TextConditionedCATE(config, d_text=D_TEXT, d_hidden=D_HIDDEN)
    
    print("[INFO] Loading frozen CATE base weights...")
    model = load_cate_checkpoint(model, ckpt_path='checkpoints/checkpoint.pth.tar', device='cpu')
    
    text_mlp_path = 'checkpoints/baseline_text_mlp.pth'
    if os.path.exists(text_mlp_path):
        model.load_state_dict(torch.load(text_mlp_path, map_location='cpu'), strict=False)
        print(f"[INFO] Loaded Text_MLP weights from {text_mlp_path}")
    else:
        print(f"[WARNING] Text_MLP weights not found at {text_mlp_path}. Using initialization defaults.")
            
    model.to(DEVICE)
    model.eval()

    # ----------------------------------------------------------------
    # 3. Dataset + unified split enforcement (full mode)
    # ----------------------------------------------------------------
    loader_kwargs = {'num_workers': 0, 'pin_memory': True if DEVICE.type == 'cuda' else False}
    if args.mode in ("jsonl", "full"):
        ds_full = NADTripletDataset(
            jsonl_path=args.jsonl_path,
            op_to_idx=op_to_idx,
            max_node=N_NODES,
            d_text=D_TEXT,
            limit=args.dataset_limit,
        )
    else:
        # Always keep it tiny for smoke tests.
        ds_full = DummyTripletDataset(
            size=4,
            op_to_idx=op_to_idx,
            max_node=N_NODES,
            d_text=D_TEXT,
            ds_type="dry_run",
        )

    if args.mode == "full":
        splits = load_or_create_dataset_splits(
            split_path=args.split_path,
            n_items=len(ds_full),
            jsonl_path=args.jsonl_path,
            seed=args.split_seed,
            train_ratio=args.split_train_ratio,
            val_ratio=args.split_val_ratio,
        )
        train_view = IndexView(ds_full, splits["train"])
        eval_idxs = splits[args.eval_split]
        test_dataset = IndexView(ds_full, eval_idxs)

        seen_graphs_set = build_seen_graphs_from_train_dataset(
            train_view,
            vocab=vocab,
            input_idx=START_IDX,
            output_idx=END_IDX,
            pad_idx=op_to_idx["PAD"],
        )
        training_graphs_list = list(seen_graphs_set)
        print(f"[INFO] Using unified split file: {args.split_path}")
        print(f"[INFO] Seen set built from TRAIN split tensors: {len(training_graphs_list)} sequences")
        print(f"[INFO] Evaluating on {args.eval_split} split: {len(test_dataset)} items")
    else:
        test_dataset = ds_full
        training_graphs_list = []

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, **loader_kwargs)

    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)

    # ----------------------------------------------------------------
    # 4. Shared Core Evaluation Loop
    # ----------------------------------------------------------------
    print(f"[INFO] Beginning SDEdit Generation (Total Batches: {len(test_loader)})...")
    all_X_child_pred_discrete = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Generation Progress")):
            if len(batch_data) != 5:
                raise RuntimeError("Expected JSONL dataset batch with (X_parent, text_emb, X_child, ds_types, adj)")
            X_parent, text_emb_batch, _, _, adj_batch = batch_data
                
            # Scale X_parent to [-1, 1] for SDE
            X_parent_scaled = data_scaler(X_parent).to(DEVICE)
            text_emb_batch = text_emb_batch.to(DEVICE)
            
            # Build per-sample attention masks from adjacency (transitive closure + diagonal ones)
            valid_nodes = padding_mask_from_onehot(X_parent, pad_idx=op_to_idx["PAD"])  # [B,N] bool
            masks = []
            for a, vn in zip(adj_batch, valid_nodes):
                m = build_attention_mask_from_adj(a, device=DEVICE)
                m = apply_padding_to_attention_mask(m, vn.to(DEVICE))
                masks.append(m)
            maskX = torch.stack(masks).to(DEVICE)

            # SDEdit Iteration using user-defined t_edit ratio
            X_child_pred_scaled = sdedit_mutation(
                model, sde, X_parent_scaled, text_emb_batch, maskX, edit_ratio=args.t_edit
            )
            
            # Denormalize and Discretize
            X_child_pred = data_inverse_scaler(X_child_pred_scaled)
            X_child_pred_discrete = discretize_generated_x(X_child_pred)
            
            # Store to CPU memory immediately to avoid GPU OOM on large datasets
            all_X_child_pred_discrete.append(X_child_pred_discrete.detach().cpu())

    # Aggregate Full Dataset
    print("[INFO] Generation complete. Aggregating continuous outputs...")
    generated_x_all = torch.cat(all_X_child_pred_discrete, dim=0)

    # ----------------------------------------------------------------
    # 5. Shared Metrics Reporting
    # ----------------------------------------------------------------
    print("[INFO] Executing Structural Metric Calculations...")
    validity, uniqueness, novelty, v_cnt, u_cnt, n_cnt, unique_arch_strings = evaluate_metrics(
        generated_x_all, 
        ops_decoder=vocab,
        training_graphs_list=training_graphs_list,
        start_idx=START_IDX,
        end_idx=END_IDX,
    )

    print("\n" + "=" * 50)
    print(f"               METRICS REPORT")
    print("=" * 50)
    print(f" Total Generated: {generated_x_all.shape[0]}")
    print(f" Valid          : {v_cnt} ({(validity * 100):.2f}%)")
    print(f" Unique (vs Val): {u_cnt} ({(uniqueness * 100):.2f}%)")
    print(f" Novel (vs Uniq): {n_cnt} ({(novelty * 100):.2f}%)")
    print("=" * 50)

    # ----------------------------------------------------------------
    # 6. Optional Persistence
    # ----------------------------------------------------------------
    os.makedirs('results', exist_ok=True)
    report = {
        'metrics': {
            'total_generated': int(generated_x_all.shape[0]),
            'validity_percent': round(validity * 100, 2),
            'uniqueness_percent': round(uniqueness * 100, 2),
            'novelty_percent': round(novelty * 100, 2),
            'raw_counts': {
                'valid': v_cnt,
                'unique': u_cnt,
                'novel': n_cnt
            }
        },
        'unique_valid_architectures_generated': unique_arch_strings
    }
    
    json_path = "results/baseline_evaluation_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"[SUCCESS] Metrics and outputs strictly persisted to `{json_path}`.")

if __name__ == "__main__":
    main()
