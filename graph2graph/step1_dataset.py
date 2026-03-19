"""
Step 1: Data Parsing & Dataloader for Conditional Graph-to-Graph Translation
=============================================================================
Parses NAD_triplet_dataset.jsonl into multi-discrete node representations
with adjacency matrices, padding masks, and pre-extracted text embeddings.

Tensor shapes (in comments throughout):
  B = batch size
  N = N_max = 110  (padded graph size — true max across dataset is 106)
  K = number of attribute keys (e.g., 8)
"""

import json
import re
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# 1. VOCABULARY BUILDER
# ============================================================================

class GraphVocabulary:
    """
    Builds and holds vocabularies for node types and each attribute key.
    
    Special tokens:
      - EMPTY (index 0): padding token for nodes beyond the real graph
      - NONE  (index 1): "no attribute" — used when a node doesn't have a
                         particular attribute key (e.g., ReLU has no kernel_size)
    """

    SPECIAL_TOKENS = ["<EMPTY>", "<NONE>"]

    # Fixed ordering of attribute keys for consistent multi-discrete encoding
    ATTR_KEYS = [
        "out_channels",
        "kernel_size",
        "stride",
        "groups",
        "dilation",
        "dim",
        "output_size",
        "pretrained_init",
    ]

    def __init__(self):
        # node_type vocab: index -> string
        self.node_type_to_idx: Dict[str, int] = {}
        self.idx_to_node_type: Dict[int, str] = {}

        # Per-attribute vocab: attr_key -> {value_string -> index}
        self.attr_to_idx: Dict[str, Dict[str, int]] = {}
        self.idx_to_attr: Dict[str, Dict[int, str]] = {}

        # Initialize special tokens
        for i, tok in enumerate(self.SPECIAL_TOKENS):
            self.node_type_to_idx[tok] = i
            self.idx_to_node_type[i] = tok

        for key in self.ATTR_KEYS:
            self.attr_to_idx[key] = {}
            self.idx_to_attr[key] = {}
            for i, tok in enumerate(self.SPECIAL_TOKENS):
                self.attr_to_idx[key][tok] = i
                self.idx_to_attr[key][i] = tok

    def _add_node_type(self, name: str) -> int:
        if name not in self.node_type_to_idx:
            idx = len(self.node_type_to_idx)
            self.node_type_to_idx[name] = idx
            self.idx_to_node_type[idx] = name
        return self.node_type_to_idx[name]

    def _add_attr_value(self, key: str, value: str) -> int:
        if key not in self.attr_to_idx:
            # Unknown attribute key — skip
            return -1
        if value not in self.attr_to_idx[key]:
            idx = len(self.attr_to_idx[key])
            self.attr_to_idx[key][value] = idx
            self.idx_to_attr[key][idx] = value
        return self.attr_to_idx[key][value]

    @property
    def num_node_types(self) -> int:
        return len(self.node_type_to_idx)

    def num_attr_classes(self, key: str) -> int:
        return len(self.attr_to_idx[key])

    @property
    def attr_dims(self) -> List[int]:
        """Returns list of vocabulary sizes per attribute key (in ATTR_KEYS order)."""
        return [self.num_attr_classes(k) for k in self.ATTR_KEYS]

    def build_from_data(self, data: List[dict]):
        """
        First pass: scan all graphs and register every node type / attr value.
        Must be called BEFORE encoding any graphs.
        """
        for sample in data:
            for graph_key in ["parent_graph", "child_graph"]:
                graph_str = sample[graph_key]
                nodes, _ = _parse_graph_string(graph_str)
                for _, (base_type, attrs) in nodes.items():
                    self._add_node_type(base_type)
                    for key in self.ATTR_KEYS:
                        if key in attrs:
                            self._add_attr_value(key, attrs[key])

    def encode_node_type(self, name: str) -> int:
        return self.node_type_to_idx.get(name, self.node_type_to_idx["<EMPTY>"])

    def encode_attr(self, key: str, value: Optional[str]) -> int:
        if value is None:
            return self.attr_to_idx[key]["<NONE>"]
        return self.attr_to_idx[key].get(value, self.attr_to_idx[key]["<NONE>"])

    def summary(self) -> str:
        lines = [f"Node types: {self.num_node_types}"]
        for key in self.ATTR_KEYS:
            lines.append(f"  {key}: {self.num_attr_classes(key)} classes")
        return "\n".join(lines)


# ============================================================================
# 2. GRAPH STRING PARSER
# ============================================================================

def _parse_params(param_str: str) -> Dict[str, str]:
    """
    Parse a parameter string like 'out_channels=C,kernel_size=3,stride=1'
    handling nested parentheses like 'kernel_size=(1,3)' or 'dim=(2,3)'.
    
    Returns: dict of {key: value_string}
    """
    params = {}
    depth = 0
    current = ""
    for ch in param_str:
        if ch == "(":
            depth += 1
            current += ch
        elif ch == ")":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            if "=" in current:
                k, v = current.strip().split("=", 1)
                params[k.strip()] = v.strip()
            current = ""
        else:
            current += ch
    if current.strip() and "=" in current:
        k, v = current.strip().split("=", 1)
        params[k.strip()] = v.strip()
    return params


def _parse_graph_string(graph_str: str) -> Tuple[Dict[int, Tuple[str, Dict]], List[Tuple[int, int]]]:
    """
    Parse a graph string from the dataset.
    
    Format:
        ##BlockName##
        0:input
        1:output
        2:Conv2d(out_channels=C,kernel_size=3,stride=1)
        ...
        0->2
        2->3
    
    Returns:
        nodes: dict {node_idx: (base_type, {attr_key: attr_value})}
        edges: list of (src, dst) tuples
    """
    nodes = {}
    edges = []
    for line in graph_str.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("##"):
            continue
        if "->" in line:
            src, dst = line.split("->")
            edges.append((int(src.strip()), int(dst.strip())))
        elif ":" in line:
            idx_str, op = line.split(":", 1)
            idx = int(idx_str.strip())
            if "(" in op:
                base = op[: op.index("(")]
                param_str = op[op.index("(") + 1 : op.rindex(")")]
                attrs = _parse_params(param_str)
            else:
                base = op.strip()
                attrs = {}
            nodes[idx] = (base, attrs)
    return nodes, edges


# ============================================================================
# 3. GRAPH ENCODER  (string → tensors)
# ============================================================================

N_MAX = 110  # Maximum number of nodes; all graphs padded to this (max in dataset = 106)


def encode_graph(
    graph_str: str,
    vocab: GraphVocabulary,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode a single graph string into padded tensors.
    
    Returns:
        node_types : LongTensor [N_MAX]           — categorical node type indices  (N_MAX=110)
        node_attrs : LongTensor [N_MAX x K]       — per-attribute categorical indices
                                                     K = len(ATTR_KEYS) = 8
        adj        : FloatTensor [N_MAX x N_MAX]  — binary adjacency matrix  (110 x 110)
        mask       : FloatTensor [N_MAX]           — 1.0 for real nodes, 0.0 for padding
    """
    nodes, edges = _parse_graph_string(graph_str)
    K = len(vocab.ATTR_KEYS)

    # Sort node indices to get a consistent ordering
    sorted_indices = sorted(nodes.keys())

    # Assert no truncation — N_MAX must be >= max graph size in dataset
    assert len(sorted_indices) <= N_MAX, (
        f"Graph has {len(sorted_indices)} nodes but N_MAX={N_MAX}. "
        f"Increase N_MAX to avoid truncation."
    )

    num_real = len(sorted_indices)

    # Build re-mapping: original_idx -> new_idx (0-based, up to N_MAX-1)
    remap = {orig: new for new, orig in enumerate(sorted_indices)}

    # --- Node types [N_MAX] ---
    node_types = torch.zeros(N_MAX, dtype=torch.long)   # 0 = <EMPTY>
    # --- Node attributes [N_MAX x K] ---
    node_attrs = torch.zeros(N_MAX, K, dtype=torch.long)  # 0 = <EMPTY>
    # For padding nodes, both type and attrs stay 0 (<EMPTY>)

    for new_idx, orig_idx in enumerate(sorted_indices):
        base_type, attrs = nodes[orig_idx]
        node_types[new_idx] = vocab.encode_node_type(base_type)
        for k_idx, key in enumerate(vocab.ATTR_KEYS):
            val = attrs.get(key, None)
            node_attrs[new_idx, k_idx] = vocab.encode_attr(key, val)

    # --- Adjacency matrix [N_MAX x N_MAX] ---
    adj = torch.zeros(N_MAX, N_MAX, dtype=torch.float32)
    for src, dst in edges:
        if src in remap and dst in remap:
            adj[remap[src], remap[dst]] = 1.0

    # --- Padding mask [N_MAX] ---
    mask = torch.zeros(N_MAX, dtype=torch.float32)
    mask[:num_real] = 1.0

    return node_types, node_attrs, adj, mask


# ============================================================================
# 4. DUMMY TEXT EMBEDDING   (placeholder for Baidu embeddings)
# ============================================================================

TEXT_EMBED_DIM = 768  # Baidu embedding dimension


def get_dummy_text_embedding(text: str, dim: int = TEXT_EMBED_DIM) -> torch.Tensor:
    """
    Placeholder: generates a deterministic pseudo-random embedding from text.
    In production, replace this with real Baidu API calls or pre-extracted .npy.
    
    Returns: FloatTensor [dim]     (e.g., [768])
    """
    # Use hash of text as seed for reproducibility
    seed = hash(text) % (2**32)
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    # L2 normalize (Baidu embeddings are often normalized)
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return torch.from_numpy(vec)


# ============================================================================
# 5. PYTORCH DATASET
# ============================================================================

class NADTripletDataset(Dataset):
    """
    PyTorch Dataset for NAD triplet data.
    
    Each sample returns:
        parent_node_types  : LongTensor  [N_MAX]           — parent node type IDs
        parent_node_attrs  : LongTensor  [N_MAX x K]       — parent attribute IDs
        parent_adj         : FloatTensor [N_MAX x N_MAX]    — parent adjacency
        parent_mask        : FloatTensor [N_MAX]            — parent padding mask
        child_node_types   : LongTensor  [N_MAX]            — child node type IDs
        child_node_attrs   : LongTensor  [N_MAX x K]        — child attribute IDs
        child_adj          : FloatTensor [N_MAX x N_MAX]     — child adjacency
        child_mask         : FloatTensor [N_MAX]             — child padding mask
        text_embedding     : FloatTensor [TEXT_EMBED_DIM]    — text embedding (768)
    
    Where:
        N_MAX = 110
        K     = 8  (number of attribute keys)
    """

    def __init__(
        self,
        jsonl_path: str,
        vocab: Optional[GraphVocabulary] = None,
        text_embed_dir: Optional[str] = None,
        text_embed_dict: Optional[dict] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            jsonl_path:       Path to NAD_triplet_dataset.jsonl
            vocab:            Pre-built GraphVocabulary. If None, builds from data.
            text_embed_dir:   Directory with pre-extracted text embeddings (.npy).
                              If None, uses dummy embeddings.
            text_embed_dict:  Dict mapping sample_id (str) -> FloatTensor [768].
                              Loaded from a .pt file. Takes priority over
                              text_embed_dir if both are provided.
            max_samples:      Limit dataset size (for debugging).
        """
        super().__init__()
        self.text_embed_dir = text_embed_dir

        # ── Real text embedding dict (keyed by sample_id) ──────────────
        # ALIGNMENT GUARANTEE: Each sample carries its own sample_id.
        # In __getitem__, we look up text_embed_dict[sample_id], so the
        # embedding is ALWAYS paired with the correct graph — regardless
        # of DataLoader shuffling, batching, or worker parallelism.
        self.text_embed_dict = text_embed_dict

        # Load raw data
        self.data: List[dict] = []
        with open(jsonl_path, "r") as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                self.data.append(json.loads(line))

        # Build or reuse vocabulary
        if vocab is None:
            self.vocab = GraphVocabulary()
            self.vocab.build_from_data(self.data)
        else:
            self.vocab = vocab

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]
        sample_id = sample["sample_id"]  # e.g., "cifar10_s000000_0cdeeba3"

        # --- Encode Parent Graph ---
        p_types, p_attrs, p_adj, p_mask = encode_graph(
            sample["parent_graph"], self.vocab
        )

        # --- Encode Child Graph ---
        c_types, c_attrs, c_adj, c_mask = encode_graph(
            sample["child_graph"], self.vocab
        )

        # --- Text Embedding ---
        # PRIORITY: text_embed_dict > text_embed_dir > dummy fallback
        #
        # WHY sample_id KEYING PREVENTS MISALIGNMENT:
        #   DataLoader shuffles INDICES, not tensors. When __getitem__(idx)
        #   is called, we extract sample_id from self.data[idx] and look up
        #   text_embed_dict[sample_id]. Since the sample_id is read from the
        #   SAME row as the graphs, the embedding is ALWAYS paired with the
        #   correct parent/child graphs — no matter what order the DataLoader
        #   yields batches, how many workers fetch in parallel, or whether
        #   shuffle=True.
        if self.text_embed_dict is not None and sample_id in self.text_embed_dict:
            # Real Baidu embedding from .pt dict:  {sample_id: Tensor[768]}
            text_emb = self.text_embed_dict[sample_id].float()  # [768]
        elif self.text_embed_dir is not None:
            # Legacy .npy per-file path
            embed_path = os.path.join(
                self.text_embed_dir, sample_id + ".npy"
            )
            text_emb = torch.from_numpy(np.load(embed_path))    # [768]
        else:
            # Dummy fallback (deterministic random from text hash)
            text_emb = get_dummy_text_embedding(sample["text"]) # [768]

        return {
            # Parent DAG  -----------------------------------------------
            "parent_node_types": p_types,   # [N_MAX]           = [110]
            "parent_node_attrs": p_attrs,   # [N_MAX x K]       = [110 x 8]
            "parent_adj":        p_adj,     # [N_MAX x N_MAX]   = [110 x 110]
            "parent_mask":       p_mask,    # [N_MAX]           = [110]
            # Child DAG  ------------------------------------------------
            "child_node_types":  c_types,   # [N_MAX]           = [110]
            "child_node_attrs":  c_attrs,   # [N_MAX x K]       = [110 x 8]
            "child_adj":         c_adj,     # [N_MAX x N_MAX]   = [110 x 110]
            "child_mask":        c_mask,    # [N_MAX]           = [110]
            # Text  -----------------------------------------------------
            "text_embedding":    text_emb,  # [TEXT_EMBED_DIM]  = [768]
        }


# ============================================================================
# 6. DATALOADER FACTORY
# ============================================================================

def create_dataloaders(
    jsonl_path: str,
    batch_size: int = 32,
    train_ratio: float = 0.9,
    text_embed_dir: Optional[str] = None,
    text_embed_dict: Optional[dict] = None,
    num_workers: int = 4,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, GraphVocabulary]:
    """
    Create train/val DataLoaders with a shared vocabulary.
    
    Args:
        text_embed_dict: Optional dict {sample_id: Tensor[768]} from .pt file.
                         If provided, real embeddings are used instead of dummy.
    
    Returns:
        train_loader : DataLoader
        val_loader   : DataLoader
        vocab        : GraphVocabulary (shared)
    
    Batched tensor shapes from the DataLoader:
        parent_node_types : [B x N_MAX]           = [B x 110]
        parent_node_attrs : [B x N_MAX x K]       = [B x 110 x 8]
        parent_adj        : [B x N_MAX x N_MAX]   = [B x 110 x 110]
        parent_mask       : [B x N_MAX]           = [B x 110]
        child_node_types  : [B x N_MAX]           = [B x 110]
        child_node_attrs  : [B x N_MAX x K]       = [B x 110 x 8]
        child_adj         : [B x N_MAX x N_MAX]   = [B x 110 x 110]
        child_mask        : [B x N_MAX]           = [B x 110]
        text_embedding    : [B x TEXT_EMBED_DIM]   = [B x 768]
    """
    # Build full dataset to construct vocabulary
    full_dataset = NADTripletDataset(
        jsonl_path=jsonl_path,
        vocab=None,
        text_embed_dir=text_embed_dir,
        text_embed_dict=text_embed_dict,
        max_samples=max_samples,
    )
    vocab = full_dataset.vocab

    # Split indices
    n = len(full_dataset)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    split = int(n * train_ratio)
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, vocab


# ============================================================================
# 7. QUICK SANITY CHECK  (run this file directly)
# ============================================================================

if __name__ == "__main__":
    import time

    JSONL_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "NAD_triplet_dataset.jsonl",
    )

    print("=" * 70)
    print("Step 1 Sanity Check: Data Parsing & Dataloader")
    print("=" * 70)

    # --- Build dataset & vocab ---
    t0 = time.time()
    train_loader, val_loader, vocab = create_dataloaders(
        jsonl_path=JSONL_PATH,
        batch_size=4,
        train_ratio=0.9,
        num_workers=0,       # 0 for debugging
        max_samples=100,     # small subset for quick test
    )
    print(f"\nDataset built in {time.time() - t0:.2f}s")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- Vocabulary summary ---
    print(f"\n--- Vocabulary ---")
    print(vocab.summary())
    print(f"Total multi-discrete dims: 1 (node_type) + {len(vocab.ATTR_KEYS)} (attrs)")
    print(f"  node_type classes     : {vocab.num_node_types}")
    for i, key in enumerate(vocab.ATTR_KEYS):
        print(f"  attr[{i}] {key:20s}: {vocab.num_attr_classes(key)} classes")

    # --- Check one batch ---
    print(f"\n--- First batch shapes ---")
    batch = next(iter(train_loader))
    for key, val in batch.items():
        print(f"  {key:25s}: {val.shape}  dtype={val.dtype}")

    # --- Verify padding mask consistency ---
    print(f"\n--- Padding mask check ---")
    p_mask = batch["parent_mask"]       # [B x 110]
    p_types = batch["parent_node_types"]  # [B x 110]
    for b in range(p_mask.shape[0]):
        n_real = int(p_mask[b].sum().item())
        # All padding positions should have type index 0 (<EMPTY>)
        pad_types = p_types[b, n_real:]
        assert (pad_types == 0).all(), f"Batch {b}: padding nodes have non-EMPTY types!"
    print("  ✓ All padding nodes have <EMPTY> type (idx=0)")

    # --- Verify adjacency symmetry and no self-loops in padding ---
    print(f"\n--- Adjacency check ---")
    c_adj = batch["child_adj"]           # [B x 110 x 110]
    c_mask = batch["child_mask"]         # [B x 110]
    for b in range(c_adj.shape[0]):
        n_real = int(c_mask[b].sum().item())
        # No edges should involve padding nodes
        pad_region = c_adj[b, n_real:, :]  # rows beyond real nodes
        assert (pad_region == 0).all(), f"Batch {b}: edges in padding row region!"
        pad_region_col = c_adj[b, :, n_real:]  # cols beyond real nodes
        assert (pad_region_col == 0).all(), f"Batch {b}: edges in padding col region!"
    print("  ✓ No edges involve padding nodes")

    # --- Round-trip decode check (type index -> name) ---
    print(f"\n--- Decode sample (batch item 0) ---")
    b = 0
    n_real = int(batch["parent_mask"][b].sum().item())
    print(f"  Parent graph: {n_real} real nodes")
    for i in range(min(n_real, 10)):
        t_idx = batch["parent_node_types"][b, i].item()
        t_name = vocab.idx_to_node_type[t_idx]
        attrs = []
        for k_idx, key in enumerate(vocab.ATTR_KEYS):
            a_idx = batch["parent_node_attrs"][b, i, k_idx].item()
            if a_idx > 1:  # not <EMPTY> or <NONE>
                a_val = vocab.idx_to_attr[key][a_idx]
                attrs.append(f"{key}={a_val}")
        attr_str = f"({', '.join(attrs)})" if attrs else ""
        print(f"    node {i}: {t_name}{attr_str}")

    print("\n" + "=" * 70)
    print("✓ Step 1 complete — all shapes and assertions passed.")
    print("=" * 70)
