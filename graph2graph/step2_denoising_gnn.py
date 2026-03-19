"""
Step 2: Denoising GNN Network for Conditional Graph-to-Graph Translation
=========================================================================
Given a noisy child DAG G_t, a clean parent DAG G_parent, and a Baidu text
vector, predict the clean child DAG G_0 (node types, attributes, edges).

Architecture:
  - Early Concatenation: embed G_t and G_parent independently, expand text
    to N_MAX, then concatenate along feature dim at the first layer.
  - 4 GIN (Graph Isomorphism Network) layers with Dropout(0.3).
  - Output heads: separate linear projections for node_type logits,
    per-attribute logits, and edge logits.

Tensor shape conventions (in comments):
  B     = batch size
  N     = N_MAX = 110
  K     = 8  (number of attribute keys)
  D     = hidden dimension (e.g. 256)
  D_txt = TEXT_EMBED_DIM = 768
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

# Import constants from Step 1
from step1_dataset import N_MAX, TEXT_EMBED_DIM, GraphVocabulary


# ============================================================================
# 1. NODE EMBEDDING  (multi-discrete → continuous vector)
# ============================================================================

class NodeEmbedding(nn.Module):
    """
    Embeds multi-discrete node representation (type + K attributes) into
    a single continuous vector via learned embedding tables + summation.

    Input:
        node_types : LongTensor [B x N]     — node type indices
        node_attrs : LongTensor [B x N x K] — attribute indices per key
    Output:
        node_emb   : FloatTensor [B x N x D] — continuous node features
    """

    def __init__(self, num_node_types: int, attr_dims: List[int], embed_dim: int):
        """
        Args:
            num_node_types: vocabulary size for node types
            attr_dims:      list of vocab sizes per attribute key (length K)
            embed_dim:      output embedding dimension D
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Embedding table for node type
        self.type_embed = nn.Embedding(num_node_types, embed_dim)  # [num_types x D]

        # Embedding table for each attribute key
        self.attr_embeds = nn.ModuleList([
            nn.Embedding(n_classes, embed_dim)  # [n_classes x D]
            for n_classes in attr_dims
        ])

    def forward(self, node_types: torch.Tensor, node_attrs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_types: [B x N]     LongTensor
            node_attrs: [B x N x K] LongTensor
        Returns:
            [B x N x D] FloatTensor
        """
        # Type embedding: [B x N] -> [B x N x D]
        emb = self.type_embed(node_types)                   # [B x N x D]

        # Sum attribute embeddings: each [B x N] -> [B x N x D]
        K = node_attrs.shape[-1]
        for k in range(K):
            attr_k = node_attrs[:, :, k]                    # [B x N]
            emb = emb + self.attr_embeds[k](attr_k)         # [B x N x D]

        return emb                                          # [B x N x D]


# ============================================================================
# 2. GIN LAYER  (Graph Isomorphism Network)
# ============================================================================

class GINLayer(nn.Module):
    """
    Single GIN layer with MLP update and dropout.
    
    h_v^{l+1} = MLP( (1 + eps) * h_v^l  +  sum_{u in N(v)} h_u^l )

    Input/Output: [B x N x D]  (batched node features)
    Adjacency:    [B x N x N]  (used as message-passing mask)
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),      # [D -> D]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),      # [D -> D]
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:   [B x N x D]  — node features
            adj: [B x N x N]  — adjacency (directed, values 0/1)
        Returns:
            h':  [B x N x D]  — updated node features
        """
        B, N, D = h.shape

        # Aggregate neighbor features via adjacency: [B x N x N] @ [B x N x D] -> [B x N x D]
        # Use adj + adj^T to make message passing bidirectional (DAG is directed,
        # but we want information flow in both directions for denoising)
        adj_sym = adj + adj.transpose(1, 2)                 # [B x N x N]
        adj_sym = (adj_sym > 0).float()                     # binarize re-entry

        neigh_sum = torch.bmm(adj_sym, h)                   # [B x N x D]

        # GIN update: (1 + eps) * h + neighbor_sum
        out = (1.0 + self.eps) * h + neigh_sum               # [B x N x D]

        # MLP
        out = self.mlp(out)                                  # [B x N x D]

        # BatchNorm (expects [B*N x D])
        out = out.view(B * N, D)                             # [B*N x D]
        out = self.bn(out)                                   # [B*N x D]
        out = out.view(B, N, D)                              # [B x N x D]

        # Residual + activation + dropout
        out = F.relu(out + h)                                # [B x N x D]
        out = self.dropout(out)                              # [B x N x D]

        return out                                           # [B x N x D]


# ============================================================================
# 3. TIMESTEP EMBEDDING  (sinusoidal)
# ============================================================================

class TimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding, following DDPM convention.

    Input:  t : LongTensor [B]        — diffusion timestep
    Output: e : FloatTensor [B x D]   — timestep embedding
    """

    def __init__(self, embed_dim: int, max_timesteps: int = 1000):
        super().__init__()
        self.embed_dim = embed_dim
        # Precompute sinusoidal table: [max_timesteps x D]
        pe = torch.zeros(max_timesteps, embed_dim)
        pos = torch.arange(0, max_timesteps, dtype=torch.float32).unsqueeze(1)  # [T x 1]
        div = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )  # [D/2]
        pe[:, 0::2] = torch.sin(pos * div)                 # [T x D/2]
        pe[:, 1::2] = torch.cos(pos * div)                 # [T x D/2]
        self.register_buffer("pe", pe)                      # [T x D]

        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] LongTensor — timestep indices
        Returns:
            [B x D] FloatTensor
        """
        emb = self.pe[t]                                    # [B x D]
        emb = self.proj(emb)                                # [B x D]
        return emb                                          # [B x D]


# ============================================================================
# 4. DENOISING GNN  (main model)
# ============================================================================

class DenoisingGNN(nn.Module):
    """
    Denoising network for Conditional Graph-to-Graph Translation.

    Input:
        - Noisy child DAG G_t:    (node_types, node_attrs, adj)
        - Clean parent DAG:       (node_types, node_attrs, adj)
        - Text embedding:         [B x D_txt]
        - Diffusion timestep t:   [B]
        - Child padding mask:     [B x N]

    Output (logits for predicting clean G_0):
        - node_type_logits:  [B x N x num_node_types]
        - node_attr_logits:  list of K tensors, each [B x N x num_classes_k]
        - edge_logits:       [B x N x N x 2]   (binary: edge / no-edge)
    """

    def __init__(
        self,
        vocab: GraphVocabulary,
        hidden_dim: int = 256,
        num_gin_layers: int = 4,
        dropout: float = 0.3,
        max_timesteps: int = 1000,
    ):
        super().__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        D = hidden_dim
        K = len(vocab.ATTR_KEYS)

        # --- Node embeddings (separate for child / parent) ---
        self.child_node_embed = NodeEmbedding(
            num_node_types=vocab.num_node_types,
            attr_dims=vocab.attr_dims,
            embed_dim=D,
        )
        self.parent_node_embed = NodeEmbedding(
            num_node_types=vocab.num_node_types,
            attr_dims=vocab.attr_dims,
            embed_dim=D,
        )

        # --- Text projection: D_txt -> D ---
        self.text_proj = nn.Sequential(
            nn.Linear(TEXT_EMBED_DIM, D),                   # [768 -> D]
            nn.ReLU(),
            nn.Linear(D, D),                                # [D -> D]
        )

        # --- Timestep embedding ---
        self.time_embed = TimestepEmbedding(D, max_timesteps)

        # --- Input projection: 3*D (child + parent + text) + D (time) -> D ---
        #   child embed [B x N x D] + parent embed [B x N x D]
        #   + text expanded [B x N x D] + time expanded [B x N x D]
        self.input_proj = nn.Sequential(
            nn.Linear(4 * D, D),                            # [4D -> D]
            nn.ReLU(),
            nn.Linear(D, D),                                # [D -> D]
        )

        # --- GIN backbone ---
        self.gin_layers = nn.ModuleList([
            GINLayer(D, dropout=dropout)
            for _ in range(num_gin_layers)
        ])

        # --- Output heads ---
        # Node type prediction: D -> num_node_types
        self.node_type_head = nn.Linear(D, vocab.num_node_types)  # [D -> C_type]

        # Attribute prediction: one head per attribute key
        self.attr_heads = nn.ModuleList([
            nn.Linear(D, vocab.num_attr_classes(key))       # [D -> C_attr_k]
            for key in vocab.ATTR_KEYS
        ])

        # Edge prediction: pairwise bilinear via MLP on concatenated node pairs
        # We predict for every (i,j) pair: 2 classes (no-edge, edge)
        self.edge_head = nn.Sequential(
            nn.Linear(2 * D, D),                            # [2D -> D]
            nn.ReLU(),
            nn.Linear(D, 2),                                # [D -> 2]
        )

    def forward(
        self,
        # Noisy child DAG (G_t)
        child_node_types: torch.Tensor,     # [B x N]
        child_node_attrs: torch.Tensor,     # [B x N x K]
        child_adj: torch.Tensor,            # [B x N x N]
        child_mask: torch.Tensor,           # [B x N]
        # Clean parent DAG
        parent_node_types: torch.Tensor,    # [B x N]
        parent_node_attrs: torch.Tensor,    # [B x N x K]
        parent_adj: torch.Tensor,           # [B x N x N]
        # Conditioning
        text_embedding: torch.Tensor,       # [B x D_txt]
        timestep: torch.Tensor,             # [B]
    ):
        """
        Forward pass with explicit shape annotations.

        Returns:
            node_type_logits : [B x N x C_type]       — logits over node types
            node_attr_logits : list of K tensors, each [B x N x C_attr_k]
            edge_logits      : [B x N x N x 2]        — logits for edge prediction
        """
        B = child_node_types.shape[0]
        N = N_MAX   # = 110
        D = self.hidden_dim

        # ===================================================================
        # (A) EMBED NODES
        # ===================================================================

        # Child node embedding:  [B x N] + [B x N x K] -> [B x N x D]
        h_child = self.child_node_embed(
            child_node_types, child_node_attrs
        )                                                   # [B x N x D] = [B x 110 x 256]

        # Parent node embedding: [B x N] + [B x N x K] -> [B x N x D]
        h_parent = self.parent_node_embed(
            parent_node_types, parent_node_attrs
        )                                                   # [B x N x D] = [B x 110 x 256]

        # ===================================================================
        # (B) PROJECT TEXT & TIMESTEP
        # ===================================================================

        # Text: [B x D_txt] -> [B x D] -> expand to [B x N x D]
        h_text = self.text_proj(text_embedding)             # [B x D]     = [B x 256]
        h_text = h_text.unsqueeze(1).expand(-1, N, -1)      # [B x N x D] = [B x 110 x 256]

        # Timestep: [B] -> [B x D] -> expand to [B x N x D]
        h_time = self.time_embed(timestep)                  # [B x D]     = [B x 256]
        h_time = h_time.unsqueeze(1).expand(-1, N, -1)      # [B x N x D] = [B x 110 x 256]

        # ===================================================================
        # (C) EARLY CONCATENATION
        # ===================================================================

        # Concatenate along feature dim: [B x N x 4D]
        h_cat = torch.cat([h_child, h_parent, h_text, h_time], dim=-1)
        #                                                   # [B x N x 4D] = [B x 110 x 1024]

        # Project down: [B x N x 4D] -> [B x N x D]
        h = self.input_proj(h_cat)                          # [B x N x D] = [B x 110 x 256]

        # ===================================================================
        # (D) GIN BACKBONE  — message passing on the noisy child adjacency
        # ===================================================================

        # We use the noisy child adjacency for message passing.
        # Additionally, we inject parent connectivity as extra signal:
        # merge adjacencies so the GNN sees both topologies.
        adj_combined = (child_adj + parent_adj).clamp(max=1.0)
        #                                                   # [B x N x N] = [B x 110 x 110]

        for gin_layer in self.gin_layers:
            h = gin_layer(h, adj_combined)                  # [B x N x D] = [B x 110 x 256]

        # ===================================================================
        # (E) OUTPUT HEADS
        # ===================================================================

        # --- (E1) Node type logits ---
        node_type_logits = self.node_type_head(h)
        #                                                   # [B x N x C_type] = [B x 110 x 27]

        # --- (E2) Per-attribute logits ---
        node_attr_logits = []
        for k, head in enumerate(self.attr_heads):
            logits_k = head(h)                              # [B x N x C_attr_k]
            node_attr_logits.append(logits_k)
        # node_attr_logits: list of K tensors

        # --- (E3) Edge logits ---
        # For every pair (i, j), concatenate h_i and h_j, then classify
        # h_i expanded: [B x N x 1 x D] -> [B x N x N x D]
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)         # [B x N x N x D] = [B x 110 x 110 x 256]
        # h_j expanded: [B x 1 x N x D] -> [B x N x N x D]
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)         # [B x N x N x D] = [B x 110 x 110 x 256]
        # Concatenate: [B x N x N x 2D]
        h_pair = torch.cat([h_i, h_j], dim=-1)             # [B x N x N x 2D] = [B x 110 x 110 x 512]
        # Edge classification: [B x N x N x 2D] -> [B x N x N x 2]
        edge_logits = self.edge_head(h_pair)                # [B x N x N x 2] = [B x 110 x 110 x 2]

        return node_type_logits, node_attr_logits, edge_logits


# ============================================================================
# 5. SANITY CHECK  (run this file directly)
# ============================================================================

if __name__ == "__main__":
    import os, sys, time

    # ---------------------------------------------------------------
    # Build a vocabulary from a small subset for testing
    # ---------------------------------------------------------------
    JSONL_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "NAD_triplet_dataset.jsonl",
    )

    from step1_dataset import create_dataloaders

    print("=" * 70)
    print("Step 2 Sanity Check: Denoising GNN Network")
    print("=" * 70)

    # Small subset for quick check
    train_loader, val_loader, vocab = create_dataloaders(
        jsonl_path=JSONL_PATH,
        batch_size=2,
        num_workers=0,
        max_samples=20,
    )
    print(f"\nVocab: {vocab.summary()}")

    # ---------------------------------------------------------------
    # Instantiate model
    # ---------------------------------------------------------------
    model = DenoisingGNN(
        vocab=vocab,
        hidden_dim=128,       # smaller for quick test
        num_gin_layers=4,
        dropout=0.3,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # ---------------------------------------------------------------
    # Forward pass with dummy batch
    # ---------------------------------------------------------------
    batch = next(iter(train_loader))
    B = batch["parent_node_types"].shape[0]

    # Dummy timestep
    t = torch.randint(0, 1000, (B,))                       # [B]

    print(f"\n--- Forward pass (B={B}) ---")
    t0 = time.time()

    node_type_logits, node_attr_logits, edge_logits = model(
        child_node_types=batch["child_node_types"],         # [B x 110]
        child_node_attrs=batch["child_node_attrs"],         # [B x 110 x 8]
        child_adj=batch["child_adj"],                       # [B x 110 x 110]
        child_mask=batch["child_mask"],                     # [B x 110]
        parent_node_types=batch["parent_node_types"],       # [B x 110]
        parent_node_attrs=batch["parent_node_attrs"],       # [B x 110 x 8]
        parent_adj=batch["parent_adj"],                     # [B x 110 x 110]
        text_embedding=batch["text_embedding"],             # [B x 768]
        timestep=t,                                         # [B]
    )

    print(f"  Forward pass time: {time.time() - t0:.3f}s")

    # ---------------------------------------------------------------
    # Verify output shapes
    # ---------------------------------------------------------------
    print(f"\n--- Output shapes ---")
    print(f"  node_type_logits : {node_type_logits.shape}")
    assert node_type_logits.shape == (B, N_MAX, vocab.num_node_types), \
        f"Expected [B x {N_MAX} x {vocab.num_node_types}], got {node_type_logits.shape}"
    print(f"    ✓ Correct: [B x N x C_type] = [{B} x {N_MAX} x {vocab.num_node_types}]")

    K = len(vocab.ATTR_KEYS)
    assert len(node_attr_logits) == K, f"Expected {K} attribute logits, got {len(node_attr_logits)}"
    for k, key in enumerate(vocab.ATTR_KEYS):
        C_k = vocab.num_attr_classes(key)
        print(f"  attr_logits[{k}] ({key:20s}): {node_attr_logits[k].shape}")
        assert node_attr_logits[k].shape == (B, N_MAX, C_k), \
            f"Expected [B x {N_MAX} x {C_k}], got {node_attr_logits[k].shape}"
        print(f"    ✓ Correct: [B x N x C_{key}] = [{B} x {N_MAX} x {C_k}]")

    print(f"  edge_logits      : {edge_logits.shape}")
    assert edge_logits.shape == (B, N_MAX, N_MAX, 2), \
        f"Expected [B x {N_MAX} x {N_MAX} x 2], got {edge_logits.shape}"
    print(f"    ✓ Correct: [B x N x N x 2] = [{B} x {N_MAX} x {N_MAX} x 2]")

    # ---------------------------------------------------------------
    # Backward pass check (gradients flow)
    # ---------------------------------------------------------------
    print(f"\n--- Backward pass check ---")
    loss = node_type_logits.sum() + edge_logits.sum()
    for logits_k in node_attr_logits:
        loss = loss + logits_k.sum()
    loss.backward()

    # Check some gradients exist
    grad_ok = True
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  ⚠ No gradient for: {name}")
            grad_ok = False
            break
    if grad_ok:
        print(f"  ✓ All parameters received gradients")

    print("\n" + "=" * 70)
    print("✓ Step 2 complete — all shapes and gradients verified.")
    print("=" * 70)
