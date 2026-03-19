"""
Step 3: Predictor Network for Classifier-Guided Diffusion
==========================================================
Predicts the Cosine Similarity between a Graph Vector (derived from the
parent + estimated-clean-child architecture pair) and the Baidu Text Vector.

Key design:  The estimated clean child G_hat_0 comes as LOGITS from the
Denoising Net (not discrete indices).  We use Softmax relaxations multiplied
by embedding matrices so that gradients flow back through G_hat_0 to the
Denoising Net during guided inference.

Architecture:
  - Differentiable soft-embedding for G_hat_0 logits
  - Standard embedding for G_parent (clean, discrete)
  - Concatenate child + parent features, process with 3-layer GCN
  - Global Mean Pooling (masked) → Graph_Vector [B x D]
  - Cosine Similarity with Text Embedding → scalar [B x 1]

Tensor shape conventions:
  B     = batch size
  N     = N_MAX = 110
  K     = 8  (attribute keys)
  D     = hidden dim
  D_txt = 768  (Baidu text embedding)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional

from step1_dataset import N_MAX, TEXT_EMBED_DIM, GraphVocabulary


# ============================================================================
# 1. DIFFERENTIABLE SOFT-EMBEDDING
# ============================================================================

class SoftNodeEmbedding(nn.Module):
    """
    Embeds node features that may be either:
      (a) Hard indices (LongTensor) — standard lookup
      (b) Soft distributions (FloatTensor logits/probs) — differentiable matmul

    This allows gradients to flow back from the Predictor through G_hat_0
    to the Denoising Net during classifier guidance.

    Hard mode:
        node_types: [B x N]     LongTensor  →  lookup  →  [B x N x D]
        node_attrs: [B x N x K] LongTensor  →  lookup  →  [B x N x D]

    Soft mode:
        node_type_probs: [B x N x C_type]          → matmul with embed weight → [B x N x D]
        node_attr_probs: list of K [B x N x C_k]   → matmul with embed weight → [B x N x D]
    """

    def __init__(self, num_node_types: int, attr_dims: List[int], embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        # Shared embedding tables (used for both hard and soft lookup)
        self.type_embed = nn.Embedding(num_node_types, embed_dim)   # [C_type x D]
        self.attr_embeds = nn.ModuleList([
            nn.Embedding(n_classes, embed_dim)                       # [C_k x D]
            for n_classes in attr_dims
        ])

    def forward(
        self,
        node_types=None,        # Hard: [B x N] LongTensor
        node_attrs=None,        # Hard: [B x N x K] LongTensor
        node_type_probs=None,   # Soft: [B x N x C_type] FloatTensor
        node_attr_probs=None,   # Soft: list of K [B x N x C_k] FloatTensor
    ) -> torch.Tensor:
        """
        Returns: [B x N x D]
        
        Exactly one of (node_types, node_type_probs) must be provided.
        """
        D = self.embed_dim

        # --- Type embedding ---
        if node_type_probs is not None:
            # Soft path: [B x N x C_type] @ [C_type x D] -> [B x N x D]
            emb = torch.matmul(
                node_type_probs,                            # [B x N x C_type]
                self.type_embed.weight                      # [C_type x D]
            )                                               # [B x N x D]
        else:
            # Hard path: standard lookup [B x N] -> [B x N x D]
            emb = self.type_embed(node_types)               # [B x N x D]

        # --- Attribute embeddings ---
        K = len(self.attr_embeds)
        if node_attr_probs is not None:
            # Soft path: for each attribute key k
            for k in range(K):
                # [B x N x C_k] @ [C_k x D] -> [B x N x D]
                attr_emb_k = torch.matmul(
                    node_attr_probs[k],                     # [B x N x C_k]
                    self.attr_embeds[k].weight              # [C_k x D]
                )                                           # [B x N x D]
                emb = emb + attr_emb_k                      # [B x N x D]
        else:
            # Hard path: standard lookup for each attribute key k
            for k in range(K):
                attr_k = node_attrs[:, :, k]                # [B x N]
                emb = emb + self.attr_embeds[k](attr_k)     # [B x N x D]

        return emb                                          # [B x N x D]


# ============================================================================
# 2. GCN LAYER  (simple, lightweight)
# ============================================================================

class GCNLayer(nn.Module):
    """
    Single Graph Convolutional layer: h' = σ(D^{-1} A H W + b)
    
    Uses degree-normalized adjacency for message passing.
    Lightweight: no BatchNorm (predictor is small), just Linear + ReLU.

    Input/Output: [B x N x D]
    Adjacency:    [B x N x N]
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)            # [D_in -> D_out]
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, h: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h:    [B x N x D_in]   — node features
            adj:  [B x N x N]      — adjacency (directed)
            mask: [B x N]          — padding mask (1=real, 0=pad)
        Returns:
            h':   [B x N x D_out]  — updated features
        """
        B, N, D_in = h.shape

        # Make adjacency bidirectional + add self-loops for GCN stability
        adj_sym = adj + adj.transpose(1, 2)                 # [B x N x N]
        adj_sym = (adj_sym > 0).float()                     # binarize
        eye = torch.eye(N, device=adj.device).unsqueeze(0)  # [1 x N x N]
        adj_hat = adj_sym + eye                             # [B x N x N]  (A + I)

        # Zero out padding rows/cols to prevent spurious message passing
        mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)     # [B x N x N]
        adj_hat = adj_hat * mask_2d                         # [B x N x N]

        # Degree normalization: D^{-1} * (A + I)
        deg = adj_hat.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B x N x 1]
        adj_norm = adj_hat / deg                            # [B x N x N]

        # Message passing: [B x N x N] @ [B x N x D_in] -> [B x N x D_in]
        h_agg = torch.bmm(adj_norm, h)                     # [B x N x D_in]

        # Linear transform: [B x N x D_in] -> [B x N x D_out]
        out = self.linear(h_agg)                            # [B x N x D_out]

        # Activation + dropout
        out = F.relu(out)                                   # [B x N x D_out]
        out = self.dropout(out)                             # [B x N x D_out]

        # Zero out padding nodes
        out = out * mask.unsqueeze(-1)                      # [B x N x D_out]

        return out                                          # [B x N x D_out]


# ============================================================================
# 3. PREDICTOR NETWORK
# ============================================================================

class PredictorNetwork(nn.Module):
    """
    Predicts Cosine Similarity between a pooled Graph Vector and the
    Baidu Text Embedding.

    Inputs:
        Parent DAG:  (node_types, node_attrs, adj, mask)  — discrete, clean
        Child  DAG:  logits from Denoising Net OR discrete clean child
            Soft mode: (node_type_logits, node_attr_logits, edge_logits, mask)
            Hard mode: (node_types, node_attrs, adj, mask)
        Text embedding: [B x D_txt]

    Output:
        cosine_sim: [B x 1]  — cosine similarity in [-1, 1]
    """

    def __init__(
        self,
        vocab: GraphVocabulary,
        hidden_dim: int = 128,
        num_gcn_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        D = hidden_dim

        # --- Soft-capable embedding for child (estimated G_hat_0) ---
        self.child_embed = SoftNodeEmbedding(
            num_node_types=vocab.num_node_types,
            attr_dims=vocab.attr_dims,
            embed_dim=D,
        )

        # --- Standard embedding for parent (always discrete) ---
        self.parent_embed = SoftNodeEmbedding(
            num_node_types=vocab.num_node_types,
            attr_dims=vocab.attr_dims,
            embed_dim=D,
        )

        # --- Input projection: child_D + parent_D -> D ---
        self.input_proj = nn.Sequential(
            nn.Linear(2 * D, D),                            # [2D -> D]
            nn.ReLU(),
        )

        # --- GCN backbone ---
        self.gcn_layers = nn.ModuleList([
            GCNLayer(D, D, dropout=dropout)
            for _ in range(num_gcn_layers)
        ])

        # --- Graph vector projection (to match text embedding space) ---
        self.graph_proj = nn.Sequential(
            nn.Linear(D, D),                                # [D -> D]
            nn.ReLU(),
            nn.Linear(D, TEXT_EMBED_DIM),                   # [D -> D_txt=768]
        )

        # --- Text projection (optional refinement) ---
        self.text_proj = nn.Sequential(
            nn.Linear(TEXT_EMBED_DIM, TEXT_EMBED_DIM),      # [D_txt -> D_txt]
        )

    def forward(
        self,
        # Parent DAG (always discrete / clean)
        parent_node_types: torch.Tensor,    # [B x N]
        parent_node_attrs: torch.Tensor,    # [B x N x K]
        parent_adj: torch.Tensor,           # [B x N x N]
        parent_mask: torch.Tensor,          # [B x N]
        # Child DAG — soft mode (from Denoising Net)
        child_node_type_logits: Optional[torch.Tensor] = None,  # [B x N x C_type]
        child_node_attr_logits: Optional[List[torch.Tensor]] = None,  # K × [B x N x C_k]
        child_edge_logits: Optional[torch.Tensor] = None,       # [B x N x N x 2]
        child_mask: Optional[torch.Tensor] = None,               # [B x N]
        # Child DAG — hard mode (for training with ground-truth)
        child_node_types: Optional[torch.Tensor] = None,   # [B x N]
        child_node_attrs: Optional[torch.Tensor] = None,   # [B x N x K]
        child_adj: Optional[torch.Tensor] = None,          # [B x N x N]
        # Text
        text_embedding: Optional[torch.Tensor] = None,     # [B x D_txt]
    ) -> torch.Tensor:
        """
        Returns:
            cosine_sim : [B x 1]  — cosine similarity in [-1, 1]
        """
        N = N_MAX  # 110

        # =================================================================
        # (A) EMBED PARENT  (always hard / discrete)
        # =================================================================
        h_parent = self.parent_embed(
            node_types=parent_node_types,                   # [B x N]
            node_attrs=parent_node_attrs,                   # [B x N x K]
        )                                                   # [B x N x D]

        # =================================================================
        # (B) EMBED CHILD  (soft or hard mode)
        # =================================================================
        soft_mode = child_node_type_logits is not None

        if soft_mode:
            # ---- Soft / differentiable path ----
            # Convert logits to probabilities via softmax (temperature=1)
            # CRITICAL: softmax preserves gradient flow back to denoising net
            child_type_probs = F.softmax(
                child_node_type_logits, dim=-1              # [B x N x C_type]
            )                                               # [B x N x C_type]

            child_attr_probs = [
                F.softmax(logits_k, dim=-1)                 # [B x N x C_k]
                for logits_k in child_node_attr_logits
            ]  # list of K × [B x N x C_k]

            h_child = self.child_embed(
                node_type_probs=child_type_probs,           # [B x N x C_type]
                node_attr_probs=child_attr_probs,           # K × [B x N x C_k]
            )                                               # [B x N x D]

            # Edge adjacency from edge logits: softmax → take prob of class 1
            # [B x N x N x 2] -> softmax -> [B x N x N x 2] -> select [:,:,:,1]
            child_adj_soft = F.softmax(
                child_edge_logits, dim=-1                   # [B x N x N x 2]
            )[:, :, :, 1]                                   # [B x N x N]
            # This is a soft adjacency in [0,1] — fully differentiable

            adj_for_mp = child_adj_soft                     # [B x N x N]
            mask_for_pool = child_mask                      # [B x N]

        else:
            # ---- Hard / discrete path (for training) ----
            h_child = self.child_embed(
                node_types=child_node_types,                # [B x N]
                node_attrs=child_node_attrs,                # [B x N x K]
            )                                               # [B x N x D]

            adj_for_mp = child_adj                          # [B x N x N]
            mask_for_pool = child_mask                      # [B x N]

        # =================================================================
        # (C) CONCATENATE PARENT + CHILD → PROJECT
        # =================================================================
        # [B x N x D] cat [B x N x D] -> [B x N x 2D]
        h_cat = torch.cat([h_child, h_parent], dim=-1)      # [B x N x 2D]

        # Project down: [B x N x 2D] -> [B x N x D]
        h = self.input_proj(h_cat)                          # [B x N x D]

        # =================================================================
        # (D) GCN BACKBONE — message passing on merged adjacency
        # =================================================================
        # Merge child + parent adjacency for richer message passing
        adj_merged = (adj_for_mp + parent_adj).clamp(max=1.0)
        #                                                   # [B x N x N]

        # Use the union mask (a node is real if real in either graph)
        union_mask = (parent_mask + mask_for_pool).clamp(max=1.0)
        #                                                   # [B x N]

        for gcn_layer in self.gcn_layers:
            h = gcn_layer(h, adj_merged, union_mask)        # [B x N x D]

        # =================================================================
        # (E) GLOBAL MEAN POOLING  (masked)
        # =================================================================
        # Zero out padding nodes
        h_masked = h * union_mask.unsqueeze(-1)             # [B x N x D]

        # Sum over nodes: [B x N x D] -> [B x D]
        h_sum = h_masked.sum(dim=1)                         # [B x D]

        # Count real nodes per graph: [B x N] -> [B x 1]
        n_real = union_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        #                                                   # [B x 1]

        # Mean pool: [B x D] / [B x 1] -> [B x D]
        graph_vec = h_sum / n_real                          # [B x D]

        # =================================================================
        # (F) PROJECT TO TEXT SPACE → COSINE SIMILARITY
        # =================================================================

        # Project graph vector to text embedding space: [B x D] -> [B x D_txt]
        graph_vec_proj = self.graph_proj(graph_vec)         # [B x D_txt] = [B x 768]

        # Project text embedding (light refinement): [B x D_txt] -> [B x D_txt]
        text_vec = self.text_proj(text_embedding)           # [B x D_txt] = [B x 768]

        # Cosine similarity: [B x D_txt] · [B x D_txt] -> [B x 1]
        cosine_sim = F.cosine_similarity(
            graph_vec_proj, text_vec, dim=-1                # [B]
        ).unsqueeze(-1)                                     # [B x 1]

        return cosine_sim                                   # [B x 1]  in [-1, 1]


# ============================================================================
# 4. SANITY CHECK  (run this file directly)
# ============================================================================

if __name__ == "__main__":
    import os
    import time

    JSONL_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "NAD_triplet_dataset.jsonl",
    )

    from step1_dataset import create_dataloaders
    from step2_denoising_gnn import DenoisingGNN

    print("=" * 70)
    print("Step 3 Sanity Check: Predictor Network")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Build vocab + loader
    # ---------------------------------------------------------------
    train_loader, _, vocab = create_dataloaders(
        jsonl_path=JSONL_PATH,
        batch_size=2,
        num_workers=0,
        max_samples=20,
    )
    print(f"Vocab: {vocab.summary()}")

    # ---------------------------------------------------------------
    # Instantiate both models
    # ---------------------------------------------------------------
    denoiser = DenoisingGNN(vocab=vocab, hidden_dim=64, num_gin_layers=4, dropout=0.3)
    predictor = PredictorNetwork(vocab=vocab, hidden_dim=64, num_gcn_layers=3, dropout=0.2)

    d_params = sum(p.numel() for p in denoiser.parameters())
    p_params = sum(p.numel() for p in predictor.parameters())
    print(f"\nDenoiser params:  {d_params:,}")
    print(f"Predictor params: {p_params:,}")

    # ---------------------------------------------------------------
    # Get a batch and run denoiser first
    # ---------------------------------------------------------------
    batch = next(iter(train_loader))
    B = batch["parent_node_types"].shape[0]
    t = torch.randint(0, 1000, (B,))

    print(f"\n--- Denoiser forward (B={B}) ---")
    node_type_logits, node_attr_logits, edge_logits = denoiser(
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
    print(f"  node_type_logits: {node_type_logits.shape}")  # [B x 110 x C_type]
    print(f"  node_attr_logits: {[l.shape for l in node_attr_logits]}")
    print(f"  edge_logits:      {edge_logits.shape}")       # [B x 110 x 110 x 2]

    # ---------------------------------------------------------------
    # Test 1: Predictor in SOFT mode (gradient through denoiser)
    # ---------------------------------------------------------------
    print(f"\n--- Predictor forward: SOFT mode ---")
    t0 = time.time()
    cos_sim_soft = predictor(
        parent_node_types=batch["parent_node_types"],       # [B x 110]
        parent_node_attrs=batch["parent_node_attrs"],       # [B x 110 x 8]
        parent_adj=batch["parent_adj"],                     # [B x 110 x 110]
        parent_mask=batch["parent_mask"],                   # [B x 110]
        child_node_type_logits=node_type_logits,            # [B x 110 x C_type] (differentiable!)
        child_node_attr_logits=node_attr_logits,            # K × [B x 110 x C_k]
        child_edge_logits=edge_logits,                      # [B x 110 x 110 x 2]
        child_mask=batch["child_mask"],                     # [B x 110]
        text_embedding=batch["text_embedding"],             # [B x 768]
    )
    print(f"  cosine_sim shape: {cos_sim_soft.shape}")
    assert cos_sim_soft.shape == (B, 1), f"Expected [{B} x 1], got {cos_sim_soft.shape}"
    print(f"  ✓ Correct: [B x 1] = [{B} x 1]")
    print(f"  values: {cos_sim_soft.detach().squeeze().tolist()}")
    assert (cos_sim_soft >= -1.0).all() and (cos_sim_soft <= 1.0).all(), \
        f"Cosine similarity out of [-1,1] range!"
    print(f"  ✓ Values in [-1, 1]")
    print(f"  Time: {time.time() - t0:.3f}s")

    # ---------------------------------------------------------------
    # Test 2: Gradient flows from predictor BACK to denoiser
    # ---------------------------------------------------------------
    print(f"\n--- Gradient flow test: Predictor → Denoiser ---")
    # Zero all grads
    denoiser.zero_grad()
    predictor.zero_grad()

    # Backward from predictor output
    loss_guidance = cos_sim_soft.sum()
    loss_guidance.backward()

    # Check denoiser parameters received gradients (through logits)
    denoiser_has_grad = False
    for name, param in denoiser.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            denoiser_has_grad = True
            break
    assert denoiser_has_grad, "No gradient reached the Denoiser from the Predictor!"
    print(f"  ✓ Gradients flow from Predictor back to Denoiser")

    # Check predictor parameters received gradients
    pred_has_grad = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in predictor.parameters()
    )
    assert pred_has_grad, "Some Predictor parameters have no gradient!"
    print(f"  ✓ All Predictor parameters received gradients")

    # ---------------------------------------------------------------
    # Test 3: Predictor in HARD mode (training with ground truth)
    # ---------------------------------------------------------------
    print(f"\n--- Predictor forward: HARD mode ---")
    predictor.zero_grad()
    cos_sim_hard = predictor(
        parent_node_types=batch["parent_node_types"],       # [B x 110]
        parent_node_attrs=batch["parent_node_attrs"],       # [B x 110 x 8]
        parent_adj=batch["parent_adj"],                     # [B x 110 x 110]
        parent_mask=batch["parent_mask"],                   # [B x 110]
        child_node_types=batch["child_node_types"],         # [B x 110]
        child_node_attrs=batch["child_node_attrs"],         # [B x 110 x 8]
        child_adj=batch["child_adj"],                       # [B x 110 x 110]
        child_mask=batch["child_mask"],                     # [B x 110]
        text_embedding=batch["text_embedding"],             # [B x 768]
    )
    print(f"  cosine_sim shape: {cos_sim_hard.shape}")
    assert cos_sim_hard.shape == (B, 1)
    print(f"  ✓ Correct: [B x 1] = [{B} x 1]")
    print(f"  values: {cos_sim_hard.detach().squeeze().tolist()}")

    # Backward in hard mode — should only affect predictor
    loss_hard = cos_sim_hard.sum()
    loss_hard.backward()
    pred_has_grad = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in predictor.parameters()
    )
    assert pred_has_grad
    print(f"  ✓ Hard mode backward OK")

    print("\n" + "=" * 70)
    print("✓ Step 3 complete — shapes, cosine bounds, and gradient flow verified.")
    print("=" * 70)
