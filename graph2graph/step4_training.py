"""
Step 4: Training Loop & Loss Masking
======================================
Implements:
  1. Discrete (categorical) diffusion forward process — noise injection
  2. Masked denoising loss (node types, attributes, edges)
  3. Predictor loss (MSE on cosine similarity)
  4. Complete training loop with separate optimizers

Tensor shape conventions:
  B     = batch size
  N     = N_MAX = 110
  K     = 8  (attribute keys)
  D     = hidden dim
  T     = max diffusion timesteps
"""

import os
import sys
import time
import json
import math
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from step1_dataset import (
    N_MAX, TEXT_EMBED_DIM, GraphVocabulary,
    NADTripletDataset, create_dataloaders,
)
from step2_denoising_gnn import DenoisingGNN
from step3_predictor import PredictorNetwork


# ============================================================================
# 1. DISCRETE DIFFUSION FORWARD PROCESS
# ============================================================================

class DiscreteDiffusion:
    """
    Categorical diffusion: at timestep t, each token independently has
    probability beta_t of being replaced with a uniform random category,
    and probability (1 - beta_t) of staying unchanged.

    This is the absorbing/uniform noise approach from Austin et al. (D3PM).

    The noise schedule is a cosine schedule:
        beta_t = 1 - (alpha_bar_t / alpha_bar_{t-1})
        alpha_bar_t = cos^2( (t/T + s) / (1+s) * pi/2 )
    """

    def __init__(self, num_timesteps: int = 1000, s: float = 0.008):
        self.T = num_timesteps

        # Cosine schedule for alpha_bar
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
        alpha_bar = torch.cos(((steps / num_timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]                # normalize so alpha_bar_0 = 1

        # beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}, clamped to [0, 0.999]
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = betas.clamp(min=0.0, max=0.999)

        self.register = {}
        self.betas = betas.float()                          # [T]
        self.alpha_bar = alpha_bar[1:].float()              # [T]  (drop t=0 entry)

    def q_sample(
        self,
        x_0: torch.Tensor,       # [B x ...] LongTensor — clean discrete tokens
        t: torch.Tensor,         # [B]       LongTensor — timesteps
        num_classes: int,         # number of categories for this token type
    ) -> torch.Tensor:
        """
        Forward diffusion: corrupt x_0 at timestep t.

        For each token independently:
          - With probability alpha_bar_t: keep original token
          - With probability (1 - alpha_bar_t): replace with Uniform({0,...,C-1})

        Args:
            x_0:         [B x *]  LongTensor (any shape after batch)
            t:           [B]      LongTensor
            num_classes: int

        Returns:
            x_t:         [B x *]  LongTensor — noisy version
        """
        device = x_0.device
        shape = x_0.shape

        # alpha_bar_t for each sample: [B] -> reshape to broadcast with x_0
        ab_t = self.alpha_bar.to(device)[t]                 # [B]
        # Reshape to [B, 1, 1, ...] to broadcast
        for _ in range(len(shape) - 1):
            ab_t = ab_t.unsqueeze(-1)                       # [B x 1 x ... ]

        # Mask: which tokens to keep (True) vs. replace (False)
        keep_mask = torch.rand_like(x_0.float()) < ab_t     # [B x *] bool

        # Uniform random replacement tokens
        random_tokens = torch.randint_like(x_0, low=0, high=num_classes)
        #                                                   # [B x *]

        # Apply: keep original where keep_mask=True, else random
        x_t = torch.where(keep_mask, x_0, random_tokens)   # [B x *]

        return x_t

    def q_sample_edges(
        self,
        adj_0: torch.Tensor,     # [B x N x N] FloatTensor — clean adjacency (0/1)
        t: torch.Tensor,         # [B]         LongTensor
    ) -> torch.Tensor:
        """
        Forward diffusion on edges (binary: 0 or 1).
        Same approach: with prob alpha_bar_t keep, else uniform Bernoulli(0.5).

        Returns:
            adj_t: [B x N x N] FloatTensor — noisy adjacency
        """
        device = adj_0.device
        B, N, _ = adj_0.shape

        ab_t = self.alpha_bar.to(device)[t]                 # [B]
        ab_t = ab_t.view(B, 1, 1)                           # [B x 1 x 1]

        keep_mask = torch.rand(B, N, N, device=device) < ab_t  # [B x N x N] bool

        random_edges = (torch.rand(B, N, N, device=device) > 0.5).float()
        #                                                   # [B x N x N]

        adj_t = torch.where(keep_mask, adj_0, random_edges) # [B x N x N]

        return adj_t


# ============================================================================
# 2. LOSS FUNCTIONS WITH PADDING MASK
# ============================================================================

def compute_denoising_loss(
    # Model predictions (logits)
    node_type_logits: torch.Tensor,         # [B x N x C_type]
    node_attr_logits: list,                 # K × [B x N x C_k]
    edge_logits: torch.Tensor,              # [B x N x N x 2]
    # Ground-truth clean targets
    node_types_0: torch.Tensor,             # [B x N]
    node_attrs_0: torch.Tensor,             # [B x N x K]
    adj_0: torch.Tensor,                    # [B x N x N]
    # Padding mask (CRITICAL)
    child_mask: torch.Tensor,               # [B x N]   (1=real, 0=pad)
    vocab: GraphVocabulary,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute masked denoising loss.

    Loss = CE(node_types) + sum_k CE(attr_k) + BCE(edges)
    All losses are computed with reduction='none' and multiplied by the
    padding mask BEFORE taking the mean.

    Returns:
        total_loss : scalar tensor
        loss_dict  : dict of individual loss components for logging
    """
    B, N = child_mask.shape   # B, 110
    K = len(vocab.ATTR_KEYS)

    # =======================================================================
    # (A) NODE TYPE LOSS — Cross-Entropy, masked
    # =======================================================================

    # CE expects [B*N, C] and target [B*N]
    # But we compute with reduction='none' to get per-node losses

    # node_type_logits: [B x N x C_type]
    # node_types_0:     [B x N]
    node_type_loss_raw = F.cross_entropy(
        node_type_logits.reshape(B * N, -1),                # [B*N x C_type]
        node_types_0.reshape(B * N),                        # [B*N]
        reduction="none",                                   # [B*N]
    ).reshape(B, N)                                         # [B x N]  ← per-node CE loss

    # MULTIPLY BY MASK: padding nodes contribute 0.0
    node_type_loss_masked = node_type_loss_raw * child_mask  # [B x N] * [B x N] = [B x N]
    #   proof: child_mask[b, n] = 0 for padding → loss[b, n] * 0 = 0 ✓

    # Mean over all REAL nodes only (not over B*N which includes padding)
    num_real_nodes = child_mask.sum().clamp(min=1.0)         # scalar
    node_type_loss = node_type_loss_masked.sum() / num_real_nodes  # scalar

    # =======================================================================
    # (B) ATTRIBUTE LOSSES — Cross-Entropy per key, masked
    # =======================================================================

    attr_losses = []
    for k in range(K):
        # node_attr_logits[k]: [B x N x C_k]
        # node_attrs_0[:, :, k]: [B x N]
        C_k = node_attr_logits[k].shape[-1]
        attr_loss_raw = F.cross_entropy(
            node_attr_logits[k].reshape(B * N, C_k),        # [B*N x C_k]
            node_attrs_0[:, :, k].reshape(B * N),            # [B*N]
            reduction="none",                                # [B*N]
        ).reshape(B, N)                                      # [B x N]  ← per-node CE

        # MULTIPLY BY MASK
        attr_loss_masked = attr_loss_raw * child_mask        # [B x N] * [B x N] = [B x N]
        attr_loss = attr_loss_masked.sum() / num_real_nodes  # scalar
        attr_losses.append(attr_loss)

    total_attr_loss = sum(attr_losses)                       # scalar (sum of K attr losses)

    # =======================================================================
    # (C) EDGE LOSS — Binary Cross-Entropy with Logits, masked
    # =======================================================================

    # edge_logits: [B x N x N x 2]  — class 0=no-edge, class 1=edge
    # adj_0:       [B x N x N]      — target adjacency (0.0 or 1.0)

    # Reshape for CE: treat as N*N "nodes" each with 2-class output
    edge_loss_raw = F.cross_entropy(
        edge_logits.reshape(B * N * N, 2),                   # [B*N*N x 2]
        adj_0.long().reshape(B * N * N),                     # [B*N*N]
        reduction="none",                                    # [B*N*N]
    ).reshape(B, N, N)                                       # [B x N x N]  ← per-edge CE

    # EDGE MASK: both endpoints must be real nodes
    # child_mask: [B x N] → edge_mask: [B x N x N]
    edge_mask = child_mask.unsqueeze(2) * child_mask.unsqueeze(1)
    #            [B x N x 1]          * [B x 1 x N]
    #          = [B x N x N]    (1.0 only where BOTH nodes are real)

    # MULTIPLY BY EDGE MASK
    edge_loss_masked = edge_loss_raw * edge_mask             # [B x N x N] * [B x N x N]
    #   proof: if either node i or j is padding, edge_mask[b,i,j]=0 → loss=0 ✓

    num_real_edges = edge_mask.sum().clamp(min=1.0)          # scalar
    edge_loss = edge_loss_masked.sum() / num_real_edges      # scalar

    # =======================================================================
    # (D) TOTAL DENOISING LOSS
    # =======================================================================

    total_loss = node_type_loss + total_attr_loss + edge_loss  # scalar

    loss_dict = {
        "node_type": node_type_loss.item(),
        "attr_total": total_attr_loss.item(),
        "edge": edge_loss.item(),
        "denoise_total": total_loss.item(),
    }

    return total_loss, loss_dict


def compute_predictor_loss(
    predictor: PredictorNetwork,
    batch: dict,
) -> Tuple[torch.Tensor, float]:
    """
    Predictor loss: MSE between predicted cosine similarity and target=1.0.

    The Predictor receives the CLEAN ground-truth G_child (not noisy),
    because by definition the clean child graph is perfectly aligned with
    the text description → target cosine similarity = 1.0.

    Returns:
        loss   : scalar tensor
        cs_val : predicted cosine similarity for logging
    """
    B = batch["parent_node_types"].shape[0]

    # Forward the predictor in HARD mode (clean discrete inputs)
    cosine_sim = predictor(
        parent_node_types=batch["parent_node_types"],       # [B x N]
        parent_node_attrs=batch["parent_node_attrs"],       # [B x N x K]
        parent_adj=batch["parent_adj"],                     # [B x N x N]
        parent_mask=batch["parent_mask"],                   # [B x N]
        child_node_types=batch["child_node_types"],         # [B x N]
        child_node_attrs=batch["child_node_attrs"],         # [B x N x K]
        child_adj=batch["child_adj"],                       # [B x N x N]
        child_mask=batch["child_mask"],                     # [B x N]
        text_embedding=batch["text_embedding"],             # [B x D_txt]
    )                                                       # [B x 1]

    # Target = 1.0 (perfect alignment)
    target = torch.ones(B, 1, device=cosine_sim.device)     # [B x 1]

    # MSE Loss
    loss = F.mse_loss(cosine_sim, target)                   # scalar

    return loss, cosine_sim.mean().item()


# ============================================================================
# 3. TRAIN STEP
# ============================================================================

def train_step(
    batch: dict,
    denoiser: DenoisingGNN,
    predictor: PredictorNetwork,
    opt_d: torch.optim.Optimizer,
    opt_p: torch.optim.Optimizer,
    diffusion: DiscreteDiffusion,
    vocab: GraphVocabulary,
    device: torch.device,
) -> Dict[str, float]:
    """
    Single training step with two separate optimizer updates.

    Step A:  Update Denoiser  — predict clean G_0 from noisy G_t
    Step B:  Update Predictor — predict cosine similarity from clean graph pair

    Args:
        batch:     dict from DataLoader
        denoiser:  DenoisingGNN model
        predictor: PredictorNetwork model
        opt_d:     optimizer for denoiser
        opt_p:     optimizer for predictor
        diffusion: DiscreteDiffusion instance
        vocab:     GraphVocabulary
        device:    torch device

    Returns:
        log_dict: dict of all loss values for logging
    """
    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}

    B = batch["parent_node_types"].shape[0]

    # ==================================================================
    # STEP A: DENOISER UPDATE
    # ==================================================================
    denoiser.train()
    opt_d.zero_grad()

    # (A1) Sample random timesteps: [B]
    t = torch.randint(0, diffusion.T, (B,), device=device)  # [B]

    # (A2) Apply forward diffusion noise to the CHILD graph
    # Noisy node types: [B x N] → noisy [B x N]
    child_types_t = diffusion.q_sample(
        batch["child_node_types"],                           # [B x N]  = [B x 110]
        t,                                                   # [B]
        num_classes=vocab.num_node_types,
    )                                                        # [B x N]  = [B x 110]

    # Noisy node attributes: [B x N x K] → noisy [B x N x K]
    # Apply noise independently per attribute key
    child_attrs_t = batch["child_node_attrs"].clone()        # [B x N x K]
    for k, key in enumerate(vocab.ATTR_KEYS):
        C_k = vocab.num_attr_classes(key)
        child_attrs_t[:, :, k] = diffusion.q_sample(
            batch["child_node_attrs"][:, :, k],              # [B x N]
            t,                                               # [B]
            num_classes=C_k,
        )                                                    # [B x N]

    # Noisy edges: [B x N x N] → noisy [B x N x N]
    child_adj_t = diffusion.q_sample_edges(
        batch["child_adj"],                                  # [B x N x N] = [B x 110 x 110]
        t,                                                   # [B]
    )                                                        # [B x N x N] = [B x 110 x 110]

    # (A3) Denoiser forward: predict clean G_0 from noisy G_t
    node_type_logits, node_attr_logits, edge_logits = denoiser(
        child_node_types=child_types_t,                      # [B x N]     (noisy)
        child_node_attrs=child_attrs_t,                      # [B x N x K] (noisy)
        child_adj=child_adj_t,                               # [B x N x N] (noisy)
        child_mask=batch["child_mask"],                      # [B x N]
        parent_node_types=batch["parent_node_types"],        # [B x N]     (clean)
        parent_node_attrs=batch["parent_node_attrs"],        # [B x N x K] (clean)
        parent_adj=batch["parent_adj"],                      # [B x N x N] (clean)
        text_embedding=batch["text_embedding"],              # [B x D_txt]
        timestep=t,                                          # [B]
    )
    # node_type_logits: [B x N x C_type]
    # node_attr_logits: K × [B x N x C_k]
    # edge_logits:      [B x N x N x 2]

    # (A4) Compute masked denoising loss
    loss_denoise, denoise_dict = compute_denoising_loss(
        node_type_logits=node_type_logits,
        node_attr_logits=node_attr_logits,
        edge_logits=edge_logits,
        node_types_0=batch["child_node_types"],              # clean targets
        node_attrs_0=batch["child_node_attrs"],
        adj_0=batch["child_adj"],
        child_mask=batch["child_mask"],
        vocab=vocab,
    )

    # (A5) Backward + step
    loss_denoise.backward()
    torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
    opt_d.step()

    # ==================================================================
    # STEP B: PREDICTOR UPDATE
    # ==================================================================
    predictor.train()
    opt_p.zero_grad()

    loss_pred, cs_mean = compute_predictor_loss(predictor, batch)

    loss_pred.backward()
    torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
    opt_p.step()

    # ==================================================================
    # LOGGING
    # ==================================================================
    log_dict = {
        **denoise_dict,
        "pred_loss": loss_pred.item(),
        "pred_cos_sim": cs_mean,
    }

    return log_dict


# ============================================================================
# 4. FULL TRAINING LOOP
# ============================================================================

def train(
    jsonl_path: str,
    hidden_dim: int = 256,
    batch_size: int = 32,
    num_epochs: int = 100,
    lr_denoiser: float = 1e-4,
    lr_predictor: float = 3e-4,
    num_timesteps: int = 1000,
    num_workers: int = 4,
    save_dir: str = "checkpoints",
    log_every: int = 10,
    device_str: str = "auto",
    text_embed_path: str = None,
):
    """
    Full training loop for the Graph-to-Graph Translation baseline.

    Args:
        text_embed_path: Optional path to a .pt file containing a dict
                         {sample_id: Tensor[768]} of real text embeddings.
                         If None or file doesn't exist, dummy embeddings are used.
    """
    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")

    # ---------------------------------------------------------------
    # Load text embedding dict (if provided)
    # ---------------------------------------------------------------
    text_embed_dict = None
    if text_embed_path is not None and os.path.exists(text_embed_path):
        text_embed_dict = torch.load(text_embed_path, map_location="cpu")
        print(f"Loaded real text embeddings: {len(text_embed_dict)} entries "
              f"from {text_embed_path}")
        # Sanity check: peek at first entry
        first_key = next(iter(text_embed_dict))
        first_val = text_embed_dict[first_key]
        print(f"  Example: '{first_key}' → Tensor{list(first_val.shape)}")
    else:
        if text_embed_path is not None:
            print(f"⚠ text_embed_path='{text_embed_path}' not found, using dummy")
        print("Using dummy text embeddings (hash-based deterministic random)")

    # ---------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------
    train_loader, val_loader, test_loader, vocab = create_dataloaders(
        jsonl_path=jsonl_path,
        batch_size=batch_size,
        text_embed_dict=text_embed_dict,
        num_workers=num_workers,
    )
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches, Test: {len(test_loader)} batches")
    print(f"Vocab:\n{vocab.summary()}")

    # ---------------------------------------------------------------
    # Models
    # ---------------------------------------------------------------
    denoiser = DenoisingGNN(
        vocab=vocab,
        hidden_dim=hidden_dim,
        num_gin_layers=4,
        dropout=0.3,
        max_timesteps=num_timesteps,
    ).to(device)

    predictor = PredictorNetwork(
        vocab=vocab,
        hidden_dim=hidden_dim // 2,   # lighter predictor
        num_gcn_layers=3,
        dropout=0.2,
    ).to(device)

    d_params = sum(p.numel() for p in denoiser.parameters())
    p_params = sum(p.numel() for p in predictor.parameters())
    print(f"Denoiser params:  {d_params:,}")
    print(f"Predictor params: {p_params:,}")

    # ---------------------------------------------------------------
    # Optimizers & Schedulers
    # ---------------------------------------------------------------
    opt_d = AdamW(denoiser.parameters(), lr=lr_denoiser, weight_decay=1e-4)
    opt_p = AdamW(predictor.parameters(), lr=lr_predictor, weight_decay=1e-4)

    sched_d = CosineAnnealingLR(opt_d, T_max=num_epochs, eta_min=1e-6)
    sched_p = CosineAnnealingLR(opt_p, T_max=num_epochs, eta_min=1e-6)

    # ---------------------------------------------------------------
    # Diffusion
    # ---------------------------------------------------------------
    diffusion = DiscreteDiffusion(num_timesteps=num_timesteps)

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # --- Train ---
        denoiser.train()
        predictor.train()
        train_logs = []

        for step, batch in enumerate(train_loader, 1):
            log = train_step(
                batch=batch,
                denoiser=denoiser,
                predictor=predictor,
                opt_d=opt_d,
                opt_p=opt_p,
                diffusion=diffusion,
                vocab=vocab,
                device=device,
            )
            train_logs.append(log)

            if step % log_every == 0:
                avg = {k: sum(d[k] for d in train_logs[-log_every:]) / log_every
                       for k in log.keys()}
                print(
                    f"  [Epoch {epoch} Step {step}/{len(train_loader)}] "
                    f"denoise={avg['denoise_total']:.4f} "
                    f"(type={avg['node_type']:.3f} attr={avg['attr_total']:.3f} "
                    f"edge={avg['edge']:.3f}) "
                    f"pred={avg['pred_loss']:.4f} cos={avg['pred_cos_sim']:.3f}"
                )

        # --- Epoch summary ---
        avg = {k: sum(d[k] for d in train_logs) / len(train_logs)
               for k in train_logs[0].keys()}

        # --- Validation ---
        val_loss = validate(
            val_loader, denoiser, predictor, diffusion, vocab, device
        )

        # --- Schedulers ---
        sched_d.step()
        sched_p.step()

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch}/{num_epochs} ({elapsed:.1f}s) | "
            f"Train denoise={avg['denoise_total']:.4f} pred={avg['pred_loss']:.4f} | "
            f"Val denoise={val_loss:.4f} | "
            f"LR_d={opt_d.param_groups[0]['lr']:.6f}"
        )

        # --- Save best ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "denoiser": denoiser.state_dict(),
                "predictor": predictor.state_dict(),
                "vocab_node_types": vocab.node_type_to_idx,
                "vocab_attrs": {k: v for k, v in vocab.attr_to_idx.items()},
                "val_loss": val_loss,
            }, os.path.join(save_dir, "best.pt"))
            print(f"  → Saved best model (val_loss={val_loss:.4f})")

        # Periodic save
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "denoiser": denoiser.state_dict(),
                "predictor": predictor.state_dict(),
            }, os.path.join(save_dir, f"epoch_{epoch}.pt"))

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


# ============================================================================
# 5. VALIDATION
# ============================================================================

@torch.no_grad()
def validate(
    val_loader,
    denoiser: DenoisingGNN,
    predictor: PredictorNetwork,
    diffusion: DiscreteDiffusion,
    vocab: GraphVocabulary,
    device: torch.device,
) -> float:
    """
    Compute average denoising loss on the validation set.
    Uses a fixed timestep sample per batch for reduced variance.
    """
    denoiser.eval()
    predictor.eval()

    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch["parent_node_types"].shape[0]

        # Fixed middle timestep for consistent validation
        t = torch.full((B,), diffusion.T // 2, device=device, dtype=torch.long)

        # Noise child
        child_types_t = diffusion.q_sample(
            batch["child_node_types"], t, vocab.num_node_types
        )
        child_attrs_t = batch["child_node_attrs"].clone()
        for k, key in enumerate(vocab.ATTR_KEYS):
            child_attrs_t[:, :, k] = diffusion.q_sample(
                batch["child_node_attrs"][:, :, k], t, vocab.num_attr_classes(key)
            )
        child_adj_t = diffusion.q_sample_edges(batch["child_adj"], t)

        # Predict
        type_logits, attr_logits, edge_logits = denoiser(
            child_node_types=child_types_t,
            child_node_attrs=child_attrs_t,
            child_adj=child_adj_t,
            child_mask=batch["child_mask"],
            parent_node_types=batch["parent_node_types"],
            parent_node_attrs=batch["parent_node_attrs"],
            parent_adj=batch["parent_adj"],
            text_embedding=batch["text_embedding"],
            timestep=t,
        )

        loss, _ = compute_denoising_loss(
            type_logits, attr_logits, edge_logits,
            batch["child_node_types"], batch["child_node_attrs"],
            batch["child_adj"], batch["child_mask"], vocab,
        )
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ============================================================================
# 6. SANITY CHECK
# ============================================================================

if __name__ == "__main__":
    JSONL_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "NAD_triplet_dataset.jsonl",
    )

    print("=" * 70)
    print("Step 4 Sanity Check: Training Loop & Loss Masking")
    print("=" * 70)

    device = torch.device("cpu")

    # Build small dataset
    train_loader, val_loader, test_loader, vocab = create_dataloaders(
        jsonl_path=JSONL_PATH,
        batch_size=4,
        num_workers=0,
        max_samples=32,
    )

    # Build models (small for testing)
    denoiser = DenoisingGNN(vocab=vocab, hidden_dim=64, num_gin_layers=4).to(device)
    predictor = PredictorNetwork(vocab=vocab, hidden_dim=32, num_gcn_layers=3).to(device)

    opt_d = AdamW(denoiser.parameters(), lr=1e-4)
    opt_p = AdamW(predictor.parameters(), lr=3e-4)
    diffusion = DiscreteDiffusion(num_timesteps=100)

    # ---------------------------------------------------------------
    # Test 1: Single train_step
    # ---------------------------------------------------------------
    print("\n--- Test 1: Single train_step ---")
    batch = next(iter(train_loader))
    log = train_step(batch, denoiser, predictor, opt_d, opt_p, diffusion, vocab, device)
    print(f"  Loss breakdown:")
    for k, v in log.items():
        print(f"    {k:20s}: {v:.6f}")

    # ---------------------------------------------------------------
    # Test 2: Verify mask kills padding gradients
    # ---------------------------------------------------------------
    print("\n--- Test 2: Padding mask proof ---")
    denoiser.zero_grad()
    batch = {k: v.to(device) for k, v in next(iter(train_loader)).items()}
    B = batch["child_mask"].shape[0]
    t = torch.randint(0, 100, (B,))

    # Noise
    child_types_t = diffusion.q_sample(batch["child_node_types"], t, vocab.num_node_types)
    child_attrs_t = batch["child_node_attrs"].clone()
    for k, key in enumerate(vocab.ATTR_KEYS):
        child_attrs_t[:, :, k] = diffusion.q_sample(
            batch["child_node_attrs"][:, :, k], t, vocab.num_attr_classes(key)
        )
    child_adj_t = diffusion.q_sample_edges(batch["child_adj"], t)

    type_logits, attr_logits, edge_logits = denoiser(
        child_types_t, child_attrs_t, child_adj_t, batch["child_mask"],
        batch["parent_node_types"], batch["parent_node_attrs"],
        batch["parent_adj"], batch["text_embedding"], t,
    )

    # Compute per-node type loss (unreduced)
    loss_raw = F.cross_entropy(
        type_logits.reshape(B * N_MAX, -1),
        batch["child_node_types"].reshape(B * N_MAX),
        reduction="none",
    ).reshape(B, N_MAX)                                     # [B x 110]

    mask = batch["child_mask"]                               # [B x 110]
    loss_masked = loss_raw * mask                            # [B x 110]

    # Check: for every padding position, masked loss should be exactly 0
    for b in range(B):
        n_real = int(mask[b].sum().item())
        pad_losses = loss_masked[b, n_real:]
        assert (pad_losses == 0.0).all(), f"Batch {b}: padding loss != 0!"
        # Also verify unmasked raw loss IS nonzero for padding (proving mask matters)
        raw_pad = loss_raw[b, n_real:]
        if raw_pad.numel() > 0 and raw_pad.sum() > 0:
            print(f"  Sample {b}: {N_MAX - n_real} pad nodes had raw loss "
                  f"{raw_pad.mean():.4f} → masked to 0.0 ✓")
    print(f"  ✓ All padding node losses are exactly 0.0 after masking")

    # Edge mask proof
    edge_mask = mask.unsqueeze(2) * mask.unsqueeze(1)        # [B x N x N]
    edge_loss_raw = F.cross_entropy(
        edge_logits.reshape(B * N_MAX * N_MAX, 2),
        batch["child_adj"].long().reshape(B * N_MAX * N_MAX),
        reduction="none",
    ).reshape(B, N_MAX, N_MAX)
    edge_loss_masked = edge_loss_raw * edge_mask
    for b in range(B):
        n_real = int(mask[b].sum().item())
        # Check padding rows
        assert (edge_loss_masked[b, n_real:, :] == 0.0).all()
        # Check padding cols
        assert (edge_loss_masked[b, :, n_real:] == 0.0).all()
    print(f"  ✓ All padding edge losses are exactly 0.0 after masking")

    # ---------------------------------------------------------------
    # Test 3: Run 3 full train steps, loss should decrease
    # ---------------------------------------------------------------
    print("\n--- Test 3: Multi-step training ---")
    losses = []
    for i in range(3):
        batch = next(iter(train_loader))
        log = train_step(batch, denoiser, predictor, opt_d, opt_p, diffusion, vocab, device)
        losses.append(log["denoise_total"])
        print(f"  Step {i+1}: denoise_loss={log['denoise_total']:.4f}  "
              f"pred_loss={log['pred_loss']:.4f}  cos_sim={log['pred_cos_sim']:.3f}")

    # ---------------------------------------------------------------
    # Test 4: Validation
    # ---------------------------------------------------------------
    print("\n--- Test 4: Validation ---")
    val_loss = validate(val_loader, denoiser, predictor, diffusion, vocab, device)
    print(f"  Val denoising loss: {val_loss:.4f}")

    print("\n" + "=" * 70)
    print("✓ Step 4 complete — training loop, loss masking, and validation verified.")
    print("=" * 70)
