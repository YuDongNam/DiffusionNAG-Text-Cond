"""
baseline_diffusion_nag.py
=========================
Text-Guided Neural Architecture Mutation via Frozen DiffusionNAG (CATE).

- Loads pre-trained CATE checkpoint (checkpoints/checkpoint.pth.tar).
- Freezes all CATE parameters.
- Trains only a Text_MLP that conditions via time-embedding addition.
- Training: denoising score matching on X_child.
- Inference: SDEdit-style mutation from X_parent conditioned on text_emb.

This baseline can be paired with a JSONL triplet dataset (parent_graph/text/child_graph)
by encoding DAG strings into fixed-length one-hot sequences and per-sample adjacency masks.
"""

import math
import os
import copy
import json
import re
import hashlib
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# Configuration (inline, matching configs/tr_scorenet.py defaults)
# ============================================================================

def get_default_config(max_node=8, n_vocab=7, data_name='NASBench201'):
    """Return a simple namespace-style config mimicking ml_collections."""

    class Config:
        """Minimal nested config object."""
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    graph_encoder = Config(
        n_layers=12,
        d_model=64,
        n_head=8,
        d_ff=128,
        dropout=0.1,
        n_vocab=int(n_vocab),
    )

    model = Config(
        name='CATE',
        nonlinearity='swish',
        dropout=0.1,
        pos_enc_type=2,
        num_scales=1000,
        sigma_min=0.1,
        sigma_max=5.0,
        graph_encoder=graph_encoder,
    )

    data = Config(
        name=data_name,
        max_node=int(max_node),
        n_vocab=int(n_vocab),
        centered=True,
    )

    config = Config(
        model=model,
        data=data,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )
    return config


# ============================================================================
# CATE Model Components (inlined from models/cate.py & models/transformer.py)
# ============================================================================

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    """Sinusoidal timestep embedding."""
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32,
                                  device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def get_act(config):
    """Activation function from config."""
    name = config.model.nonlinearity.lower()
    if name == 'swish':
        return nn.SiLU()
    elif name == 'elu':
        return nn.ELU()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif name == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError(f'Activation {name} not supported.')


# --- Transformer layers (from models/transformer.py) ---

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.matmul(attn, value), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_k = config.d_model // config.n_head
        self.linears = clones(nn.Linear(self.d_model, self.d_model), 4)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        bs = query.size(0)
        query, key, value = [
            l(x).view(bs, -1, self.n_head, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        x, attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(bs, -1, self.n_head * self.d_k)
        return self.linears[3](x), attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w_1 = nn.Linear(config.d_model, config.d_ff)
        self.w_2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SelfAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x, mask):
        x_ = self.norm(x)
        x_, attn = self.attn(x_, x_, x_, mask)
        return self.dropout(x_) + x, attn


class FeedForwardBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.feed_forward = PositionwiseFeedForward(config)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x):
        x_ = self.norm(x)
        x_ = self.feed_forward(x_)
        return self.dropout(x_) + x


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = SelfAttentionBlock(config)
        self.feed_forward = FeedForwardBlock(config)

    def forward(self, x, mask):
        x, attn = self.self_attn(x, mask)
        x = self.feed_forward(x)
        return x, attn


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = clones(EncoderBlock(config), config.n_layers)
        self.norms = clones(nn.LayerNorm(config.d_model), config.n_layers)

    def forward(self, x, mask):
        outputs = []
        attns = []
        for layer, norm in zip(self.layers, self.norms):
            x, attn = layer(x, mask)
            outputs.append(norm(x))
            attns.append(attn)
        return outputs[-1], outputs, attns


class SemanticEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.fc = nn.Linear(config.n_vocab, config.d_model)

    def forward(self, x):
        return self.fc(x) * math.sqrt(self.d_model)


# --- Graph Encoder ---

class GraphEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_f = Encoder(config)

    def forward(self, x, mask):
        h_f, hs_f, attns_f = self.encoder_f(x, mask)
        h = torch.cat(hs_f, dim=-1)
        return h


# --- MLP ---

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim,
                 use_bn=False, activate_func=F.relu):
        super().__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.use_bn = use_bn
        self.activate_func = activate_func

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            if self.use_bn:
                self.batch_norms = nn.ModuleList()
                for _ in range(num_layers - 1):
                    self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        h = x
        for layer in range(self.num_layers - 1):
            h = self.linears[layer](h)
            if self.use_bn:
                h = self.batch_norms[layer](h)
            h = self.activate_func(h)
        return self.linears[self.num_layers - 1](h)


# --- Positional Encoding ---

class PositionalEncoding_Cell(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        NUM_STAGE = 1
        max_len = int(max_len / NUM_STAGE)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2).float()
        self.encoding[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        self.encoding = torch.cat([self.encoding] * NUM_STAGE, dim=0)

    def forward(self, x):
        _, seq_len, _ = x.size()
        return self.encoding[:seq_len, :].to(x.device)


# --- CATE Score Network ---

class CATE(nn.Module):
    """CATE Transformer Score Network (identical to original)."""
    def __init__(self, config):
        super().__init__()
        self.opEmb = SemanticEmbedding(config.model.graph_encoder)
        self.dropout_op = nn.Dropout(p=config.model.dropout)
        self.d_model = config.model.graph_encoder.d_model
        self.act = get_act(config)

        # Time embedding MLPs
        self.timeEmb1 = nn.Linear(self.d_model, self.d_model * 4)
        self.timeEmb2 = nn.Linear(self.d_model * 4, self.d_model)

        # Graph encoder
        self.graph_encoder = GraphEncoder(config.model.graph_encoder)

        self.fdim = int(config.model.graph_encoder.n_layers *
                        config.model.graph_encoder.d_model)
        self.final = MLP(
            num_layers=3,
            input_dim=self.fdim,
            hidden_dim=2 * self.fdim,
            output_dim=config.data.n_vocab,
            use_bn=False,
            activate_func=F.elu,
        )

        # Positional encoding
        if hasattr(config.model, 'pos_enc_type'):
            self.pos_enc_type = config.model.pos_enc_type
            if self.pos_enc_type == 2:
                self.pos_encoder = PositionalEncoding_Cell(
                    d_model=self.d_model, max_len=config.data.max_node)
            else:
                self.pos_encoder = None
        else:
            self.pos_encoder = None

    def forward(self, X, time_cond, maskX):
        emb_x = self.dropout_op(self.opEmb(X))

        if self.pos_encoder is not None:
            emb_p = self.pos_encoder(emb_x)
            emb_x = emb_x + emb_p

        # Time embedding
        timesteps = time_cond
        emb_t = get_timestep_embedding(timesteps, self.d_model)
        emb_t = self.timeEmb1(emb_t)
        emb_t = self.timeEmb2(self.act(emb_t))
        emb_t = emb_t.unsqueeze(1)
        emb = emb_x + emb_t

        h_x = self.graph_encoder(emb, maskX)
        h_x = self.final(h_x)
        return h_x


# ============================================================================
# Text_MLP: Trainable text conditioning module
# ============================================================================

class TextMLP(nn.Module):
    """MLP mapping text embeddings to CATE's time-embedding space.

    Architecture: Linear(d_text, d_hidden) -> SiLU -> Linear(d_hidden, d_model)
    The last linear layer is ZERO-INITIALIZED.
    """
    def __init__(self, d_text=512, d_hidden=256, d_model=64):
        super().__init__()
        self.fc1 = nn.Linear(d_text, d_hidden)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(d_hidden, d_model)

        # Zero-initialize the last layer
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, text_emb):
        """
        Args:
            text_emb: [B, d_text]
        Returns:
            [B, d_model]
        """
        return self.fc2(self.act(self.fc1(text_emb)))


# ============================================================================
# TextConditionedCATE: Wrapper with frozen CATE + trainable TextMLP
# ============================================================================

class TextConditionedCATE(nn.Module):
    """Wraps a frozen CATE model and injects text via time-embedding addition.

    Forward pass:
        emb_x = opEmb(X) + pos_enc
        emb_t = timeEmb(timesteps)
        text_cond = TextMLP(text_emb)              # <-- trainable
        emb = emb_x + emb_t + text_cond            # injection point
        output = graph_encoder(emb) -> final MLP
    """

    def __init__(self, config, d_text=512, d_hidden=256):
        super().__init__()
        self.cate = CATE(config)
        d_model = config.model.graph_encoder.d_model
        self.text_mlp = TextMLP(d_text=d_text, d_hidden=d_hidden,
                                d_model=d_model)

    def freeze_cate(self):
        """Freeze all CATE parameters."""
        for p in self.cate.parameters():
            p.requires_grad = False

    def forward(self, X, time_cond, maskX, text_emb):
        """
        Args:
            X:         [B, N, n_vocab]  node features
            time_cond: [B]             continuous timestep labels
            maskX:     [B, N, N]       attention mask
            text_emb:  [B, d_text]     text embedding

        Returns:
            score: [B, N, n_vocab]
        """
        # --- Replicate CATE forward with text injection ---
        emb_x = self.cate.dropout_op(self.cate.opEmb(X))

        if self.cate.pos_encoder is not None:
            emb_p = self.cate.pos_encoder(emb_x)
            emb_x = emb_x + emb_p

        # Time embedding
        emb_t = get_timestep_embedding(time_cond, self.cate.d_model)
        emb_t = self.cate.timeEmb1(emb_t)
        emb_t = self.cate.timeEmb2(self.cate.act(emb_t))
        emb_t = emb_t.unsqueeze(1)  # [B, 1, d_model]

        # Text conditioning (trainable)
        text_cond = self.text_mlp(text_emb)         # [B, d_model]
        text_cond = text_cond.unsqueeze(1)           # [B, 1, d_model]

        # Injection: e_cond = e_t + TextMLP(text_emb)
        emb = emb_x + emb_t + text_cond

        # Rest of CATE (frozen)
        h_x = self.cate.graph_encoder(emb, maskX)
        h_x = self.cate.final(h_x)
        return h_x


# ============================================================================
# VE-SDE Helpers (inlined from sde_lib.py)
# ============================================================================

class VESDE:
    """Variance Exploding SDE (minimal inline version)."""

    def __init__(self, sigma_min=0.1, sigma_max=5.0, N=1000):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.N = N
        self.T = 1.0

    def marginal_prob(self, x, t):
        """p_t(x) has mean=x and std=sigma(t)."""
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(
            torch.tensor(2.0 * (np.log(self.sigma_max) -
                                np.log(self.sigma_min)),
                         device=t.device))
        return drift, diffusion

    def prior_sampling(self, shape, device='cpu'):
        return torch.randn(*shape, device=device) * self.sigma_max


# ============================================================================
# Attention mask helper (transitive closure + self-attention)
# ============================================================================

def floyd_mask(adj):
    """Compute transitive-closure attention mask from adjacency matrix.

    Uses repeated matrix squaring instead of O(N³) Floyd–Warshall.
    Converges in O(log₂ N) iterations for DAGs.

    Args:
        adj: [N, N] tensor (single graph, shared across batch).
    Returns:
        mask: [N, N] float tensor.
    """
    r = adj.cpu().numpy().copy()
    N = r.shape[0]
    reach = (r > 0).astype(np.float32)
    for _ in range(int(np.ceil(np.log2(max(N, 2))))):
        reach = ((reach + reach @ reach) > 0).astype(np.float32)
    r = np.where(reach > 0, 1.0, r)
    # Always allow each node to attend to itself.
    np.fill_diagonal(r, 1)
    return torch.tensor(r, dtype=torch.float32)


def build_attention_mask_from_adj(adj, device='cpu'):
    """Build an attention mask from a per-sample adjacency matrix.

    Args:
        adj: [N, N] float/bool tensor with 1 indicating an edge i->j.
        device: torch device.
    Returns:
        mask: [N, N] float tensor in {0,1} with transitive closure and diagonal ones.
    """
    mask = floyd_mask(adj).to(device)
    return mask


def padding_mask_from_onehot(x_onehot, pad_idx):
    """Return a bool padding mask [B,N] where True means valid (non-PAD)."""
    types = torch.argmax(x_onehot, dim=-1)  # [B,N]
    return types != pad_idx


def apply_padding_to_attention_mask(attn_mask, valid_nodes_mask):
    """Apply node-level padding to an [N,N] attention mask.

    Args:
        attn_mask: [N,N] float/bool mask.
        valid_nodes_mask: [N] bool where True means non-PAD node.
    Returns:
        [N,N] float mask where any PAD row/col is fully masked (0),
        and diagonal is 1 only for valid nodes.
    """
    v = valid_nodes_mask.to(attn_mask.device).float()
    pad_block = (v[:, None] * v[None, :])  # [N,N]
    out = attn_mask.float() * pad_block
    # keep self-attention only for valid nodes
    out.fill_diagonal_(0.0)
    out = out + torch.diag(v)
    return out


# ============================================================================
# Data scaler (centered: [0,1] -> [-1,1])
# ============================================================================

def data_scaler(x):
    """Scale from [0,1] to [-1,1]."""
    return x * 2.0 - 1.0


def data_inverse_scaler(x):
    """Scale from [-1,1] to [0,1]."""
    return (x + 1.0) / 2.0


# ============================================================================
# NAD JSONL triplet dataset utilities
# ============================================================================

_NODE_LINE_RE = re.compile(r"^\s*(\d+)\s*:\s*(.+?)\s*$")
_EDGE_LINE_RE = re.compile(r"^\s*(\d+)\s*->\s*(\d+)\s*$")


def parse_nad_graph_string(graph_str):
    """Parse a NAD graph string into (node_ops, edges).

    Expected format (as in NAD_triplet_dataset.jsonl):
    - Optional header like '##ResNetBasicBlock_CIFAR##'
    - Node lines: 'idx:OpName(...)' or 'idx:OpName'
    - Edge lines: 'src->dst'
    """
    node_ops = {}
    edges = []
    for raw in graph_str.splitlines():
        line = raw.strip()
        if not line or line.startswith("##"):
            continue
        m_node = _NODE_LINE_RE.match(line)
        if m_node:
            idx = int(m_node.group(1))
            op = m_node.group(2).strip()
            # Keep only the operator/type prefix before optional '(' for stability.
            op = op.split("(", 1)[0].strip()
            node_ops[idx] = op
            continue
        m_edge = _EDGE_LINE_RE.match(line)
        if m_edge:
            s = int(m_edge.group(1))
            t = int(m_edge.group(2))
            edges.append((s, t))
            continue
    if not node_ops:
        raise ValueError("No nodes parsed from graph_str")
    max_idx = max(node_ops.keys())
    ops = [node_ops.get(i, "PAD") for i in range(max_idx + 1)]
    return ops, edges


def build_vocab_from_jsonl(jsonl_path, max_items=None):
    """Build a node-op vocab from a JSONL triplet dataset."""
    ops = {"PAD"}
    n = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            for k in ("parent_graph", "child_graph"):
                g = rec.get(k)
                if not isinstance(g, str):
                    continue
                node_ops, _ = parse_nad_graph_string(g)
                ops.update(node_ops)
            n += 1
            if max_items is not None and n >= max_items:
                break
    vocab = sorted(ops)
    op_to_idx = {op: i for i, op in enumerate(vocab)}
    return vocab, op_to_idx


def encode_ops_one_hot(node_ops, op_to_idx, max_node):
    """Encode node op sequence to fixed-length one-hot [max_node, V]."""
    if len(node_ops) > max_node:
        raise ValueError(f"Graph has {len(node_ops)} nodes > max_node={max_node}")
    V = len(op_to_idx)
    x = torch.zeros(max_node, V, dtype=torch.float32)
    pad_idx = op_to_idx["PAD"]
    for i in range(max_node):
        op = node_ops[i] if i < len(node_ops) else "PAD"
        j = op_to_idx.get(op, pad_idx)
        x[i, j] = 1.0
    return x


def encode_edges_to_adj(edges, max_node):
    """Encode directed edges into an adjacency matrix [max_node, max_node]."""
    adj = torch.zeros(max_node, max_node, dtype=torch.float32)
    for s, t in edges:
        if 0 <= s < max_node and 0 <= t < max_node:
            adj[s, t] = 1.0
    return adj


def featurize_text_hash(text, d_text=512):
    """Deterministic lightweight text featurizer (no external models).

    Produces a fixed-length float vector suitable as a baseline conditioning signal.
    """
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    v = torch.zeros(d_text, dtype=torch.float32)
    if not tokens:
        return v
    for tok in tokens:
        h = hashlib.sha256(tok.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "little") % d_text
        sign = 1.0 if (h[4] & 1) == 0 else -1.0
        v[idx] += sign
    v = v / (v.norm(p=2) + 1e-8)
    return v


class NADTripletDataset(Dataset):
    """JSONL triplet dataset: (X_parent, text_emb, X_child, dataset_type, adj).

    Reads `NAD_triplet_dataset.jsonl`-style records with fields:
      - parent_graph: str
      - child_graph: str
      - text: str
      - dataset: str
    """

    def __init__(self, jsonl_path, op_to_idx, max_node=110, d_text=512, limit=None):
        super().__init__()
        self.jsonl_path = jsonl_path
        self.op_to_idx = op_to_idx
        self.max_node = int(max_node)
        self.d_text = int(d_text)
        self._offsets = []

        # Build file offsets for random access.
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self._offsets.append(pos)
                    if limit is not None and len(self._offsets) >= limit:
                        break

    def __len__(self):
        return len(self._offsets)

    def __getitem__(self, idx):
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            f.seek(self._offsets[idx])
            rec = json.loads(f.readline())

        parent_ops, parent_edges = parse_nad_graph_string(rec["parent_graph"])
        child_ops, child_edges = parse_nad_graph_string(rec["child_graph"])

        X_parent = encode_ops_one_hot(parent_ops, self.op_to_idx, self.max_node)
        X_child = encode_ops_one_hot(child_ops, self.op_to_idx, self.max_node)
        text_emb = featurize_text_hash(rec.get("text", ""), d_text=self.d_text)
        ds_type = rec.get("dataset", "unknown")

        # For masking we only need a consistent adjacency per sample; use the parent.
        adj = encode_edges_to_adj(parent_edges, self.max_node)
        return X_parent, text_emb, X_child, ds_type, adj


# ============================================================================
# Unified split logic (reproducibility)
# ============================================================================

def load_or_create_dataset_splits(
    *,
    split_path: str,
    n_items: int,
    jsonl_path: Optional[str] = None,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Dict[str, List[int]]:
    """Load dataset split indices from disk or create and persist them.

    The split is defined over the *index space* [0, n_items).
    If `split_path` exists, we load and validate that `n_items` matches.
    Otherwise we deterministically generate a random permutation (seeded),
    partition it into train/val/test, and save it to `split_path`.
    """
    if n_items <= 0:
        raise ValueError(f"n_items must be > 0, got {n_items}")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0,1)")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0,1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    os.makedirs(os.path.dirname(split_path) or ".", exist_ok=True)

    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        meta = payload.get("meta", {})
        saved_n = meta.get("n_items")
        if saved_n is None:
            raise ValueError(f"Split file missing meta.n_items: {split_path}")
        if int(saved_n) != int(n_items):
            raise ValueError(
                f"Split file n_items mismatch: file={saved_n} current={n_items}. "
                f"Refusing to proceed for reproducibility. ({split_path})"
            )
        splits = payload.get("splits")
        if not isinstance(splits, dict) or not all(k in splits for k in ("train", "val", "test")):
            raise ValueError(f"Split file missing splits train/val/test: {split_path}")
        return {k: list(map(int, splits[k])) for k in ("train", "val", "test")}

    # Create new split deterministically (STRATIFIED by JSONL 'dataset' label).
    if jsonl_path is None:
        raise ValueError(
            "jsonl_path is required to CREATE a new stratified split. "
            f"(split_path={split_path})"
        )

    import random as _random

    # Read stratification labels in the same index order as NADTripletDataset:
    # non-empty lines are counted, and indices are in that encountered order.
    labels: List[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        while len(labels) < n_items:
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            rec = json.loads(line)
            lbl = rec.get("dataset", "unknown")
            if not isinstance(lbl, str) or not lbl:
                lbl = "unknown"
            labels.append(lbl)

    if len(labels) != n_items:
        raise ValueError(
            f"Unable to read {n_items} labeled items from {jsonl_path}. "
            f"Read {len(labels)}. Refusing to create split."
        )

    # Group indices by label.
    label_to_indices: Dict[str, List[int]] = {}
    for i, lbl in enumerate(labels):
        label_to_indices.setdefault(lbl, []).append(i)

    def _alloc_counts(n: int):
        """Allocate (n_train, n_val, n_test) with 8:1:1 ratio, robust for small n."""
        if n <= 0:
            return 0, 0, 0
        if n == 1:
            return 1, 0, 0
        if n == 2:
            return 1, 0, 1
        # n >= 3: ensure each split has at least 1 item.
        nt = int(round(n * train_ratio))
        nv = int(round(n * val_ratio))
        nt = max(1, min(nt, n - 2))
        nv = max(1, min(nv, n - nt - 1))
        nte = n - nt - nv
        if nte <= 0:
            nte = 1
            # steal from train preferentially
            if nt > 1:
                nt -= 1
            else:
                nv -= 1
        return nt, nv, nte

    train, val, test = [], [], []
    label_counts = {}
    for lbl, idxs in sorted(label_to_indices.items()):
        # Deterministic shuffle per label group.
        rnd = _random.Random(int(seed) + (abs(hash(lbl)) % 1000003))
        idxs = idxs.copy()
        rnd.shuffle(idxs)

        nt, nv, nte = _alloc_counts(len(idxs))
        label_counts[lbl] = {"total": len(idxs), "train": nt, "val": nv, "test": nte}

        train += idxs[:nt]
        val += idxs[nt: nt + nv]
        test += idxs[nt + nv: nt + nv + nte]

    # Final shuffle within each split for mixed ordering.
    rnd_all = _random.Random(int(seed))
    rnd_all.shuffle(train)
    rnd_all.shuffle(val)
    rnd_all.shuffle(test)

    payload = {
        "meta": {
            "n_items": int(n_items),
            "seed": int(seed),
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "stratify_key": "dataset",
            "label_counts": label_counts,
        },
        "splits": {"train": train, "val": val, "test": test},
    }
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return {"train": train, "val": val, "test": test}


class IndexView(Dataset):
    """A view of a dataset restricted to a list of indices."""

    def __init__(self, base: Dataset, indices: List[int]):
        self.base = base
        self.indices = list(map(int, indices))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]


class DummyTripletDataset(Dataset):
    """Tiny synthetic dataset for smoke-testing the pipeline.

    Produces trivially valid sequences under the (input, output) validity rule:
      - token[0] == input
      - exactly one output token at position 1
      - rest PAD
    """

    def __init__(self, size, op_to_idx, max_node=110, d_text=512, ds_type="dummy"):
        super().__init__()
        self.size = int(size)
        self.op_to_idx = op_to_idx
        self.max_node = int(max_node)
        self.d_text = int(d_text)
        self.ds_type = ds_type

        if "PAD" not in self.op_to_idx or "input" not in self.op_to_idx or "output" not in self.op_to_idx:
            raise ValueError("DummyTripletDataset requires PAD/input/output in vocab.")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        V = len(self.op_to_idx)
        pad = self.op_to_idx["PAD"]
        inp = self.op_to_idx["input"]
        out = self.op_to_idx["output"]

        X_parent = torch.zeros(self.max_node, V, dtype=torch.float32)
        X_child = torch.zeros(self.max_node, V, dtype=torch.float32)
        # input at 0, output at 1, rest PAD
        X_parent[0, inp] = 1.0
        X_parent[1, out] = 1.0
        X_child.copy_(X_parent)
        if self.max_node > 2:
            X_parent[2:, pad] = 1.0
            X_child[2:, pad] = 1.0

        text_emb = torch.zeros(self.d_text, dtype=torch.float32)
        adj = torch.zeros(self.max_node, self.max_node, dtype=torch.float32)
        if self.max_node > 1:
            adj[0, 1] = 1.0
        return X_parent, text_emb, X_child, self.ds_type, adj


def collate_fn(batch):
    """Custom collate that handles string dataset_type."""
    if len(batch[0]) == 4:
        X_parents, text_embs, X_children, ds_types = zip(*batch)
        adjs = None
    else:
        X_parents, text_embs, X_children, ds_types, adjs = zip(*batch)
    X_parents = torch.stack(X_parents)
    text_embs = torch.stack(text_embs)
    X_children = torch.stack(X_children)
    # ds_types stays as a list of strings
    if adjs is None:
        return X_parents, text_embs, X_children, list(ds_types)
    return X_parents, text_embs, X_children, list(ds_types), torch.stack(adjs)


# ============================================================================
# Score function wrapper (VE-SDE continuous)
# ============================================================================

def get_score_fn(sde, model, text_emb, maskX, train=False, continuous=True):
    """Wrap model output as a score function for VE-SDE.

    For VESDE continuous: labels = sigma(t), and the model output IS the score
    (no std rescaling needed unlike VP-SDE).
    """
    def score_fn(x, t):
        if continuous:
            labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
        else:
            raise NotImplementedError("Only continuous mode is supported.")

        if train:
            model.train()
        else:
            model.eval()

        score = model(x, labels, maskX, text_emb)
        return score

    return score_fn


# ============================================================================
# Training: Denoising Score Matching
# ============================================================================

def _node_valid_mask_from_onehot(x_onehot, pad_idx):
    """Return a float mask [B,N,1] with 1 for non-PAD nodes."""
    with torch.no_grad():
        # x_onehot: [B,N,V]
        types = torch.argmax(x_onehot, dim=-1)  # [B,N]
        valid = (types != pad_idx).float()
        return valid.unsqueeze(-1)


def train_step(model, sde, X_child, text_emb, maskX, optimizer,
               node_mask=None, eps=1e-5):
    """One training step: denoising score matching on X_child.

    Args:
        model: TextConditionedCATE
        sde: VESDE instance
        X_child: [B, N, n_vocab] scaled to [-1, 1]
        text_emb: [B, d_text]
        maskX: [B, N, N] attention mask
        optimizer: optimizer for TextMLP params only
        node_mask: optional [B, N, 1] float mask, 1 for valid nodes.
        eps: small constant to avoid t=0

    Returns:
        loss: scalar tensor
    """
    model.train()
    optimizer.zero_grad()

    B = X_child.shape[0]
    device = X_child.device

    # Sample random continuous time t in [eps, T]
    t = torch.rand(B, device=device) * (sde.T - eps) + eps

    # Sample noise
    z = torch.randn_like(X_child)

    # Forward diffuse: x_t = mean + std * z
    mean, std = sde.marginal_prob(X_child, t)
    perturbed_data = mean + std[:, None, None] * z

    # For VESDE continuous: labels = sigma(t)
    labels = sde.marginal_prob(torch.zeros_like(X_child), t)[1]

    # Model output (score prediction)
    score = model(perturbed_data, labels, maskX, text_emb)

    # DSM loss: ||score * std + z||^2 (reduce mean)
    losses = torch.square(score * std[:, None, None] + z)  # [B,N,V]
    if node_mask is not None:
        losses = losses * node_mask  # broadcast over V
        denom = node_mask.sum() * losses.shape[-1]
        loss = losses.sum() / (denom + 1e-8)
    else:
        loss = losses.mean()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.text_mlp.parameters(), max_norm=1.0)
    optimizer.step()

    return float(loss.detach().item())


# ============================================================================
# Inference: SDEdit-style Mutation
# ============================================================================

@torch.no_grad()
def sdedit_mutation(model, sde, X_parent, text_emb, maskX,
                    edit_ratio=0.5, eps=1e-5):
    """SDEdit-style mutation: noise X_parent to t_edit, then reverse-denoise.

    Args:
        model: TextConditionedCATE (eval mode)
        sde: VESDE instance
        X_parent: [B, N, n_vocab] scaled to [-1,1]
        text_emb: [B, d_text]
        maskX: [B, N, N]
        edit_ratio: fraction of max timesteps to noise to (e.g., 0.5)
        eps: numerical stability

    Returns:
        X_edited: [B, N, n_vocab]
    """
    model.eval()
    B = X_parent.shape[0]
    device = X_parent.device

    # Determine t_edit
    t_edit = edit_ratio  # continuous time in [0, T=1]

    # --- Forward: noise X_parent to t_edit ---
    mean, std = sde.marginal_prob(X_parent, torch.full((B,), t_edit,
                                                        device=device))
    z = torch.randn_like(X_parent)
    x = mean + std[:, None, None] * z

    # --- Reverse: denoise from t_edit to eps ---
    N_steps = int(sde.N * edit_ratio)  # number of reverse steps
    timesteps = torch.linspace(t_edit, eps, N_steps, device=device)

    for i in range(N_steps):
        t = timesteps[i]
        vec_t = torch.full((B,), t, device=device)

        # Compute score
        labels = sde.marginal_prob(torch.zeros_like(x), vec_t)[1]
        score = model(x, labels, maskX, text_emb)

        # Euler-Maruyama reverse step
        dt = -1.0 / sde.N
        drift, diffusion = sde.sde(x, vec_t)
        # Reverse drift: drift - diffusion^2 * score
        drift_rev = drift - diffusion[:, None, None] ** 2 * score
        x_mean = x + drift_rev * dt

        # Add noise (except last step)
        if i < N_steps - 1:
            noise = torch.randn_like(x)
            x = x_mean + diffusion[:, None, None] * np.sqrt(-dt) * noise
        else:
            x = x_mean

    return x


# ============================================================================
# Checkpoint loading
# ============================================================================

def load_cate_checkpoint(model, ckpt_path='checkpoints/checkpoint.pth.tar',
                         device='cpu'):
    """Load pre-trained CATE weights into TextConditionedCATE.

    Handles:
      - checkpoint['state_dict'] or checkpoint['model'] formats
      - DataParallel 'module.' prefix stripping
    """
    if not os.path.exists(ckpt_path):
        print(f"[WARNING] Checkpoint not found at {ckpt_path}. "
              "Using randomly initialized weights.")
        return model

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Extract the state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Strip 'module.' prefix if present (DataParallel)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        cleaned_state_dict[new_key] = v

    # Load into the CATE sub-module
    cate_state = model.cate.state_dict()
    matched = {k: v for k, v in cleaned_state_dict.items()
               if k in cate_state and cate_state[k].shape == v.shape}

    cate_state.update(matched)
    model.cate.load_state_dict(cate_state)

    print(f"[INFO] Loaded {len(matched)}/{len(cate_state)} CATE parameters "
          f"from {ckpt_path}")

    return model


# ============================================================================
# Training entrypoint (JSONL NAD triplets)
# ============================================================================

def train(
    *,
    jsonl_path,
    max_node=110,
    d_text=512,
    d_hidden=256,
    batch_size=32,
    lr=2e-4,
    num_steps=1000,
    vocab_scan_limit=None,
    dataset_limit=None,
    split_path="results/dataset_splits.json",
    split_seed=42,
    split_train_ratio=0.8,
    split_val_ratio=0.1,
    device=None,
    save_path="checkpoints/baseline_text_mlp.pth",
    log_every=50,
):
    """Train TextMLP on NAD triplets by denoising score matching on X_child.

    Data flow (NAD-specific):
      JSONL -> (X_child one-hot, text_emb hash, adj) -> maskX=floyd(adj) ->
      scale(X_child) -> add noise -> predict score conditioned on (maskX,text_emb).

    The model forward call used here is bit-identical to the one used during
    SDEdit inference: `model(x, labels, maskX, text_emb)`.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab, op_to_idx = build_vocab_from_jsonl(jsonl_path, max_items=vocab_scan_limit)
    if "PAD" not in op_to_idx or "input" not in op_to_idx or "output" not in op_to_idx:
        raise ValueError("Vocab must contain PAD/input/output for NAD training.")
    pad_idx = op_to_idx["PAD"]

    config = get_default_config(max_node=max_node, n_vocab=len(vocab), data_name="NAD_JSONL_TRAIN")
    config.device = device

    model = TextConditionedCATE(config, d_text=d_text, d_hidden=d_hidden)
    model = load_cate_checkpoint(model, ckpt_path='checkpoints/checkpoint.pth.tar', device='cpu')
    model.freeze_cate()
    model.to(device)

    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    optimizer = torch.optim.Adam(model.text_mlp.parameters(), lr=lr)

    ds_full = NADTripletDataset(
        jsonl_path=jsonl_path,
        op_to_idx=op_to_idx,
        max_node=max_node,
        d_text=d_text,
        limit=dataset_limit,
    )
    splits = load_or_create_dataset_splits(
        split_path=split_path,
        n_items=len(ds_full),
        jsonl_path=jsonl_path,
        seed=split_seed,
        train_ratio=split_train_ratio,
        val_ratio=split_val_ratio,
    )
    ds = IndexView(ds_full, splits["train"])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, drop_last=True)
    it = iter(dl)

    for step in range(1, num_steps + 1):
        try:
            X_parent, text_emb, X_child, ds_types, adj = next(it)
        except StopIteration:
            it = iter(dl)
            X_parent, text_emb, X_child, ds_types, adj = next(it)

        # PAD-masking (node-level): used for BOTH attention and loss.
        valid_nodes = padding_mask_from_onehot(X_child, pad_idx=pad_idx)  # [B,N] bool
        node_mask = _node_valid_mask_from_onehot(X_child, pad_idx=pad_idx).to(device)  # [B,N,1]

        # Build per-sample attention mask from adjacency, then apply padding_mask.
        masks = []
        for a, vn in zip(adj, valid_nodes):
            m = build_attention_mask_from_adj(a, device=device)
            m = apply_padding_to_attention_mask(m, vn.to(device))
            masks.append(m)
        maskX = torch.stack(masks).to(device)

        # Forward diffusion base MUST be X_child.
        X_child = data_scaler(X_child).to(device)
        text_emb = text_emb.to(device)

        loss = train_step(model, sde, X_child, text_emb, maskX, optimizer, node_mask=node_mask)
        if step % log_every == 0 or step == 1:
            print(f"[train] step={step}/{num_steps} loss={loss:.6f}")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[train] saved weights to {save_path}")


# ============================================================================
# __main__: Verification
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train TextMLP on NAD JSONL triplets.")
    parser.add_argument("--jsonl_path", type=str, default="NAD_triplet_dataset.jsonl")
    parser.add_argument("--max_node", type=int, default=110)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--vocab_scan_limit", type=int, default=None)
    parser.add_argument("--dataset_limit", type=int, default=None)
    parser.add_argument("--save_path", type=str, default="checkpoints/baseline_text_mlp.pth")
    parser.add_argument("--split_path", type=str, default="results/dataset_splits.json")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_train_ratio", type=float, default=0.8)
    parser.add_argument("--split_val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    if args.device == "auto":
        device = None
    else:
        device = torch.device(args.device)

    train(
        jsonl_path=args.jsonl_path,
        max_node=args.max_node,
        batch_size=args.batch_size,
        lr=args.lr,
        num_steps=args.num_steps,
        vocab_scan_limit=args.vocab_scan_limit,
        dataset_limit=args.dataset_limit,
        split_path=args.split_path,
        split_seed=args.split_seed,
        split_train_ratio=args.split_train_ratio,
        split_val_ratio=args.split_val_ratio,
        device=device,
        save_path=args.save_path,
    )
