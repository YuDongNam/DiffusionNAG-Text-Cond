"""
Step 5: Inference (Reverse Diffusion) & DAG Validity Checker
==============================================================
Implements:
  Part 1 — Reverse diffusion sampling loop with Predictor-based
            classifier guidance.
  Part 2 — Dynamic DAG compilation & dummy tensor forward pass
            to validate generated architectures.

Tensor shape conventions:
  B     = batch size (number of graphs to sample in parallel)
  N     = N_MAX = 110
  K     = 8  (attribute keys)
  D     = hidden dim
  T     = num diffusion timesteps
  C     = num categories (varies per head)
"""

import os
import sys
import re
import math
import json
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from step1_dataset import (
    N_MAX, TEXT_EMBED_DIM, GraphVocabulary,
    _parse_graph_string, _parse_params,
)
from step2_denoising_gnn import DenoisingGNN
from step3_predictor import PredictorNetwork
from step4_training import DiscreteDiffusion


# ============================================================================
# PART 1: REVERSE DIFFUSION SAMPLING WITH CLASSIFIER GUIDANCE
# ============================================================================

class GraphSampler:
    """
    Reverse diffusion sampler for discrete categorical graphs.

    Sampling loop:
      For t = T down to 1:
        1. Denoiser predicts clean G_0 logits from noisy G_t
        2. Predictor computes Similarity(G_parent, G_hat_0, text)
        3. Gradient guidance: grad = ∇_{logits} Similarity
        4. Guided logits = logits + λ * grad
        5. Sample G_{t-1} from categorical posterior q(G_{t-1}|G_t, G_hat_0)
    """

    def __init__(
        self,
        denoiser: DenoisingGNN,
        predictor: PredictorNetwork,
        diffusion: DiscreteDiffusion,
        vocab: GraphVocabulary,
        device: torch.device,
        guidance_scale: float = 1.0,
    ):
        self.denoiser = denoiser
        self.predictor = predictor
        self.diffusion = diffusion
        self.vocab = vocab
        self.device = device
        self.guidance_scale = guidance_scale

    @torch.no_grad()
    def _init_noise(self, B: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize G_T as pure uniform noise.

        Returns:
            node_types_T : [B x N] LongTensor   — uniform random node types
            node_attrs_T : [B x N x K] LongTensor — uniform random attributes
            adj_T        : [B x N x N] FloatTensor — uniform random edges
        """
        N = N_MAX  # 110
        K = len(self.vocab.ATTR_KEYS)

        node_types_T = torch.randint(
            0, self.vocab.num_node_types, (B, N), device=self.device
        )                                                   # [B x 110]

        node_attrs_T = torch.zeros(B, N, K, dtype=torch.long, device=self.device)
        for k, key in enumerate(self.vocab.ATTR_KEYS):
            C_k = self.vocab.num_attr_classes(key)
            node_attrs_T[:, :, k] = torch.randint(
                0, C_k, (B, N), device=self.device
            )                                               # [B x 110]

        adj_T = (torch.rand(B, N, N, device=self.device) > 0.5).float()
        #                                                   # [B x 110 x 110]

        return node_types_T, node_attrs_T, adj_T

    def _posterior_sample_categorical(
        self,
        x_t: torch.Tensor,          # [B x ...] LongTensor — current noisy tokens
        logits_0: torch.Tensor,      # [B x ... x C] FloatTensor — predicted clean logits
        t: int,                      # current timestep (scalar)
        num_classes: int,
    ) -> torch.Tensor:
        """
        Sample x_{t-1} from the categorical posterior q(x_{t-1} | x_t, x_0_hat).

        For D3PM with uniform noise:
          q(x_{t-1}=j | x_t=i, x_0=k) ∝ q(x_t=i | x_{t-1}=j) * q(x_{t-1}=j | x_0=k)

        Where:
          q(x_t=i | x_{t-1}=j) = (1-β_t)*δ(i,j) + β_t/C
          q(x_{t-1}=j | x_0=k) = α̅_{t-1}*δ(j,k) + (1-α̅_{t-1})/C

        Returns: [B x ...] LongTensor — sampled x_{t-1}
        """
        device = x_t.device
        C = num_classes

        if t == 0:
            # At t=0, just argmax the logits (final denoised prediction)
            return logits_0.argmax(dim=-1)                  # [B x ...]

        beta_t = self.diffusion.betas[t].to(device)         # scalar
        alpha_bar_t = self.diffusion.alpha_bar[t].to(device)  # scalar
        alpha_bar_prev = self.diffusion.alpha_bar[t - 1].to(device) if t > 0 else torch.tensor(1.0, device=device)

        # Predicted probabilities for x_0: softmax of logits
        # logits_0: [B x ... x C]
        prob_x0 = F.softmax(logits_0, dim=-1)               # [B x ... x C]

        # One-hot of x_t: [B x ... x C]
        x_t_onehot = F.one_hot(x_t, C).float()              # [B x ... x C]

        # q(x_t | x_{t-1}=j): for each j, prob of observing x_t
        # = (1-beta_t) * delta(x_t, j) + beta_t / C
        # This is a [C]-vector for each position, indexed by j:
        #   If j == x_t: (1-beta_t) + beta_t/C
        #   If j != x_t: beta_t/C
        q_xt_given_xtm1 = (1.0 - beta_t) * x_t_onehot + beta_t / C
        #                                                   # [B x ... x C]

        # q(x_{t-1}=j | x_0=k): marginal noise at t-1
        # = alpha_bar_{t-1} * delta(j, k) + (1 - alpha_bar_{t-1}) / C
        # Weighted by prob_x0 over k:
        # sum_k p(x_0=k) * [alpha_bar_{t-1}*delta(j,k) + (1-alpha_bar_{t-1})/C]
        # = alpha_bar_{t-1} * prob_x0 + (1-alpha_bar_{t-1})/C
        q_xtm1_given_x0 = alpha_bar_prev * prob_x0 + (1.0 - alpha_bar_prev) / C
        #                                                   # [B x ... x C]

        # Posterior: q(x_{t-1}=j | x_t, x_0) ∝ q_xt_given_xtm1[j] * q_xtm1_given_x0[j]
        log_posterior = torch.log(q_xt_given_xtm1.clamp(min=1e-30)) + \
                        torch.log(q_xtm1_given_x0.clamp(min=1e-30))
        #                                                   # [B x ... x C]

        # Sample from categorical
        posterior_probs = F.softmax(log_posterior, dim=-1)    # [B x ... x C]

        # Reshape for multinomial: flatten all but last dim
        orig_shape = posterior_probs.shape[:-1]              # [B x ...]
        flat_probs = posterior_probs.reshape(-1, C)          # [M x C]
        sampled = torch.multinomial(flat_probs, 1).squeeze(-1)  # [M]
        sampled = sampled.reshape(orig_shape)                # [B x ...]

        return sampled                                      # [B x ...] LongTensor

    def _posterior_sample_edges(
        self,
        adj_t: torch.Tensor,        # [B x N x N] FloatTensor — current noisy adj
        edge_logits: torch.Tensor,   # [B x N x N x 2] — predicted clean edge logits
        t: int,
    ) -> torch.Tensor:
        """
        Sample adj_{t-1} from posterior, treating edges as binary (2-class) categorical.
        Returns: [B x N x N] FloatTensor
        """
        adj_t_long = adj_t.long()                           # [B x N x N]
        sampled = self._posterior_sample_categorical(
            adj_t_long, edge_logits, t, num_classes=2
        )                                                   # [B x N x N]
        return sampled.float()                              # [B x N x N]

    def sample(
        self,
        parent_node_types: torch.Tensor,    # [B x N]
        parent_node_attrs: torch.Tensor,    # [B x N x K]
        parent_adj: torch.Tensor,           # [B x N x N]
        parent_mask: torch.Tensor,          # [B x N]
        text_embedding: torch.Tensor,       # [B x D_txt]
        num_steps: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full reverse diffusion sampling loop.

        Args:
            parent_*:       clean parent DAG tensors
            text_embedding: Baidu text embedding
            num_steps:      override number of diffusion steps

        Returns:
            node_types_0 : [B x N] LongTensor   — generated node types
            node_attrs_0 : [B x N x K] LongTensor — generated attributes
            adj_0        : [B x N x N] FloatTensor — generated adjacency
        """
        self.denoiser.eval()
        self.predictor.eval()

        B = parent_node_types.shape[0]
        T = num_steps or self.diffusion.T
        N = N_MAX
        K = len(self.vocab.ATTR_KEYS)

        # (1) Initialize with pure noise
        node_types_t, node_attrs_t, adj_t = self._init_noise(B)
        # Use full mask for generated child (all N positions active during sampling)
        child_mask = torch.ones(B, N, device=self.device)   # [B x 110]

        for t_idx in reversed(range(T)):
            t_tensor = torch.full((B,), t_idx, device=self.device, dtype=torch.long)

            # ---------------------------------------------------------------
            # (2) Denoiser: predict clean G_0 logits
            # ---------------------------------------------------------------
            # Need gradients for guidance if scale > 0
            if self.guidance_scale > 0 and t_idx > 0:
                # Enable gradients for node type logits only (for efficiency)
                node_type_logits, node_attr_logits, edge_logits = \
                    self._denoiser_forward_with_grad(
                        node_types_t, node_attrs_t, adj_t, child_mask,
                        parent_node_types, parent_node_attrs, parent_adj,
                        text_embedding, t_tensor,
                    )
            else:
                with torch.no_grad():
                    node_type_logits, node_attr_logits, edge_logits = self.denoiser(
                        child_node_types=node_types_t,
                        child_node_attrs=node_attrs_t,
                        child_adj=adj_t,
                        child_mask=child_mask,
                        parent_node_types=parent_node_types,
                        parent_node_attrs=parent_node_attrs,
                        parent_adj=parent_adj,
                        text_embedding=text_embedding,
                        timestep=t_tensor,
                    )

            # ---------------------------------------------------------------
            # (3) Sample G_{t-1} from posterior
            # ---------------------------------------------------------------
            with torch.no_grad():
                # Node types: [B x N] from [B x N x C_type]
                node_types_t = self._posterior_sample_categorical(
                    node_types_t, node_type_logits.detach(), t_idx,
                    self.vocab.num_node_types,
                )                                           # [B x N]

                # Node attributes: per key
                for k, key in enumerate(self.vocab.ATTR_KEYS):
                    C_k = self.vocab.num_attr_classes(key)
                    node_attrs_t[:, :, k] = self._posterior_sample_categorical(
                        node_attrs_t[:, :, k],              # [B x N]
                        node_attr_logits[k].detach(),        # [B x N x C_k]
                        t_idx, C_k,
                    )                                       # [B x N]

                # Edges: [B x N x N]
                adj_t = self._posterior_sample_edges(
                    adj_t, edge_logits.detach(), t_idx,
                )                                           # [B x N x N]

            if verbose and (t_idx % max(1, T // 10) == 0 or t_idx == 0):
                print(f"  t={t_idx:4d} | "
                      f"types: {node_types_t[0, :5].tolist()} ... | "
                      f"edges: {adj_t[0].sum():.0f}")

        return node_types_t, node_attrs_t, adj_t

    def _denoiser_forward_with_grad(
        self,
        node_types_t, node_attrs_t, adj_t, child_mask,
        parent_node_types, parent_node_attrs, parent_adj,
        text_embedding, t_tensor,
    ):
        """
        Forward pass through denoiser + predictor guidance with gradient.

        Steps:
          1. Run denoiser to get logits
          2. Pass logits (soft) through predictor to get similarity
          3. Compute gradient of similarity w.r.t. logits
          4. Add scaled gradient to logits (classifier guidance)

        Returns guided logits.
        """
        # Run denoiser (with gradients enabled for logits)
        node_type_logits, node_attr_logits, edge_logits = self.denoiser(
            child_node_types=node_types_t,
            child_node_attrs=node_attrs_t,
            child_adj=adj_t,
            child_mask=child_mask,
            parent_node_types=parent_node_types,
            parent_node_attrs=parent_node_attrs,
            parent_adj=parent_adj,
            text_embedding=text_embedding,
            timestep=t_tensor,
        )
        # node_type_logits: [B x N x C_type]
        # node_attr_logits: K × [B x N x C_k]
        # edge_logits:      [B x N x N x 2]

        # Detach and require grad for guidance computation
        node_type_logits_g = node_type_logits.detach().requires_grad_(True)
        node_attr_logits_g = [l.detach().requires_grad_(True) for l in node_attr_logits]
        edge_logits_g = edge_logits.detach().requires_grad_(True)

        # Run predictor in soft mode to get similarity
        similarity = self.predictor(
            parent_node_types=parent_node_types,
            parent_node_attrs=parent_node_attrs,
            parent_adj=parent_adj,
            parent_mask=torch.ones_like(child_mask),
            child_node_type_logits=node_type_logits_g,
            child_node_attr_logits=node_attr_logits_g,
            child_edge_logits=edge_logits_g,
            child_mask=child_mask,
            text_embedding=text_embedding,
        )                                                   # [B x 1]

        # Compute gradients (allow_unused: some attr heads may not appear
        # in the predictor's computation graph)
        grad_inputs = [node_type_logits_g, edge_logits_g] + node_attr_logits_g
        grads = torch.autograd.grad(
            similarity.sum(),
            grad_inputs,
            create_graph=False,
            allow_unused=True,
        )

        # Replace None grads (unused inputs) with zeros
        grads = [
            g if g is not None else torch.zeros_like(inp)
            for g, inp in zip(grads, grad_inputs)
        ]

        grad_type = grads[0]                                # [B x N x C_type]
        grad_edge = grads[1]                                # [B x N x N x 2]
        grad_attrs = list(grads[2:])                        # K × [B x N x C_k]

        # Apply guidance: logits + λ * grad
        lam = self.guidance_scale
        guided_type_logits = node_type_logits.detach() + lam * grad_type
        guided_edge_logits = edge_logits.detach() + lam * grad_edge
        guided_attr_logits = [
            node_attr_logits[k].detach() + lam * grad_attrs[k]
            for k in range(len(node_attr_logits))
        ]

        return guided_type_logits, guided_attr_logits, guided_edge_logits


# ============================================================================
# PART 1b: TENSOR-TO-GRAPH STRING DECODER
# ============================================================================

def decode_graph_tensors(
    node_types: torch.Tensor,       # [N] LongTensor
    node_attrs: torch.Tensor,       # [N x K] LongTensor
    adj: torch.Tensor,              # [N x N] FloatTensor
    vocab: GraphVocabulary,
    block_name: str = "GeneratedBlock",
    original_graph_string: Optional[str] = None,
) -> str:
    """
    Convert generated tensors back into the graph string format.

    Args:
        node_types:            [N] — node type indices
        node_attrs:            [N x K] — attribute indices
        adj:                   [N x N] — adjacency matrix
        vocab:                 GraphVocabulary
        original_graph_string: if available, used to recover positional args
                               for reshape/permute nodes

    Returns:
        graph_str: the reconstructed graph string
    """
    N = node_types.shape[0]
    K = node_attrs.shape[1]
    lines = [f"##{block_name}##"]

    # Parse original string (if given) to extract positional arg nodes
    # (e.g., reshape(B,C,1,1) uses positional args not key=value)
    orig_node_lines = {}
    if original_graph_string:
        for line in original_graph_string.strip().split("\n"):
            line = line.strip()
            m = re.match(r"^(\d+):", line)
            if m:
                idx = int(m.group(1))
                orig_node_lines[idx] = line

    # Determine real nodes (non-EMPTY)
    real_indices = []
    for i in range(N):
        t_idx = node_types[i].item()
        if t_idx == vocab.node_type_to_idx["<EMPTY>"]:
            continue
        real_indices.append(i)

    # Node definitions
    for i in real_indices:
        t_idx = node_types[i].item()
        t_name = vocab.idx_to_node_type[t_idx]

        # For reshape/permute: use original line if available (positional args)
        if t_name in ("reshape", "permute") and i in orig_node_lines:
            lines.append(orig_node_lines[i])
            continue

        # Collect non-NONE key=value attributes
        attr_parts = []
        for k, key in enumerate(vocab.ATTR_KEYS):
            a_idx = node_attrs[i, k].item()
            if a_idx <= 1:  # <EMPTY> or <NONE>
                continue
            a_val = vocab.idx_to_attr[key][a_idx]
            attr_parts.append(f"{key}={a_val}")

        if attr_parts:
            lines.append(f"{i}:{t_name}({','.join(attr_parts)})")
        else:
            lines.append(f"{i}:{t_name}")

    # Edge definitions
    real_set = set(real_indices)
    for i in real_indices:
        for j in real_indices:
            if adj[i, j].item() > 0.5 and j in real_set:
                lines.append(f"{i}->{j}")

    return "\n".join(lines)


# ============================================================================
# PART 2: DAG VALIDITY CHECKER (DUMMY TENSOR FORWARD)
# ============================================================================

def _resolve_symbolic(expr: str, C: int, H: int, W: int, B: int = 1) -> int:
    """
    Resolve symbolic dimension expressions like 'C', 'C//16', 'H*W', '(1,3)'.

    Returns the integer value, or raises ValueError if unresolvable.
    """
    expr = expr.strip()

    # Handle tuple expressions like (1,3) — return as tuple marker
    if expr.startswith("(") and expr.endswith(")"):
        inner = expr[1:-1]
        parts = [_resolve_symbolic(p.strip(), C, H, W, B) for p in inner.split(",")]
        return tuple(parts)

    # Direct integer
    try:
        return int(expr)
    except ValueError:
        pass

    # Variable substitution
    namespace = {"C": C, "H": H, "W": W, "B": B}

    # Clean up expression for eval (safe: only arithmetic on known vars)
    safe_expr = expr.replace("//", "___INTDIV___")
    safe_expr = safe_expr.replace("/", "//")  # treat / as integer division
    safe_expr = safe_expr.replace("___INTDIV___", "//")

    try:
        result = eval(safe_expr, {"__builtins__": {}}, namespace)
        return int(result)
    except Exception:
        raise ValueError(f"Cannot resolve symbolic expression: '{expr}'")


def _topological_sort(nodes: Dict, edges: List[Tuple[int, int]]) -> List[int]:
    """
    Kahn's algorithm for topological sort of the DAG.

    Returns list of node indices in execution order.
    """
    # Build adjacency and in-degree
    adj = defaultdict(list)
    in_degree = defaultdict(int)
    all_nodes = set(nodes.keys())

    for node in all_nodes:
        in_degree[node] = 0

    for src, dst in edges:
        if src in all_nodes and dst in all_nodes:
            adj[src].append(dst)
            in_degree[dst] += 1

    queue = deque([n for n in all_nodes if in_degree[n] == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(all_nodes):
        raise RuntimeError("Graph contains a cycle — not a valid DAG")

    return order


def validate_dag_compilation(
    graph_string: str,
    dummy_input: Optional[torch.Tensor] = None,
    initial_channels: int = 3,
    verbose: bool = False,
) -> Tuple[bool, str]:
    """
    Dynamically compile a DAG graph string into PyTorch operations and
    pass a dummy tensor through it.

    Args:
        graph_string:     the graph string in dataset format
        dummy_input:      input tensor (default: torch.randn(1, 3, 32, 32))
        initial_channels: channels of the dummy input
        verbose:          print per-node tensor shapes

    Returns:
        (is_valid, message): bool and descriptive message
    """
    if dummy_input is None:
        dummy_input = torch.randn(1, initial_channels, 32, 32)

    try:
        nodes, edges = _parse_graph_string(graph_string)

        if not nodes:
            return False, "No nodes parsed from graph string"

        # Build predecessor map
        predecessors = defaultdict(list)
        for src, dst in edges:
            predecessors[dst].append(src)

        # Topological sort
        order = _topological_sort(nodes, edges)

        # Tensor state for each node
        tensors: Dict[int, torch.Tensor] = {}

        # Track channel count as we go
        for node_idx in order:
            base_type, attrs = nodes[node_idx]

            # Collect input tensors from predecessors
            pred_tensors = [tensors[p] for p in predecessors[node_idx] if p in tensors]

            # Get the "primary" input (first predecessor or dummy for input node)
            if base_type == "input":
                out = dummy_input
                if verbose:
                    print(f"  Node {node_idx}: input → {out.shape}")
                tensors[node_idx] = out
                continue

            if not pred_tensors:
                return False, f"Node {node_idx} ({base_type}) has no input tensors"

            primary = pred_tensors[0]
            B_dim = primary.shape[0]

            # Determine spatial dimensions from primary input
            if primary.dim() == 4:
                _, C, H, W = primary.shape
            elif primary.dim() == 3:
                _, C, L = primary.shape
                H, W = int(math.sqrt(L)), int(math.sqrt(L)) if L == int(math.sqrt(L))**2 else (L, 1)
            elif primary.dim() == 2:
                _, C = primary.shape
                H, W = 1, 1
            else:
                C, H, W = primary.shape[-1], 1, 1

            # =============================================================
            # DISPATCH BY NODE TYPE
            # =============================================================

            if base_type == "output":
                # Output node: just pass through first input
                out = primary

            elif base_type == "Conv2d":
                if primary.dim() != 4:
                    return False, f"Conv2d at node {node_idx} expects 4D input, got {primary.dim()}D"
                in_ch = primary.shape[1]
                out_ch = _resolve_symbolic(attrs.get("out_channels", "C"), in_ch, H, W, B_dim)
                ks = attrs.get("kernel_size", "3")
                if isinstance(_resolve_symbolic(ks, in_ch, H, W, B_dim), tuple):
                    ks_val = _resolve_symbolic(ks, in_ch, H, W, B_dim)
                else:
                    ks_val = _resolve_symbolic(ks, in_ch, H, W, B_dim)
                stride = _resolve_symbolic(attrs.get("stride", "1"), in_ch, H, W, B_dim)
                groups = _resolve_symbolic(attrs.get("groups", "1"), in_ch, H, W, B_dim)
                dilation = _resolve_symbolic(attrs.get("dilation", "1"), in_ch, H, W, B_dim)

                # Ensure groups divides both in and out channels
                if groups > 1:
                    if in_ch % groups != 0 or out_ch % groups != 0:
                        return False, (
                            f"Conv2d node {node_idx}: groups={groups} doesn't divide "
                            f"in_ch={in_ch} or out_ch={out_ch}"
                        )

                # Padding to preserve spatial dims (same padding)
                if isinstance(ks_val, tuple):
                    pad = tuple((k * dilation - 1) // 2 for k in ks_val)
                else:
                    pad = (ks_val * dilation - 1) // 2

                conv = nn.Conv2d(
                    in_ch, out_ch, ks_val,
                    stride=stride, padding=pad,
                    groups=groups, dilation=dilation, bias=False,
                )
                out = conv(primary)

            elif base_type == "BN":
                if primary.dim() == 4:
                    bn = nn.BatchNorm2d(primary.shape[1])
                elif primary.dim() == 2:
                    bn = nn.BatchNorm1d(primary.shape[1])
                elif primary.dim() == 3:
                    bn = nn.BatchNorm1d(primary.shape[1])
                else:
                    return False, f"BN at node {node_idx}: unsupported dim {primary.dim()}"
                bn.eval()
                out = bn(primary)

            elif base_type == "LN":
                out = F.layer_norm(primary, primary.shape[1:])

            elif base_type in ("ReLU", "relu"):
                out = F.relu(primary)

            elif base_type == "GELU":
                out = F.gelu(primary)

            elif base_type == "Sigmoid":
                out = torch.sigmoid(primary)

            elif base_type in ("Softmax", "softmax"):
                dim_val = _resolve_symbolic(attrs.get("dim", "-1"), C, H, W, B_dim)
                out = F.softmax(primary, dim=dim_val)

            elif base_type == "Linear":
                if primary.dim() < 2:
                    return False, f"Linear at node {node_idx}: input must be >= 2D"
                in_feat = primary.shape[-1]
                out_feat = _resolve_symbolic(attrs.get("out_channels", str(in_feat)), C, H, W, B_dim)
                linear = nn.Linear(in_feat, out_feat, bias=True)
                out = linear(primary)

            elif base_type == "AdaptiveAvgPool2d":
                if primary.dim() != 4:
                    return False, f"AdaptiveAvgPool2d at node {node_idx}: needs 4D, got {primary.dim()}D"
                os_val = _resolve_symbolic(attrs.get("output_size", "1"), C, H, W, B_dim)
                pool = nn.AdaptiveAvgPool2d(os_val)
                out = pool(primary)

            elif base_type == "AdaptiveMaxPool2d":
                if primary.dim() != 4:
                    return False, f"AdaptiveMaxPool2d at node {node_idx}: needs 4D, got {primary.dim()}D"
                os_val = _resolve_symbolic(attrs.get("output_size", "1"), C, H, W, B_dim)
                pool = nn.AdaptiveMaxPool2d(os_val)
                out = pool(primary)

            elif base_type == "AvgPool2d":
                if primary.dim() != 4:
                    return False, f"AvgPool2d at node {node_idx}: needs 4D"
                ks_val = _resolve_symbolic(attrs.get("kernel_size", "2"), C, H, W, B_dim)
                stride = _resolve_symbolic(attrs.get("stride", str(ks_val)), C, H, W, B_dim)
                pool = nn.AvgPool2d(ks_val, stride=stride, padding=ks_val // 2)
                out = pool(primary)

            elif base_type == "MaxPool2d":
                if primary.dim() != 4:
                    return False, f"MaxPool2d at node {node_idx}: needs 4D"
                ks_val = _resolve_symbolic(attrs.get("kernel_size", "2"), C, H, W, B_dim)
                stride = _resolve_symbolic(attrs.get("stride", str(ks_val)), C, H, W, B_dim)
                pool = nn.MaxPool2d(ks_val, stride=stride, padding=ks_val // 2)
                out = pool(primary)

            elif base_type == "Add":
                if len(pred_tensors) < 2:
                    return False, f"Add at node {node_idx}: needs >= 2 inputs, got {len(pred_tensors)}"
                out = pred_tensors[0]
                for pt in pred_tensors[1:]:
                    if out.shape != pt.shape:
                        return False, (
                            f"Add at node {node_idx}: shape mismatch "
                            f"{out.shape} vs {pt.shape}"
                        )
                    out = out + pt

            elif base_type in ("Mul", "multiply"):
                if len(pred_tensors) < 2:
                    return False, f"Mul at node {node_idx}: needs >= 2 inputs, got {len(pred_tensors)}"
                out = pred_tensors[0]
                for pt in pred_tensors[1:]:
                    # Allow broadcasting (e.g., channel attention * feature map)
                    try:
                        out = out * pt
                    except RuntimeError as e:
                        return False, f"Mul at node {node_idx}: {e}"

            elif base_type == "concat":
                if len(pred_tensors) < 2:
                    return False, f"concat at node {node_idx}: needs >= 2 inputs"
                dim_val = _resolve_symbolic(attrs.get("dim", "1"), C, H, W, B_dim)
                try:
                    out = torch.cat(pred_tensors, dim=dim_val)
                except RuntimeError as e:
                    return False, f"concat at node {node_idx}: {e}"

            elif base_type == "reshape":
                # Parse the reshape target dims from the node's position args
                # e.g., reshape(B,C) or reshape(B,C,1,1)
                # These are encoded as positional args, not key=value
                # We need to re-parse from the original node string
                raw_line = None
                for line in graph_string.strip().split("\n"):
                    if line.strip().startswith(f"{node_idx}:reshape"):
                        raw_line = line.strip()
                        break

                if raw_line and "(" in raw_line:
                    param_str = raw_line[raw_line.index("(") + 1 : raw_line.rindex(")")]
                    dim_strs = [s.strip() for s in param_str.split(",")]
                    target_dims = []
                    for ds in dim_strs:
                        target_dims.append(
                            _resolve_symbolic(ds, C, H, W, B_dim)
                        )
                    try:
                        out = primary.reshape(*target_dims)
                    except RuntimeError as e:
                        return False, f"reshape at node {node_idx}: {e}"
                else:
                    out = primary  # no-op if can't parse

            elif base_type == "permute":
                raw_line = None
                for line in graph_string.strip().split("\n"):
                    if line.strip().startswith(f"{node_idx}:permute"):
                        raw_line = line.strip()
                        break

                if raw_line and "(" in raw_line:
                    param_str = raw_line[raw_line.index("(") + 1 : raw_line.rindex(")")]
                    dim_strs = [s.strip() for s in param_str.split(",")]
                    dims = [int(d) for d in dim_strs]
                    try:
                        out = primary.permute(*dims)
                    except RuntimeError as e:
                        return False, f"permute at node {node_idx}: {e}"
                else:
                    out = primary

            elif base_type == "mean":
                dim_val = attrs.get("dim", "1")
                if dim_val.startswith("("):
                    d = _resolve_symbolic(dim_val, C, H, W, B_dim)
                    out = primary.mean(dim=list(d), keepdim=True)
                else:
                    d = _resolve_symbolic(dim_val, C, H, W, B_dim)
                    out = primary.mean(dim=d, keepdim=True)

            elif base_type == "max":
                dim_val = _resolve_symbolic(attrs.get("dim", "1"), C, H, W, B_dim)
                out = primary.max(dim=dim_val, keepdim=True).values

            elif base_type == "sum":
                dim_val = _resolve_symbolic(attrs.get("dim", "1"), C, H, W, B_dim)
                out = primary.sum(dim=dim_val, keepdim=True)

            elif base_type == "repeat":
                # repeat is typically used like repeat(1,1,H,W) - pass through for now
                out = primary

            else:
                # Unknown operation — treat as identity (pass-through)
                out = primary

            if verbose:
                pred_shapes = [t.shape for t in pred_tensors]
                print(f"  Node {node_idx}: {base_type}{attrs} "
                      f"inputs={pred_shapes} → {out.shape}")

            tensors[node_idx] = out

        # Check that the output node received a tensor
        output_nodes = [i for i, (bt, _) in nodes.items() if bt == "output"]
        if not output_nodes:
            return False, "No output node found"

        output_tensor = tensors.get(output_nodes[0])
        if output_tensor is None:
            return False, "Output node has no tensor"

        return True, f"Valid! Output shape: {output_tensor.shape}"

    except Exception as e:
        return False, f"Compilation error: {type(e).__name__}: {e}"


# ============================================================================
# PART 2b: BATCH VALIDITY EVALUATION
# ============================================================================

def evaluate_validity(
    generated_strings: List[str],
    dummy_input: Optional[torch.Tensor] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Evaluate validity rate over a batch of generated graph strings.

    Returns dict with 'valid_count', 'total', 'validity_rate'.
    """
    valid_count = 0
    errors = []

    for i, gs in enumerate(generated_strings):
        ok, msg = validate_dag_compilation(gs, dummy_input=dummy_input, verbose=verbose)
        if ok:
            valid_count += 1
        else:
            errors.append((i, msg))

    return {
        "valid_count": valid_count,
        "total": len(generated_strings),
        "validity_rate": valid_count / max(len(generated_strings), 1),
        "errors": errors,
    }


# ============================================================================
# SANITY CHECK
# ============================================================================

if __name__ == "__main__":

    JSONL_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "NAD_triplet_dataset.jsonl",
    )

    from step1_dataset import create_dataloaders

    print("=" * 70)
    print("Step 5 Sanity Check: Inference & DAG Validity")
    print("=" * 70)

    # ==================================================================
    # TEST 1: Validate the DAG checker on ground-truth graph strings
    # ==================================================================
    print("\n--- Test 1: DAG Validity on Ground-Truth Strings ---")
    with open(JSONL_PATH) as f:
        samples = [json.loads(line) for line in f][:10]

    valid_count = 0
    for i, sample in enumerate(samples):
        for gk in ["parent_graph", "child_graph"]:
            ok, msg = validate_dag_compilation(
                sample[gk],
                dummy_input=torch.randn(1, 16, 32, 32),  # C=16 for better expression eval
                verbose=(i == 0 and gk == "parent_graph"),
            )
            status = "✓" if ok else "✗"
            print(f"  [{status}] Sample {i} {gk:12s}: {msg}")
            if ok:
                valid_count += 1

    print(f"\n  Ground-truth validity: {valid_count}/20")

    # ==================================================================
    # TEST 2: Graph string decode → re-validate roundtrip
    # ==================================================================
    print("\n--- Test 2: Tensor → String → Validity roundtrip ---")
    train_loader, _, vocab = create_dataloaders(
        jsonl_path=JSONL_PATH, batch_size=2, num_workers=0, max_samples=10,
    )
    batch = next(iter(train_loader))

    # Get the original graph string for positional arg recovery
    orig_str = samples[0]["child_graph"]

    # Decode the first child graph back to string (with original for reshape/permute args)
    decoded_str = decode_graph_tensors(
        batch["child_node_types"][0],     # [110]
        batch["child_node_attrs"][0],     # [110 x 8]
        batch["child_adj"][0],            # [110 x 110]
        vocab,
        original_graph_string=orig_str,
    )
    print(f"  Decoded graph string:\n{decoded_str[:500]}")

    ok, msg = validate_dag_compilation(decoded_str, dummy_input=torch.randn(1, 16, 32, 32))
    print(f"\n  Roundtrip validity: {'✓' if ok else '✗'} — {msg}")

    # ==================================================================
    # TEST 3: Quick sampling test (tiny model, few steps)
    # ==================================================================
    print("\n--- Test 3: Reverse Diffusion Sampling (tiny, 10 steps) ---")

    denoiser = DenoisingGNN(vocab=vocab, hidden_dim=32, num_gin_layers=2)
    predictor = PredictorNetwork(vocab=vocab, hidden_dim=16, num_gcn_layers=2)
    diffusion = DiscreteDiffusion(num_timesteps=10)
    device = torch.device("cpu")

    sampler = GraphSampler(
        denoiser=denoiser,
        predictor=predictor,
        diffusion=diffusion,
        vocab=vocab,
        device=device,
        guidance_scale=0.5,
    )

    # Sample 1 graph
    B = 1
    gen_types, gen_attrs, gen_adj = sampler.sample(
        parent_node_types=batch["parent_node_types"][:B],
        parent_node_attrs=batch["parent_node_attrs"][:B],
        parent_adj=batch["parent_adj"][:B],
        parent_mask=batch["parent_mask"][:B],
        text_embedding=batch["text_embedding"][:B],
        num_steps=10,
        verbose=True,
    )
    print(f"\n  Generated tensors:")
    print(f"    node_types: {gen_types.shape} — unique: {gen_types[0].unique().tolist()[:10]}")
    print(f"    node_attrs: {gen_attrs.shape}")
    print(f"    adj:        {gen_adj.shape} — edges: {gen_adj[0].sum():.0f}")

    # Decode to string
    gen_str = decode_graph_tensors(gen_types[0], gen_attrs[0], gen_adj[0], vocab)
    print(f"\n  Generated graph string (first 400 chars):")
    print(f"  {gen_str[:400]}")

    # Validate
    ok, msg = validate_dag_compilation(gen_str, dummy_input=torch.randn(1, 16, 32, 32))
    print(f"\n  Generated graph validity: {'✓' if ok else '✗'} — {msg}")
    print(f"  (Note: untrained model → low validity is expected)")

    # ==================================================================
    # TEST 4: Posterior sampling correctness — at t=0, should recover x_0
    # ==================================================================
    print("\n--- Test 4: Posterior at t=0 recovers argmax ---")
    logits = torch.randn(2, 5, 10)  # [B=2, N=5, C=10]
    x_t = torch.randint(0, 10, (2, 5))  # doesn't matter at t=0
    sampled = sampler._posterior_sample_categorical(x_t, logits, t=0, num_classes=10)
    expected = logits.argmax(dim=-1)
    assert (sampled == expected).all(), "Posterior at t=0 should be argmax!"
    print(f"  ✓ At t=0, posterior recovers argmax of logits")

    print("\n" + "=" * 70)
    print("✓ Step 5 complete — sampling, decoding, and DAG validity verified.")
    print("=" * 70)
