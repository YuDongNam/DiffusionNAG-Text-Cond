# Text-Conditioned Graph-to-Graph Translation Baseline

A discrete diffusion baseline for **DAG-to-DAG translation**, where a parent neural architecture graph is transformed into a child architecture graph guided by a natural-language text embedding. Built on concepts from [DiffusionNAG](https://arxiv.org/abs/2305.16943), this implementation replaces continuous diffusion with a **D3PM-style categorical diffusion** process over discrete node types, attributes, and edges.

## Architecture

```
Parent DAG ──┐
Text Embed ──┤──▶ DenoisingGNN (4-layer GIN) ──▶ Predicted Clean G₀
Noisy G_t  ──┤     ├─ Node Type logits   [N × 27]
Timestep t ──┘     ├─ Attribute logits    [N × K × C_k]
                   └─ Edge logits         [N × N × 2]
                              │
                    ┌─────────▼──────────┐
                    │ PredictorNetwork   │
                    │ (3-layer GCN)      │
                    │ cos_sim(graph,text) │
                    └─────────┬──────────┘
                              │ ∇ guidance
                              ▼
                      Guided Posterior
                      q(x_{t-1}|x_t, x̂₀)
```

## File Structure

```
graph2graph/
├── step1_dataset.py        # Data parsing, GraphVocabulary, padding to N_MAX=110
├── step2_denoising_gnn.py  # 4-layer GIN denoiser with 3 output heads
├── step3_predictor.py      # Classifier guidance via cosine similarity (soft/hard)
├── step4_training.py       # Discrete diffusion, masked losses, training loop
├── step5_inference.py      # Reverse sampling, graph decoding, DAG validity checker
├── evaluate.py             # Quantitative evaluation (Validity, Alignment, Modification)
├── test_run.py             # CPU dry-run training test
└── test_inference.py       # End-to-end inference test with checkpoint loading
```

| Module | Key Design Decision |
|---|---|
| **step1** | `N_MAX=110` determined by dataset scan; `<EMPTY>`/`<NONE>` special tokens; sample_id–keyed embedding dict |
| **step2** | Early concatenation of all inputs; GIN layers for permutation-equivariant message passing |
| **step3** | `SoftNodeEmbedding` enables gradient flow from predictor back through logits during guidance |
| **step4** | Cosine noise schedule; CE losses masked by `child_mask` before averaging (zero gradient from padding) |
| **step5** | D3PM categorical posterior; `autograd.grad` with `allow_unused=True`; 25+ op dynamic DAG compiler |

## Data Preparation

### Dataset

The training data is a JSONL file (`NAD_triplet_dataset.jsonl`) where each line contains:

```json
{
  "sample_id": "cifar10_s000000_0cdeeba3",
  "parent_graph": "##ParentBlock##\n0:input\n1:output\n...",
  "child_graph": "##ChildBlock##\n0:input\n1:output\n...",
  "text": "spatial feature enhancement module"
}
```

### Text Embeddings (CRITICAL)

Text embeddings **must** be saved as a PyTorch dictionary mapping `sample_id` → `Tensor[768]`:

```python
import torch

embed_dict = {
    "cifar10_s000000_0cdeeba3": torch.tensor([0.0123, -0.0456, ...]),  # 768-dim
    "cifar10_s000001_61df0aad": torch.tensor([0.0789, 0.0012, ...]),
    # ... one entry per sample
}
torch.save(embed_dict, "baidu_text_embeddings.pt")
```

> **⚠️ WARNING:** The dictionary **must** be keyed by `sample_id` (e.g., `"cifar10_s000000_0cdeeba3"`), **not** by raw text strings. Multiple samples can share the same text description, so using text as keys would cause collisions and silent data corruption.

**Why this design?** The `sample_id` key is read from the same JSONL row as the graphs inside `__getitem__`, so the embedding is always paired with the correct graph regardless of DataLoader shuffling, batching order, or multi-worker parallelism.

If no `.pt` file is provided, the system falls back to deterministic dummy embeddings (hash-seeded random vectors). This is useful for pipeline testing but carries no semantic information.

## Quick Start

### Training

```bash
cd DiffusionNAG

# Full training (GPU recommended)
python graph2graph/step4_training.py  # uses defaults: hidden_dim=256, epochs=100

# With real text embeddings
python -c "
from graph2graph.step4_training import train
train(
    jsonl_path='NAD_triplet_dataset.jsonl',
    text_embed_path='baidu_text_embeddings.pt',  # real embeddings
    hidden_dim=256,
    batch_size=32,
    num_epochs=100,
    device_str='auto',
)
```

## Inference & Evaluation

To evaluate the model quantitatively over the validation set, run the `evaluate.py` script. This script computes the Validity Rate, Text-Graph Alignment (Cosine Similarity), and Modification Rate:

```bash
python evaluate.py --batch_size 16 --num_timesteps 1000
```

Actual inference can also be performed programmatically by importing the `GraphSampler` from `step5_inference.py`. The sampler runs the reverse diffusion process and applies classifier guidance using your trained denoiser and predictor models.

```python
from graph2graph.step5_inference import GraphSampler, decode_graph_tensors, validate_dag_compilation

# Initialize sampler with trained models
sampler = GraphSampler(denoiser, predictor, diffusion, vocab, device, guidance_scale=1.0)

# Generate child DAG from parent DAG + text embedding
gen_types, gen_attrs, gen_adj = sampler.sample(
    parent_node_types, parent_node_attrs, parent_adj, parent_mask,
    text_embedding, num_steps=100,
)

# Decode and validate
graph_str = decode_graph_tensors(gen_types[0], gen_attrs[0], gen_adj[0], vocab)
is_valid, msg = validate_dag_compilation(graph_str, dummy_input=torch.randn(1, 64, 32, 32))
```

## Testing & Dry Runs

The repository includes two testing scripts for verifying the pipeline logic on a CPU. These scripts use artificially reduced timesteps, miniature models, and dummy data. **Do not use these for actual training or inference.**

```bash
# Verify the training pipeline (compiles without tensor shape errors)
python graph2graph/test_run.py

# Verify the inference pipeline (runs 50 steps of reverse diffusion and tests DAG validity)
python graph2graph/test_inference.py
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- NumPy

No additional dependencies are required. All graph operations are implemented in pure PyTorch.

## License

See the root repository license.
