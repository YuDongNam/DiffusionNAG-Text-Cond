# Text-Guided Neural Architecture Mutation via SDEdit (Baseline)

## Abstract
This repository provides a self-contained, strictly controlled PyTorch baseline for **Text-Guided Neural Architecture Design (NAD)**. Unlike standard search tasks limited to fixed topological bounds (e.g., 8-node NAS-Bench-201), our goal is to mutate **variable-length** parent architectures into specialized children based on natural language conditions. It adapts the pre-trained continuous DiffusionNAG (CATE) score network into an SDEdit-style pipeline without re-training or corrupting the base backbone.

## Key Architectural & Engineering Choices
To ensure the mathematical validity of the generation process and prevent catastrophic forgetting of the pre-trained graph distributions, we enforce strict structural decisions:

1. **Frozen Base Encoder (`requires_grad=False`)**
   The entire CATE Transformer architecture and its semantic embeddings are strictly frozen. The model retains 100% of its unconditional generating capabilities, serving as a powerful structural prior.
   
2. **Global Conditioning Injection**
   Text embeddings are injected into the latent space as an additive component to the diffusion timestep embedding:
   $$e_{\text{total}} = e_{\text{node}} + e_{\text{time}} + \text{Text\_MLP}(e_{\text{text}})$$
   This approach allows the text signal to globally influence the denoising vector field without modifying the internal Transformer blocks.
   
3. **Zero-Initialization Strategy**
   The final linear projection layer of the `Text_MLP` adapter is formally **zero-initialized**. At step zero of training, the text condition exerts exactly $0$ mathematical influence on the score prediction. This prevents "gradient shock" (random noise signals) from corrupting the frozen representations during early optimization phases.
   
4. **Data Flow Separation (Anti-Leakage)**
   *   **Training Phase**: The model is trained exclusively using **Denoising Score Matching on $X_{child}$**. It learns to reconstruct the target child architecture from noise, conditioned on text. 
   *   **Inference Phase**: Mutation is achieved via **SDEdit**. We start from a known $X_{parent}$, add noise up to a ratio $t_{edit}$, and then reverse-denoise towards the child distribution using the text condition.
   *   $X_{parent}$ is strictly excluded from the training loop to prevent the model from simply learning a "copy-paste" shortcut.

## Dynamic Topology & Memory Safety
The baseline is engineered to handle real-world variable-length DAGs found in the NAD triplet dataset:

*   **Variable-Length Support**: Unlike fixed-cell NAS, we support graphs up to 110 nodes.
*   **Padding Masking**: A strict dynamic `padding_mask` is applied to both the CATE attention mechanism and the Loss calculation. This ensures the model never attends to or generates signals for `<PAD>` tokens.
*   **Dynamic Discretization**: During evaluation, the output sequence is scanned dynamically via `argmax` to locate the terminating `[output]` node. This allows for variable-length pipe-string representation without hardcoding indices.

## Directory Structure

```text
DiffusionNAG/
├── NAS-Bench-201/
│   ├── baseline_diffusion_nag.py    # Training script (Score Matching on X_child)
│   ├── evaluate_baseline.py         # Evaluation script (SDEdit from X_parent)
│   ├── models/                      # CATE model definitions
│   ├── checkpoints/                 # Model Weights
│   │   ├── checkpoint.pth.tar       # [REQUIRED] Pre-trained frozen CATE weights
│   │   └── baseline_text_mlp.pth    # [GENERATED] Trained Text_MLP adapter weights
│   └── results/                     # [GENERATED] Metrics & split files
│       ├── dataset_splits.json      # [GENERATED] Unified stratified split file
│       └── baseline_evaluation_metrics.json
├── NAD_triplet_dataset.jsonl        # [REQUIRED] Triplet dataset (Parent/Text/Child)
└── README.md
```

## Quick Start & Reproducibility

To ensure fair comparison, we implement a **Unified Stratified Split**. The split indices (stratified by the `dataset` label) are locked in `dataset_splits.json` to guarantee 0% data leakage across different runs.

### 1. Training the Text_MLP Adapter
```bash
python NAS-Bench-201/baseline_diffusion_nag.py \
  --jsonl_path NAD_triplet_dataset.jsonl \
  --max_node 110 \
  --batch_size 32 \
  --num_steps 1000
```

### 2. Rigorous Evaluation
This command enforces the unified split file to ensure the `seen_graphs_set` exactly matches the ground truth training split.
```bash
python NAS-Bench-201/evaluate_baseline.py \
  --mode full \
  --jsonl_path NAD_triplet_dataset.jsonl \
  --max_node 110 \
  --split_path results/dataset_splits.json \
  --t_edit 0.5 \
  --eval_split test \
  --device cuda
```

## Evaluation Metrics
Metrics are calculated using a strict cascading funnel:
1.  **Validity**: Dynamic topological checker confirming exactly one `[input]` (index 0) and one `[output]`.
2.  **Uniqueness**: Ratio of valid architectures that are mathematically distinct.
3.  **Novelty**: Ratio of unique architectures that do *not* exist in the training split (checked via hashed tensors from actual `train` indices).

## Citation
If you find this baseline or the pre-trained DiffusionNAG (CATE) weights helpful, please cite the original paper:

```bibtex
@inproceedings{
  nam2024diffusionnag,
  title={Diffusion{NAG}: Predictor-guided Neural Architecture Generation with Diffusion Models},
  author={Yudong Nam and Seokhyun Moon and Kyungsu Kim},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=9G3x4PZw7X}
}
```
