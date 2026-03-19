"""
evaluate.py
===========
Standalone evaluation script for the Conditional Graph-to-Graph Translation baseline.
Runs reverse diffusion over the validation/test set and computes quantitative metrics:
1. Validity (%): Is the generated DAG compilable and executable?
2. Text-Graph Alignment: Cosine similarity between generated graph and target text using Predictor.
3. Modification Rate (%): Did the model actually modify the parent graph?

Outputs are saved to `results.json`.
"""
import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add graph2graph to sys.path so inner modules can import each other
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph2graph"))

from step1_dataset import create_dataloaders, encode_graph
from step2_denoising_gnn import DenoisingGNN
from step3_predictor import PredictorNetwork
from step4_training import DiscreteDiffusion
from step5_inference import GraphSampler, decode_graph_tensors, validate_dag_compilation


def evaluate(
    jsonl_path: str,
    ckpt_path: str,
    text_embed_path: str = None,
    batch_size: int = 16,
    num_timesteps: int = 1000,
    guidance_scale: float = 1.0,
    device_str: str = "auto",
):
    # Device setup
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")

    # Load Text Embeddings
    text_embed_dict = None
    if text_embed_path and os.path.exists(text_embed_path):
        text_embed_dict = torch.load(text_embed_path, map_location="cpu")
        print(f"Loaded {len(text_embed_dict)} real text embeddings.")

    # Data Loader (Validation Split)
    _, val_loader, vocab = create_dataloaders(
        jsonl_path=jsonl_path,
        batch_size=batch_size,
        train_ratio=0.9,
        text_embed_dict=text_embed_dict,
        num_workers=4,
        max_samples=None,  # Must use full dataset to rebuild full vocab
    )
    print(f"Validating on {len(val_loader.dataset)} samples ({len(val_loader)} batches).")

    # Load Models
    hidden_dim = 256  # Default training dim, adapt if your checkpoint differs
    # Try to infer hidden_dim from checkpoint structure if possible, but we'll assume 256
    
    denoiser = DenoisingGNN(
        vocab=vocab, hidden_dim=hidden_dim, num_gin_layers=4, dropout=0.0
    ).to(device)
    
    predictor = PredictorNetwork(
        vocab=vocab, hidden_dim=hidden_dim // 2, num_gcn_layers=3, dropout=0.0
    ).to(device)
    
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        try:
            denoiser.load_state_dict(ckpt["denoiser"])
            predictor.load_state_dict(ckpt["predictor"])
        except RuntimeError as e:
            # Fallback if checkpoint was trained with hidden_dim=64 (like test_run.py)
            print("Shape mismatch detected, trying hidden_dim=64 (test_run size)...")
            hidden_dim = 64
            denoiser = DenoisingGNN(vocab=vocab, hidden_dim=hidden_dim, num_gin_layers=4).to(device)
            predictor = PredictorNetwork(vocab=vocab, hidden_dim=hidden_dim // 2, num_gcn_layers=3).to(device)
            denoiser.load_state_dict(ckpt["denoiser"])
            predictor.load_state_dict(ckpt["predictor"])
    else:
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Train the model first.")

    denoiser.eval()
    predictor.eval()
    
    diffusion = DiscreteDiffusion(num_timesteps=1000)
    sampler = GraphSampler(denoiser, predictor, diffusion, vocab, device, guidance_scale=guidance_scale)

    # Metrics
    total_samples = 0
    valid_count = 0
    modified_count = 0
    total_cosine_sim = 0.0

    print("\nStarting evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader)):
            B = batch["parent_node_types"].size(0)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Extract Parent Graph Strings (for modification rate check)
            parent_strings = []
            for b in range(B):
                p_str = decode_graph_tensors(
                    batch["parent_node_types"][b],
                    batch["parent_node_attrs"][b],
                    batch["parent_adj"][b],
                    vocab,
                    block_name="ParentBlock"
                )
                parent_strings.append(p_str)

            # 1. Generate Child DAGs
            # We must enable requires_grad inside sampler for guidance, so we briefly allow grad
            with torch.enable_grad():
                gen_types, gen_attrs, gen_adj = sampler.sample(
                    parent_node_types=batch["parent_node_types"],
                    parent_node_attrs=batch["parent_node_attrs"],
                    parent_adj=batch["parent_adj"],
                    parent_mask=batch["parent_mask"],
                    text_embedding=batch["text_embedding"],
                    num_steps=num_timesteps,
                    verbose=False,
                )
            
            # Generate padding masks for the generated graphs
            empty_idx = vocab.node_type_to_idx["<EMPTY>"]
            gen_mask = (gen_types != empty_idx).float()  # [B, N]

            # 2. Text-Graph Alignment (Cosine Similarity in HARD mode)
            # The predictor requires the parent graph AND the child graph (hard mode)
            align_sim = predictor(
                parent_node_types=batch["parent_node_types"],
                parent_node_attrs=batch["parent_node_attrs"],
                parent_adj=batch["parent_adj"],
                parent_mask=batch["parent_mask"],
                child_node_types=gen_types,
                child_node_attrs=gen_attrs,
                child_adj=gen_adj,
                child_mask=gen_mask,
                text_embedding=batch["text_embedding"]
            ).squeeze(1)  # [B]
            total_cosine_sim += align_sim.sum().item()

            # 3. Validity & Modification Rate
            dummy_input = torch.randn(1, 16, 32, 32)
            for b in range(B):
                gen_str = decode_graph_tensors(
                    gen_types[b], gen_attrs[b], gen_adj[b], vocab, block_name="GenBlock"
                )
                
                # Check Validity
                is_valid, _ = validate_dag_compilation(gen_str, dummy_input)
                if is_valid:
                    valid_count += 1
                
                # Check Modification
                # Extract the core architecture part (ignore the ##BlockName## header)
                p_core = "\n".join(parent_strings[b].split("\n")[1:])
                g_core = "\n".join(gen_str.split("\n")[1:])
                if p_core != g_core:
                    modified_count += 1
            
            total_samples += B

    # Aggregate Metrics
    metrics = {
        "Total_Samples": total_samples,
        "Validity_Rate": float(valid_count) / total_samples,
        "Text_Graph_Alignment": total_cosine_sim / total_samples,
        "Modification_Rate": float(modified_count) / total_samples,
        "Num_Timesteps": num_timesteps,
        "Guidance_Scale": guidance_scale
    }

    print("\n" + "="*50)
    print("🎯 Evaluation Results")
    print("="*50)
    print(f"Total Samples Tested : {metrics['Total_Samples']}")
    print(f"Validing Rate (%)    : {metrics['Validity_Rate'] * 100:.2f}%")
    print(f"Modification Rate (%): {metrics['Modification_Rate'] * 100:.2f}%")
    print(f"Text-Graph Alignment : {metrics['Text_Graph_Alignment']:.4f}")
    print("="*50)

    with open("results.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Saved metrics to results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DiffusionNAG Baseline")
    parser.add_argument("--jsonl_path", type=str, default="NAD_triplet_dataset.jsonl")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/best.pt")
    parser.add_argument("--text_embed_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    evaluate(
        jsonl_path=args.jsonl_path,
        ckpt_path=args.ckpt_path,
        text_embed_path=args.text_embed_path,
        batch_size=args.batch_size,
        num_timesteps=args.num_timesteps,
        guidance_scale=args.guidance_scale,
        device_str=args.device,
    )
