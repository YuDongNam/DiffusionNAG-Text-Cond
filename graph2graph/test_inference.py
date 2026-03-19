"""
추론 테스트 스크립트
===================
학습된 체크포인트 또는 즉석 학습 모델로 역방향 Diffusion 샘플링 + DAG 유효성 검사 실행.
CPU에서 동작하도록 최적화 (timesteps=50으로 축소).
"""
import os
import json
import torch

from step1_dataset import (
    N_MAX, TEXT_EMBED_DIM, GraphVocabulary,
    NADTripletDataset, create_dataloaders,
)
from step2_denoising_gnn import DenoisingGNN
from step3_predictor import PredictorNetwork
from step4_training import DiscreteDiffusion
from step5_inference import GraphSampler, decode_graph_tensors, validate_dag_compilation

JSONL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "NAD_triplet_dataset.jsonl",
)

# 체크포인트 경로: train()의 save_dir="checkpoints"는 CWD 기준이므로
# 여러 후보 경로를 탐색한다
_here = os.path.dirname(os.path.abspath(__file__))
CKPT_CANDIDATES = [
    os.path.join(_here, "checkpoints", "best.pt"),          # graph2graph/checkpoints/
    os.path.join(_here, "..", "checkpoints", "best.pt"),    # DiffusionNAG/checkpoints/
    os.path.join(os.getcwd(), "checkpoints", "best.pt"),    # CWD/checkpoints/
]

device = torch.device("cpu")
HIDDEN_DIM = 64          # test_run.py와 동일
NUM_SAMPLE_STEPS = 50    # CPU 속도를 위해 1000 → 50으로 축소
GUIDANCE_SCALE = 1.0

print("=" * 70)
print("🔮 추론 테스트: Reverse Diffusion + DAG Validity")
print("=" * 70)

# ── 1. 체크포인트 탐색 ──────────────────────────────────────────────
CKPT_PATH = None
for cand in CKPT_CANDIDATES:
    if os.path.exists(cand):
        CKPT_PATH = cand
        break

# ── 2. Vocab 구축 (전체 데이터 기준 — 체크포인트와 동일한 차원 보장) ──
# 반드시 max_samples=None 으로 전체 데이터를 읽어야
# 학습 시 만든 vocab(27 types)과 동일한 임베딩 크기를 얻는다.
# max_samples=20 같이 축소하면 vocab이 18 types가 되어
# load_state_dict 시 shape mismatch 발생!
train_loader, val_loader, vocab = create_dataloaders(
    jsonl_path=JSONL_PATH, batch_size=4, num_workers=0,
    max_samples=None,   # ← 전체 데이터로 vocab 구축 (CRITICAL)
)
print(f"Vocab: {vocab.num_node_types} node types, {len(vocab.ATTR_KEYS)} attr keys")

# ── 3. 모델 생성 ────────────────────────────────────────────────────
denoiser = DenoisingGNN(
    vocab=vocab, hidden_dim=HIDDEN_DIM, num_gin_layers=4,
    dropout=0.3, max_timesteps=1000,
).to(device)

predictor = PredictorNetwork(
    vocab=vocab, hidden_dim=HIDDEN_DIM // 2, num_gcn_layers=3, dropout=0.2,
).to(device)

# ── 4. 체크포인트 로드 ───────────────────────────────────────────────
if CKPT_PATH is not None:
    print(f"\n✓ 체크포인트 발견: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device)
    denoiser.load_state_dict(ckpt["denoiser"])
    predictor.load_state_dict(ckpt["predictor"])
    print(f"  Epoch {ckpt.get('epoch', '?')}, Val Loss: {ckpt.get('val_loss', '?')}")
else:
    print(f"\n⚠ 체크포인트 없음 (탐색 경로: {CKPT_CANDIDATES})")
    print("  학습 안 된 모델로 추론합니다 (유효하지 않은 결과가 예상됩니다)")

denoiser.eval()
predictor.eval()

# ── 4. 테스트 배치 준비 ──────────────────────────────────────────────
batch = next(iter(val_loader))
batch = {k: v.to(device) for k, v in batch.items()}
B = 2  # 2개 그래프만 생성

print(f"\n--- 입력: Parent DAG (B={B}) ---")
for b in range(B):
    n_real = int(batch["parent_mask"][b].sum().item())
    n_edges = int(batch["parent_adj"][b].sum().item())
    print(f"  Sample {b}: {n_real} real nodes, {n_edges} edges")

# ── 5. 역방향 Diffusion 샘플링 ──────────────────────────────────────
diffusion = DiscreteDiffusion(num_timesteps=1000)

sampler = GraphSampler(
    denoiser=denoiser,
    predictor=predictor,
    diffusion=diffusion,
    vocab=vocab,
    device=device,
    guidance_scale=GUIDANCE_SCALE,
)

print(f"\n--- 역방향 Diffusion 샘플링 (steps={NUM_SAMPLE_STEPS}, λ={GUIDANCE_SCALE}) ---")

# timesteps를 1000에서 50으로 서브샘플링 (등간격)
gen_types, gen_attrs, gen_adj = sampler.sample(
    parent_node_types=batch["parent_node_types"][:B],
    parent_node_attrs=batch["parent_node_attrs"][:B],
    parent_adj=batch["parent_adj"][:B],
    parent_mask=batch["parent_mask"][:B],
    text_embedding=batch["text_embedding"][:B],
    num_steps=NUM_SAMPLE_STEPS,
    verbose=True,
)

# ── 6. 생성된 텐서 → 그래프 문자열 변환 ─────────────────────────────
print("\n--- 생성된 그래프 ---")
for b in range(B):
    graph_str = decode_graph_tensors(
        gen_types[b], gen_attrs[b], gen_adj[b], vocab,
        block_name=f"Generated_Sample_{b}",
    )

    # 비어있지 않은 (non-EMPTY) 노드 수
    empty_idx = vocab.node_type_to_idx["<EMPTY>"]
    n_real = int((gen_types[b] != empty_idx).sum().item())
    n_edges = int(gen_adj[b].sum().item())

    print(f"\n  [Sample {b}] {n_real} real nodes, {n_edges} edges")
    # 처음 500자만 출력
    for line in graph_str.split("\n")[:25]:
        print(f"    {line}")
    if len(graph_str.split("\n")) > 25:
        print(f"    ... ({len(graph_str.split(chr(10)))} lines total)")

    # ── 7. DAG 유효성 검사 ───────────────────────────────────────────
    ok, msg = validate_dag_compilation(
        graph_str,
        dummy_input=torch.randn(1, 16, 32, 32),
        verbose=False,
    )
    status = "✓ Valid" if ok else "✗ Invalid"
    print(f"  DAG Validity: {status} — {msg}")

# ── 8. 비교: Ground-Truth Child Graph도 검증 ────────────────────────
print("\n--- 비교: Ground-Truth Child DAG ---")
with open(JSONL_PATH) as f:
    gt_samples = [json.loads(line) for line in f][:B]

for b, sample in enumerate(gt_samples):
    ok, msg = validate_dag_compilation(
        sample["child_graph"],
        dummy_input=torch.randn(1, 16, 32, 32),
    )
    n_nodes = sample["child_graph"].count(":")
    print(f"  [GT Sample {b}] ~{n_nodes} nodes → {'✓ Valid' if ok else '✗ Invalid'} — {msg}")

print("\n" + "=" * 70)
print("🏁 추론 테스트 완료!")
print("=" * 70)
