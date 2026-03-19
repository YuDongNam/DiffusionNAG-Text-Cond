# 텍스트 조건부 그래프-투-그래프 번역 베이스라인

**DAG-to-DAG 번역**을 위한 이산 확산 베이스라인입니다. 부모 신경망 아키텍처 그래프를 자연어 텍스트 임베딩의 안내를 받아 자식 아키텍처 그래프로 변환합니다. [DiffusionNAG](https://arxiv.org/abs/2305.16943)의 개념을 기반으로 하며, 연속 확산 대신 이산 노드 타입·속성·엣지에 대한 **D3PM 범주형 확산** 프로세스를 사용합니다.

## 아키텍처

```
부모 DAG  ──┐
텍스트 임베딩 ─┤──▶ DenoisingGNN (4층 GIN) ──▶ 예측된 클린 G₀
노이즈 G_t  ──┤     ├─ 노드 타입 로짓   [N × 27]
타임스텝 t  ──┘     ├─ 속성 로짓       [N × K × C_k]
                    └─ 엣지 로짓       [N × N × 2]
                              │
                    ┌─────────▼──────────┐
                    │ PredictorNetwork   │
                    │ (3층 GCN)          │
                    │ cos_sim(그래프,텍스트)│
                    └─────────┬──────────┘
                              │ ∇ 가이던스
                              ▼
                      가이드된 사후분포
                      q(x_{t-1}|x_t, x̂₀)
```

## 파일 구조

```
graph2graph/
├── step1_dataset.py        # 데이터 파싱, GraphVocabulary, N_MAX=110 패딩
├── step2_denoising_gnn.py  # 3개 출력 헤드를 가진 4층 GIN 디노이저
├── step3_predictor.py      # 코사인 유사도 기반 분류기 가이던스 (소프트/하드 모드)
├── step4_training.py       # 이산 확산, 마스크된 손실함수, 학습 루프
├── step5_inference.py      # 역방향 샘플링, 그래프 디코딩, DAG 유효성 검증기
├── evaluate.py             # 정량적 평가 (유효성, 정렬도, 수정 비율)
├── test_run.py             # CPU 드라이런 학습 테스트
└── test_inference.py       # 체크포인트 로드 → 추론 end-to-end 테스트
```

| 모듈 | 핵심 설계 결정 |
|---|---|
| **step1** | 데이터셋 스캔으로 `N_MAX=110` 결정; `<EMPTY>`/`<NONE>` 특수 토큰; sample_id 키 기반 임베딩 딕셔너리 |
| **step2** | 모든 입력의 조기 연결(early concat); 순열 등변(permutation-equivariant) GIN 레이어 |
| **step3** | `SoftNodeEmbedding`으로 가이던스 시 로짓 → 프레딕터 간 그래디언트 흐름 보장 |
| **step4** | 코사인 노이즈 스케줄; `child_mask`로 CE 손실을 마스킹 후 평균 (패딩 그래디언트 0 보장) |
| **step5** | D3PM 범주형 사후분포; `autograd.grad(allow_unused=True)`; 25종+ 연산 동적 DAG 컴파일러 |

## 데이터 준비

### 데이터셋

학습 데이터는 JSONL 파일 (`NAD_triplet_dataset.jsonl`)이며, 각 줄은 다음을 포함합니다:

```json
{
  "sample_id": "cifar10_s000000_0cdeeba3",
  "parent_graph": "##ParentBlock##\n0:input\n1:output\n...",
  "child_graph": "##ChildBlock##\n0:input\n1:output\n...",
  "text": "spatial feature enhancement module"
}
```

### 텍스트 임베딩 (핵심 사항)

텍스트 임베딩은 반드시 `sample_id` → `Tensor[768]` 매핑의 PyTorch 딕셔너리로 저장해야 합니다:

```python
import torch

embed_dict = {
    "cifar10_s000000_0cdeeba3": torch.tensor([0.0123, -0.0456, ...]),  # 768차원
    "cifar10_s000001_61df0aad": torch.tensor([0.0789, 0.0012, ...]),
    # ... 샘플당 하나씩
}
torch.save(embed_dict, "baidu_text_embeddings.pt")
```

> **⚠️ 경고:** 딕셔너리 키는 반드시 `sample_id` (예: `"cifar10_s000000_0cdeeba3"`)여야 하며, **원본 텍스트 문자열을 키로 사용하면 안 됩니다.** 동일한 텍스트 설명을 공유하는 샘플이 존재하므로, 텍스트를 키로 사용하면 충돌(collision)이 발생하여 데이터가 조용히 오염됩니다.

**설계 근거:** `sample_id` 키는 `__getitem__` 내에서 그래프와 동일한 JSONL 행에서 읽히므로, DataLoader의 셔플링, 배치 순서, 멀티워커 병렬 처리와 무관하게 임베딩이 항상 올바른 그래프와 짝을 이룹니다.

`.pt` 파일이 제공되지 않으면 결정론적 더미 임베딩(해시 시드 랜덤 벡터)으로 폴백합니다. 파이프라인 테스트에는 유용하지만 의미 정보는 포함하지 않습니다.

## 빠른 시작

### 학습

```bash
cd DiffusionNAG

# 전체 학습 (GPU 권장)
python graph2graph/step4_training.py  # 기본값: hidden_dim=256, epochs=100

# 실제 텍스트 임베딩 사용
python -c "
from graph2graph.step4_training import train
train(
    jsonl_path='NAD_triplet_dataset.jsonl',
    text_embed_path='baidu_text_embeddings.pt',  # 실제 임베딩
    hidden_dim=256,
    batch_size=32,
    num_epochs=100,
    device_str='auto',
)
"

```

## 추론 및 평가

검증 데이터셋에 대해 모델을 정량적으로 평가하려면 `evaluate.py` 스크립트를 실행하십시오. 이 스크립트는 유효성 비율(Validity Rate), 텍스트-그래프 정렬도(Text-Graph Alignment), 그리고 수정 비율(Modification Rate)을 계산합니다:

```bash
python evaluate.py --batch_size 16 --num_timesteps 1000 --seed 42
```

실제 추론은 `step5_inference.py`에서 `GraphSampler`를 임포트하여 프로그래밍 방식으로도 수행할 수 있습니다. 샘플러는 학습된 디노이저와 예측기 모델을 사용하여 역방향 확산 프로세스를 실행하고 분류기 가이던스를 적용합니다.

```python
from graph2graph.step5_inference import GraphSampler, decode_graph_tensors, validate_dag_compilation

# 학습된 모델로 샘플러 초기화
sampler = GraphSampler(denoiser, predictor, diffusion, vocab, device, guidance_scale=1.0)

# 부모 DAG + 텍스트 임베딩 → 자식 DAG 생성
gen_types, gen_attrs, gen_adj = sampler.sample(
    parent_node_types, parent_node_attrs, parent_adj, parent_mask,
    text_embedding, num_steps=100,
)

# 디코딩 및 유효성 검증
graph_str = decode_graph_tensors(gen_types[0], gen_attrs[0], gen_adj[0], vocab)
is_valid, msg = validate_dag_compilation(graph_str, dummy_input=torch.randn(1, 64, 32, 32))
```

## 테스트 및 드라이런

저장소에는 CPU에서 파이프라인 로직을 검증하기 위한 두 가지 테스트 스크립트가 포함되어 있습니다. 이 스크립트들은 극단적으로 줄어든 타임스텝, 초소형 모델 사이즈, 그리고 더미 데이터를 사용합니다. **실제 학습이나 추론에 이 스크립트들을 사용하지 마십시오.**

```bash
# 학습 파이프라인 검증 (텐서 형태 오류 없이 컴파일되는지/실행되는지 확인)
python graph2graph/test_run.py

# 추론 파이프라인 검증 (역방향 확산 50 스텝 실행 및 DAG 유효성 테스트)
python graph2graph/test_inference.py
```

## 요구 사항

- Python 3.9+
- PyTorch 2.0+
- NumPy

추가 의존성은 필요하지 않습니다. 모든 그래프 연산은 순수 PyTorch로 구현되어 있습니다.

## 라이선스

루트 저장소의 라이선스를 참조하세요.
