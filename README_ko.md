# SDEdit을 활용한 텍스트 기반 신경망 구조 변이 (베이스라인)

## 개요
이 저장소는 **텍스트 기반 신경망 구조 설계(NAD)**를 위한 엄격하게 통제된 PyTorch 베이스라인을 제공합니다. 고정된 토폴로지(예: 8-노드 NAS-Bench-201)에 국한된 기존 탐색 작업과 달리, 본 프로젝트는 자연어 조건에 따라 **가변 길이** 부모 아키텍처를 최적의 자식 아키텍처로 변이시키는 것을 목표로 합니다. 사전 학습된 연속형 DiffusionNAG (CATE) 스코어 네트워크를 SDEdit 스타일의 파이프라인으로 확장하여, 백본 가중치 훼손 없이 텍스트 조건부 생성이 가능하도록 설계되었습니다.

## 핵심 아키텍처 및 엔지니어링 설계
생성 과정의 수학적 무결성을 보장하고 사전 학습된 그래프 분포의 치명적인 망각(Catastrophic Forgetting)을 방지하기 위해 다음 원칙을 엄격히 준수합니다:

1. **동결된 베이스 인코더 (`requires_grad=False`)**
   CATE 트랜스포머 아키텍처와 시맨틱 임베딩 전체를 동결했습니다. 이를 통해 모델은 강력한 구조적 사전 확률(Prior)로서 무조건부 그래프 생성 능력을 100% 유지합니다.
   
2. **전역 조건 주입 (Global Conditioning)**
   텍스트 임베딩은 잠재 공간 내에서 확산 시간 임베딩과 더해져 주입됩니다:
   $$e_{\text{total}} = e_{\text{node}} + e_{\text{time}} + \text{Text\_MLP}(e_{\text{text}})$$
   이 방식은 트랜스포머 블록 내부를 수정하지 않고도 텍스트 신호가 스코어 예측 벡터 필드에 전역적인 영향력을 행사할 수 있게 합니다.
   
3. **영점 초기화(Zero-Initialization) 전략**
   `Text_MLP` 어댑터의 마지막 선형 투영 레이어는 **영점 초기화**됩니다. 학습 초기(Step 0)에는 텍스트 조건의 수학적 영향력이 정확히 0이 되므로, 랜덤 노이즈 신호가 동결된 표현형을 망가뜨리는 "그래디언트 쇼크"를 완벽하게 방지합니다.
   
4. **데이터 흐름 분리 (Anti-Leakage)**
   *   **학습 단계**: **$X_{child}$에 대한 Denoising Score Matching**만 수행합니다. 텍스트 조건 하에 자식 아키텍처를 복원하는 법을 학습하며, $X_{parent}$는 학습 루프에서 완전히 제외됩니다.
   *   **추론 단계**: **SDEdit**을 사용하여 변이를 수행합니다. 주어진 $X_{parent}$에 변이 비율 $t_{edit}$만큼 노이즈를 더한 뒤, 텍스트 조건부 역방향 노이즈 제거 과정을 통해 자식 분포로 이동합니다.
   *   부모 구조를 학습에서 제외함으로써, 모델이 단순히 입력을 복사하는 지름길을 학습하는 것을 방지합니다.

## 동적 토폴로지 및 메모리 안전성
본 베이스라인은 NAD 트리플렛 데이터셋에 존재하는 실제 가변 길이 DAG를 처리하도록 설계되었습니다:

*   **가변 길이 지원**: 고정 셀 방식과 달리 최대 110개 노드까지 지원합니다.
*   **패딩 마스킹 (Padding Masking)**: CATE 어텐션 및 Loss 계산 시 동적 `padding_mask`를 적용합니다. 모델이 `<PAD>` 토큰을 참조하거나 해당 위치에서 학습 신호를 발생시키는 것을 차단합니다.
*   **동적 이산화 (Dynamic Discretization)**: 평가 시 `argmax`를 통해 시퀀스를 스캔하고 종료 노드(`[output]`)를 동적으로 찾습니다. 이를 통해 하드코딩 없이 가변 길이 파이프 문자열 변환이 가능합니다.

## 디렉토리 구조

```text
DiffusionNAG/
├── NAS-Bench-201/
│   ├── baseline_diffusion_nag.py    # 학습 스크립트 ($X_{child}$ 기반 Score Matching)
│   ├── evaluate_baseline.py         # 평가 스크립트 ($X_{parent}$ 기반 SDEdit)
│   ├── models/                      # CATE 모델 정의
│   ├── checkpoints/                 # 가중치 파일
│   │   ├── checkpoint.pth.tar       # [필수] 사전 학습된 동결 CATE 가중치
│   │   └── baseline_text_mlp.pth    # [자동 생성] 학습된 Text_MLP 어댑터 가중치
│   └── results/                     # [자동 생성] 결과 메트릭 및 split 파일
│       ├── dataset_splits.json      # [자동 생성] 통합 계층화 분할 파일
│       └── baseline_evaluation_metrics.json
├── NAD_triplet_dataset.jsonl        # [필수] 트리플렛 데이터셋 (Parent/Text/Child)
└── README.md
```

## 빠른 시작 및 재현성

공정한 비교를 위해 **통합 계층화 분할(Unified Stratified Split)**을 구현했습니다. `dataset_splits.json`에 저장된 인덱스는 모든 실험에서 동일하게 유지되어 데이터 누수(Leakage)를 0%로 보장합니다.

### 1. Text_MLP 어댑터 학습
```bash
python NAS-Bench-201/baseline_diffusion_nag.py \
  --jsonl_path NAD_triplet_dataset.jsonl \
  --max_node 110 \
  --batch_size 32 \
  --num_steps 1000
```

### 2. 엄격한 성능 평가
통합 분할 파일을 강제 적용하여, 학습 시 사용된 데이터를 정확히 제외하고 성능을 측정합니다.
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

## 평가 지표 (Metrics)
모든 지표는 엄격한 단계별 필터를 통해 계산됩니다:
1.  **유효성 (Validity)**: 인덱스 0의 `[input]`과 유일한 `[output]` 존재 여부를 확인하는 동적 토폴로지 검사.
2.  **고유성 (Uniqueness)**: 유효한 결과 중 수학적으로 서로 다른 아키텍처의 비율.
3.  **참신성 (Novelty)**: 학습 세트에 존재하지 않는 새로운 아키텍처의 비율 (학습 시 실제 사용된 텐서 해싱 비교).
## 인용 (Citation)
본 베이스라인이나 사전 학습된 DiffusionNAG (CATE) 가중치가 연구에 도움이 되었다면, 원본 논문을 인용해주세요:

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
