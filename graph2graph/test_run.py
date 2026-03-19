from step4_training import train

print("🚀 로컬 CPU Dry-run 테스트를 시작합니다...")

try:
    train(
        jsonl_path="NAD_triplet_dataset.jsonl",
        hidden_dim=64,           # CPU 연산 속도를 위해 256 -> 64로 대폭 축소
        batch_size=4,            # RAM 낭비를 막고 빠른 스텝 업데이트를 위해 축소
        num_epochs=2,            # 파이프라인이 1바퀴 돌고 2에폭으로 넘어가는지만 확인
        lr_denoiser=1e-4,
        lr_predictor=3e-4,
        num_timesteps=1000,      # 이건 Diffusion 마르코프 체인 수학을 위해 그대로 유지
        device_str="cpu",        # 명시적으로 CPU 강제 할당
    )
    print("✅ 테스트 완료: CPU 환경에서 텐서 충돌 없이 학습 루프가 정상 작동합니다.")
except Exception as e:
    print(f"❌ 에러 발생: {e}")