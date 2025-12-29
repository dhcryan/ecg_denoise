# Response to reviewer comment: missing diffusion/GAN/modern Transformer baselines

## Reviewer comment
> The baseline set misses recent diffusion-based denoisers, modern GANs, and newer Transformer variants. For a SOTA claim, either include a couple of representative contenders or temper the claim.

## Our response (English)
Thank you for pointing this out — we agree that our original baseline set does not fully reflect the newest generation of ECG denoisers/restorers, especially diffusion/score-based methods and modern GAN variants. In the revision, we will **avoid over-claiming “SOTA” beyond the compared baseline scope** and will strengthen the empirical comparison by adding representative recent contenders.

Specifically:
1. **Tempered claim / clarified scope.** We revised the wording to state that our method is *competitive within the set of evaluated baselines*, rather than claiming universal state-of-the-art across all recent generative denoisers.
2. **Added representative modern contenders (in progress / to be included in the updated experiments).** We are integrating the following three publicly available methods as additional baselines under the same dataset/noise protocol and evaluation metrics:
   - Operational CycleGAN-based blind ECG restoration: https://github.com/OzerCanDevecioglu/Blind-ECG-Restoration-by-Operational-Cycle-GANs.git
   - Score-based (diffusion) ECG denoising: https://github.com/HuayuLiArizona/Score-based-ECG-Denoising.git
   - Fully-gated DAE: https://github.com/AhmedAShaheen/fully_gated_DAE.git

To ensure fairness, we will:
- Use the **same train/test split** and **noise composition/SNR settings** used in our main experiments.
- Report the **same quantitative metrics** (e.g., RMSE/PRD/COS-SIM/SNR and task-driven metrics where applicable).
- Match **input length / sampling rate / preprocessing** as closely as possible to the original implementations, and clearly document any unavoidable differences.

## 저자 답변 (Korean)
지적 감사합니다. 말씀하신 대로 기존 baseline 구성은 최근의 **diffusion(=score-based) 기반 ECG denoising**, **현대 GAN 기반 restoration**, 그리고 **최신 DAE 변형**을 충분히 포함하지 못해 “SOTA” 주장에는 한계가 있습니다.

수정본에서는 다음과 같이 대응하겠습니다.
1. **SOTA 문구 톤다운 및 범위 명시**: 전면적인 SOTA가 아니라, *비교한 baseline 범위 내에서 경쟁력/우수 성능*임을 명확히 표현하겠습니다.
2. **대표 최신 모델 3종 추가 비교(진행 중)**: 아래 3개 공개 구현을 동일한 데이터/노이즈 설정과 동일한 지표로 재현하여 비교 baseline에 포함하겠습니다.
   - Operational CycleGAN 기반 blind restoration: https://github.com/OzerCanDevecioglu/Blind-ECG-Restoration-by-Operational-Cycle-GANs.git
   - Score-based / diffusion ECG denoising: https://github.com/HuayuLiArizona/Score-based-ECG-Denoising.git
   - Fully-gated DAE: https://github.com/AhmedAShaheen/fully_gated_DAE.git

공정 비교를 위해 (i) 동일 split/동일 noise 조합, (ii) 동일 평가 지표, (iii) 입력 길이/샘플링레이트/전처리 정합을 우선하며, 구현 차이로 인해 불가피한 설정 차이가 있다면 본문/부록에 명시하겠습니다.
