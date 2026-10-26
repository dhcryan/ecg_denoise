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
