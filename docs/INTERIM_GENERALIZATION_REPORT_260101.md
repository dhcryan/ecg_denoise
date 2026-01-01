# 중간 결과보고서 — ECG Denoising 일반화 테스트 및 성능평가 (260101)

작성일: 2026-01-01  
대상: Ajou 스타일 ECG(Lead II 단일 채널, 원본 267Hz)  
목적: 별도 clean GT 없이, 사전학습 denoising 모델의 **현장 데이터 일반화 성능**을 정량/정성적으로 점검

---

## 1) 요약 (Executive Summary)

- **전원(User01–User05) 배치 파이프라인 실행 성공** 및 결과 저장 완료.
- Denoising 이후 **HF 기반 잡음 지표가 평균 +15.22 dB 개선**(사용자별 11.31–20.20 dB).
- **Residual SNR proxy**는 평균 22.68 dB 수준.
- R-peak 안정성(원신호 vs denoise 신호 매칭)은 평균 **F1=0.619** (Precision=0.576, Recall=0.682).
- Sliding-window(w=10s/30s, step=1s) 기준 worst-window에서 match_rate 최소값이 0으로 기록되는 구간이 있어, **박동 수가 매우 적은 창(예: 1 beat)에서의 민감도**가 관찰됨.

> 본 보고서는 “일반화 테스트/성능평가” 관점으로만 기술하며, 데이터 생성/가공 방식의 상세는 포함하지 않습니다.

---

## 2) 평가 파이프라인 개요

### 입력/전처리
- 입력: 단일 컬럼 TXT(267Hz)
- 전처리: 360Hz 리샘플링, baseline centering, 초기 구간 트림
- 프레이밍: window=512, hop=256
- 추론: 전체 신호에 대해 overlap-add로 복원

### 모델/체크포인트
- 모델: Transformer 기반 Denoising Autoencoder(`Transformer_DAE`)
- 체크포인트: `0908/Transformer_DAE_weights.best.weights.h5`
- 실행 환경: TensorFlow/Keras 3 계열 환경에서 실행

### Label-free 지표(핵심)
- **HF SNR proxy 개선량(dB)**: denoise 후 고주파(잡음) 성분 감소를 이용한 상대 지표
- **Residual SNR proxy(dB)**: (raw - deno)를 잔차로 보고 신호 대비 잔차 크기 추정
- **R-peak 안정성**: raw/deno 각각에서 R-peak 검출 후 매칭하여 Precision/Recall/F1
- **Sliding-window 분포/최악 구간**: 10s/30s 창을 1s step으로 스캔하여 중앙값/분산 및 worst-window 추출

---

## 3) 결과 요약 (User01–User05)

기준 파일: `phase_batch_summary_260101.csv`

| User | HF gain (dB) | Residual (dB) | R-peak F1 | Prec | Rec | worst match (10s) | worst match (30s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| User01 | 14.95 | 23.63 | 0.593 | 0.500 | 0.727 | 0.0 | 0.0 |
| User02 | 15.17 | 19.69 | 0.696 | 0.727 | 0.667 | 0.0 | 0.0 |
| User03 | 11.31 | 20.30 | 0.615 | 0.615 | 0.615 | 0.0 | 0.0 |
| User04 | 14.46 | 23.15 | 0.583 | 0.500 | 0.700 | 0.0 | 0.0 |
| User05 | 20.20 | 26.62 | 0.609 | 0.538 | 0.700 | 0.0 | 0.0 |

**집계(5명 평균)**
- HF gain: **15.22 dB** (min 11.31 / max 20.20)
- Residual SNR proxy: **22.68 dB**
- R-peak stability: **F1 0.619 / Precision 0.576 / Recall 0.682**

---

## 4) 해석 및 코멘트

### 4.1 잡음 억제 관점
- 모든 사용자에서 HF gain이 +11 dB 이상으로 관찰되어, **고주파 잡음 억제는 일관적으로 작동**하는 패턴.
- User05에서 HF gain이 가장 크며(20.20 dB), 해당 케이스는 원신호의 고주파 잡음 비중이 상대적으로 높았을 가능성이 큼.

### 4.2 형태 보존/박동 안정성 관점(R-peak)
- R-peak F1이 0.58–0.70 범위로, **denoise로 인한 peak 형태/극성 변화 또는 detector 민감도 변화**가 일부 발생했을 가능성이 있음.
- Precision이 Recall보다 낮은 사용자(User01/04/05)가 있어, denoise 신호에서 검출되는 peak 수가 더 많아지는 패턴(추가 peak)이 동반될 수 있음.

### 4.3 worst-window(match_rate=0) 관련
- worst-window 최소 match_rate가 0인 이유는 대부분 **창 내 박동 수가 매우 적은 경우(예: 1 beat) 매칭 실패가 바로 0으로 떨어지는 구조** 때문일 수 있음.
- 따라서 “최악 구간”은 실제로는 **매칭이 어려운 저박동/경계 창의 영향**일 수 있으므로,
  - `n_pairs`, `n_rpeaks_raw`, `n_rpeaks_deno`를 함께 보고 해석하는 것을 권장.

---

## 5) 재현 방법(요약)

배치 실행 예:

```bash
/home/dhc99/anaconda3/bin/conda run -p /home/dhc99/anaconda3/envs/ECGDENOISE --no-capture-output \
  python tools/run_phase_batch_transformer_dae.py \
  --data_dir ajou_phase_augmented_267 \
  --out_root . \
  --date 260101 \
  --users User01 User02 User03 User04 User05 \
  --pretrained_weights 0908/Transformer_DAE_weights.best.weights.h5
```

출력:
- 사용자별 폴더: `ajou_outputs_phase_<User>_260101/`
- 전역 요약: `phase_batch_summary_260101.csv`

---

## 6) 한계 및 다음 액션

- **Clean GT 부재**로 인해 절대적 복원 성능(예: true SNR, PRD)을 직접 산출할 수 없고, 현재는 label-free proxy 중심.
- SciPy/NumPy 버전 경고가 있으나 현재 파이프라인은 동작함. 재현성과 안정성을 위해 환경 정합 권장.
- 다음 단계 제안:
  1) worst-window 후보 구간(예: center_s=117s/107s 부근)을 사용자별로 확대 시각화하여 원인(peak polarity/형태 변화 vs detector 설정) 확인
  2) R-peak detector 파라미터 튜닝 또는 polarity-robust 옵션을 고정하여 지표 변동성 감소
  3) 동일 파이프라인으로 추가 날짜/추가 사용자 확장하여 일반화 분산 확인
