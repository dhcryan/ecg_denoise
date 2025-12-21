# 길이 불일치 문제 (Length Mismatch) 해결

## 🔴 발생 원인

### 에러 메시지
```
IndexError: boolean index did not match indexed array along dimension 0; 
dimension is 88576 but corresponding boolean dimension is 88369
```

### 근본 원인

1. **입력 신호 길이**: `lead2_360` → 88,369 샘플
2. **프레이밍 후 패딩**: 512 샘플 윈도우 + 256 샘플 홉으로 프레이밍할 때 패딩 추가
3. **Overlap-Add 복원**: 패딩된 길이로 복원 → 88,576 샘플
4. **시각화 시 인덱싱 오류**: `denoised_360[t_vis_idx]`에서 길이 불일치

### 문제 상황 다이어그램

```
원본 신호 (Lead II @ 267Hz)
    ↓
Resample (267Hz → 360Hz)
    └─ lead2_360: 88,369 샘플
    ↓
프레이밍 + 패딩 (WIN=512, HOP=256)
    └─ 패딩 추가로 길이 증가
    ↓
모델 예측
    ↓
Overlap-Add 복원
    └─ denoised_full: 88,576 샘플 (원본보다 길어짐!)
    ↓
시각화 시도
    └─ t_vis_idx (88,369개) vs denoised_360 (88,576개)
    └─ ❌ IndexError!
```

---

## ✅ 해결 방법

### 수정 전 (셀 8)
```python
# 문제: 길이 정렬 없음
denoised_360 = overlap_add(Y_hat_frames, WIN, HOP)[:padded_len]
# 여전히 길이가 다를 수 있음

# Downsample back to 267 Hz
denoised_267 = resample_poly(denoised_360, up_ds, down_ds)
```

### 수정 후 (셀 8) - **CRITICAL FIX**
```python
# Overlap-Add 복원
denoised_full = overlap_add(Y_hat_frames, WIN, HOP)

# ★ 길이를 원본 신호에 맞추기
min_len = min(len(lead2_360), len(denoised_full))
lead2_360 = lead2_360[:min_len]
denoised_360 = denoised_full[:min_len]

# 검증
assert len(lead2_360) == len(denoised_360)
print(f"✓ Lengths match: {len(lead2_360) == len(denoised_360)}")

# Downsample
denoised_267 = resample_poly(denoised_360, up_ds, down_ds)
```

### 시각화 셀들의 방어적 코딩 (셀 13, 14, 15, 16)

```python
# 1. 길이 검증 (AssertionError 조기 감지)
assert len(lead2_360) == len(denoised_360), \
    f"Length mismatch: {len(lead2_360)} vs {len(denoised_360)}"

# 2. 인덱싱 안전성
min_len = min(len(lead2_360), len(denoised_360))
sig_raw = lead2_360[:min_len]
sig_deno = denoised_360[:min_len]

# 3. R-peak 인덱싱 필터링
rpeak_vis_raw = rpeak_raw[rpeak_raw < np.sum(t_vis_idx)]
rpeak_vis_deno = rpeak_deno[rpeak_deno < np.sum(t_vis_idx)]

# 4. 조건부 플로팅 (R-peak이 없어도 에러 안 남)
if len(rpeak_vis_raw) > 0:
    ax.plot(t[rpeak_vis_raw], sig_raw[rpeak_vis_raw], 'rx', ...)
else:
    ax.plot([], [], 'rx', ...)  # 범례만 표시
```

---

## 📋 수정된 셀 목록

| 셀 | 이름 | 수정 사항 |
|-----|------|---------|
| **8** | Denoise Full Signal | ★ 길이 정렬: `min_len` 적용 |
| **13** | R-Peak Comparison | R-peak 인덱싱 필터링, 조건부 플로팅 |
| **14** | PSD Visualization | 동일 길이로 안전한 비교 |
| **15** | Interactive Plot | Plotly에도 길이 정렬 적용 |
| **16** | Summary Report | 최종 길이 확인 및 보고 |

---

## 🔍 상세 분석: 왜 길이가 달라지나?

### Framing 과정

```
원본: 88,369 샘플
WIN = 512
HOP = 256

n_frames = 1 + (n - win + hop - 1) // hop
         = 1 + (88369 - 512 + 256 - 1) // 256
         = 1 + 344
         = 345 프레임

Overlap-Add 복원 시:
total_len = (n_frames - 1) * hop + win
          = (345 - 1) * 256 + 512
          = 88,064 + 512
          = 88,576 샘플

원본 길이: 88,369
복원 길이: 88,576
차이: +207 샘플 (패딩으로 인한 증가)
```

### 왜 패딩이 추가되나?

```python
def frame_signal(x, ...):
    n = len(x)
    n_frames = 1 + (n - win + hop - 1) // hop
    total_needed = (n_frames - 1) * hop + win
    
    if total_needed > n:
        extra = total_needed - n
        x = np.pad(x, (0, extra), mode="reflect")  # ← 여기서 패딩 추가!
```

**해결**: `[:padded_len]`로 자르려 했으나, `padded_len`은 프레이밍 후 패딩된 길이이지, 원본 길이가 아니었음.

---

## ✨ 개선 효과

### 이전
- ❌ 길이 불일치로 IndexError 발생
- ❌ 시각화 실행 불가
- ❌ 분석 완료 불가

### 개선 후
- ✅ `min_len` 기반 길이 정렬
- ✅ 원본 신호 길이 보존
- ✅ 시각화/분석 정상 동작
- ✅ 조건부 플로팅으로 엣지 케이스 처리

---

## 🧪 테스트 권장사항

```python
# 셀 8 실행 후 검증
print(f"Lead2_360 length: {len(lead2_360)}")
print(f"Denoised_360 length: {len(denoised_360)}")
print(f"Match: {len(lead2_360) == len(denoised_360)}")

# 셀 13 실행 전 스냅샷
print(f"t_vis_idx sum: {np.sum(t_vis_idx)}")
print(f"lead2_360 length: {len(lead2_360)}")
assert np.sum(t_vis_idx) <= len(lead2_360)
```

---

## 📚 관련 개념

### Overlap-Add (Overlap-Add Synthesis)
- 프레임을 겹치게 복원하는 신호 처리 기법
- 창(window) 함수로 경계 부분 부드럽게 처리
- 한스(Hann) 창이 대칭이고 50% 오버랩일 때 완벽한 복원 가능

### 패딩 (Padding)
- 신호가 윈도우 크기보다 작을 때 추가
- "reflect" 모드: 경계에서 신호 반사
- 불필요한 길이 증가를 초래할 수 있음

### 해결책
- **명시적 길이 정렬**: 원본 길이 기준으로 trim
- **조기 검증**: AssertionError로 문제 즉시 감지
- **방어적 인덱싱**: 조건부 플로팅으로 엣지 케이스 처리

---

## 🎯 최종 체크리스트

- [x] 셀 8: `min_len` 기반 길이 정렬 추가
- [x] 셀 13: R-peak 인덱싱 필터링 + 조건부 플로팅
- [x] 셀 14: 안전한 길이 비교
- [x] 셀 15: Plotly도 길이 정렬 적용
- [x] 셀 16: 최종 보고서 길이 확인
- [x] 개선사항 문서화

이제 노트북을 다시 실행하면 **IndexError가 발생하지 않아야 합니다!** ✅
