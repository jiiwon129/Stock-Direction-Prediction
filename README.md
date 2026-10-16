# Stock Direction Prediction

> 개장 전 공개 정보만을 활용해 당일 주가 등락을 5개 등급으로 예측한 Business Analytics 팀 프로젝트

## Overview

- **Task**: 전일 종가 대비 당일 종가 등락을 5개 등급(급하락·하락·보합·상승·급상승)으로 예측
- **Universe**: 50개 종목
- **Main idea**: 이미 관측된 장전 가격 변화와 과거 장전 이후 잔차의 경험분포를 결합해 등락 확률을 산출
- **My role**: 예측 구조 설계, 최종 모델 구현, 코드 정리 및 검증
- **Tech**: Python, pandas, NumPy

## Problem & Approach

전일 일봉 정보만으로는 장 마감 이후부터 개장 전까지 발생한 가격 변화를 충분히 반영하기 어렵다고 판단.
이를 보완하기 위해 **장전 가격 변화(pre-market gap)**와 과거의 **장전 이후 남은 수익률 분포**를 함께 사용.

최종 모델은 다음 정보를 활용

- `pre_gap`: 개장 전 마지막 가격과 전일 종가의 차이
- `post_gap`: 전일 시간외 가격 변화
- `rv20`: 최근 20거래일 종가 수익률 변동성
- `intra_vol20`: 최근 20거래일 장중 수익률 변동성
- `error_vol20`: 과거 장전 이후 잔차의 최근 변동성
- `error_down60`, `error_up60`: 하락·상승 방향을 분리한 잔차 변동성

과거 잔차의 경험분포에서 각 등락 구간의 확률을 계산하고, 과제의 비대칭 오분류 비용을 반영해 최종 등급을 선택

## Preventing Look-ahead Bias

예측 시점 이후 정보가 모델 입력에 들어가지 않도록 데이터 사용 시점을 명시적으로 제한

- 입력 데이터는 `known_at < cutoff` 조건을 만족하는 경우만 사용
- 당일 종가 및 정답(label)은 피처 생성 과정에서 사용하지 않음
- 09:30에 완성되는 장전 봉 역시 예측 입력에서 제외
- 과거 잔차를 계산할 때도 각 과거 날짜에서 당시 개장 전에 확인 가능한 가격만 사용

## Results

2026-06-01 ~ 2026-09-14의 73거래일, 50종목 **3,650건**에 대한 후향 평가 결과

| Metric | Result |
|---|---:|
| Assignment score | **0.5943** |
| Baseline score | 0.1492 |
| Score improvement | **+0.4451** |
| 5-class accuracy | 25.67% |
| Direction accuracy* | 63.14% |

\* 실제 보합을 제외하고 상승/하락 방향만 비교한 값

> 이 구간은 모델 개발 과정에서 이미 확인한 적이 있는 **retrospective evaluation interval**이며, 완전히 미사용한 독립 최종 테스트셋으로 해석하지 않음.

## Validation

최종 코드 정리 과정에서 성능 수치뿐 아니라 **코드 변경 전후의 동작 일치와 미래 정보 차단**을 함께 점검

- 522거래일 × 50종목 = **26,100행**의 입력 피처 비교
- 기존 코드와 정리본의 7개 피처 및 가용성 확인 변수가 일치하는지 점검
- 평가용·제출용 모델의 예측 결과 일치 여부 확인
- 미래 시점 데이터를 의도적으로 주입해도 피처와 예측이 변하지 않는지 검사
- 데이터가 부족한 종목/시점에 대한 예외 처리 확인

자세한 검증 항목은 [`docs/validation_summary.md`](docs/validation_summary.md)에 정리

## Repository Structure

```text
.
├── README.md
├── src/
│   └── model.py                 # 최종 피처 생성 및 예측 로직
├── notebooks/
│   └── model_overview.ipynb     # 모델 구조와 평가 결과 요약
├── evaluation/
│   └── metrics.json             # 평가 지표 및 confusion matrix
└── docs/
    ├── handoff_verification.json
    └── validation_summary.md    # 코드 정리 전후 및 누출 방지 검증 요약
```
