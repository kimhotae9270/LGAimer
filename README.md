# 📈 Resort Menu Demand Forecasting

리조트 내 식음업장 메뉴별 **매출 수요 예측 시스템**입니다.  
전통적인 시계열 모델과 딥러닝 모델을 혼합하여 **+1일 ~ +7일 Horizon 예측**을 수행합니다.  

---

## ⚙️ 주요 기능
- **데이터 전처리 & 피처 엔지니어링**
  - Lag, Rolling Mean, Zero Streak 등 다양한 시계열 기반 피처 생성
  - 아이템(메뉴)별 카테고리 인코딩 및 결측치 보정
- **모델 학습**
  - 🌳 Tree 기반 모델: **LightGBM**, **XGBoost** (GPU 가속)
  - 🤖 딥러닝 기반 모델: **N-BEATS** (PyTorch 구현, Item Embedding 포함)
  - 📉 통계 기반 모델: **SARIMAX** (아이템별 AIC 기반 선택), **Zero-Inflated Poisson (ZIP)**
- **앙상블(Ensemble)**
  - XGBoost + LightGBM + N-BEATS + SARIMAX + ZIP 결과를 가중합
  - `scipy.optimize.minimize` 기반 자동 가중치 튜닝 (sMAPE 최소화)
- **추론 & 결과 저장**
  - 테스트 블록 단위 예측 수행
  - 최종 결과를 CSV 제출 파일로 생성

---

## 🛠 기술 스택
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-004C6D?style=for-the-badge&logo=lightgbm&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-003366?style=for-the-badge&logo=python&logoColor=white)

---

## 📊 주요 하이라이트
- **N-BEATS 모델 직접 구현**
  - Item Embedding 적용
  - SMAPE Loss 지원
  - AMP(자동 혼합 정밀도) 및 CosineAnnealingLR 스케줄러 적용
- **Auto Weight Tuning**
  - LightGBM / XGBoost / N-BEATS / ZIP 가중치 자동 조정
  - sMAPE 기준 최적화
- **GPU 활용**
  - PyTorch 학습 및 LGBM/XGB GPU 가속 지원

---
