# HybridTwoWay 피드백 기반 하이브리드 모델

> **프로젝트 유형:** 전장 인식 연구 / 객체 탐지
> **프레임워크:** PyTorch
> **모델 구조:** Anomaly-Aware CNN Stem → ViT Encoder → 반복적 피드백 → PANLite Neck → YOLOHead
> **목표:** CNN의 국소 특징과 ViT의 전역 문맥을 반복적 피드백 루프로 결합하여 탐지 정확도 향상


> **사용파일:** model.ipynb


---

## 1️⃣ 프로젝트 배경

전장 환경의 객체 탐지는 위장, 가림, 작은 크기 등 비정형적 특징 때문에 기존 모델로 어렵습니다. CNN은 텍스처 등 국소 정보에 강하지만 전체적인 맥락 파악이 어렵고, ViT는 전역 관계 추론에 유리하지만 세밀한 공간 정보를 놓칠 수 있습니다.

본 프로젝트는 이 두 장점을 결합하고, ViT가 파악한 **전역 문맥을 다시 CNN 특징맵에 주입(Feedback)**하여 국소 특징을 재조정하는 반복적 구조를 통해 탐지 성능을 극대화하는 것을 목표로 합니다. 특히 Anomaly-Aware 특징 추출과 반복적 피드백 메커니즘을 통해 전장 환경의 도전적인 조건에 강건한 탐지 모델을 개발합니다.

---

## 2️⃣ 설계 철학

- **Anomaly-Aware Stem:** 초기 CNN 단계에서 고주파(High-Frequency) 특징(에지, 질감)을 별도 분기로 추출하여 일반 특징과 융합함으로써, 위장 객체나 비정형 특징에 대한 민감도를 높입니다. Gaussian blur를 적용한 원본과의 차이를 통해 고주파 성분을 추출합니다.
- **Global Context Encoding:** CNN 특징맵을 ViT에 입력하여 이미지 전체의 관계성을 모델링하고, 객체와 배경, 객체와 객체 간의 전역 문맥 정보를 추출합니다. Positional Embedding을 추가하여 공간 정보를 보존합니다.
- **Iterative Feedback:** ViT가 추출한 전역 문맥을 **Feedback Adapter**를 통해 CNN 특징맵에 다시 주입합니다. 이 과정을 통해 국소 특징이 전역 문맥에 맞게 보정됩니다. 이 피드백은 반복적으로 수행되어 점진적인 특징 개선을 달성합니다.
- **Multi-Scale Detection:** PANLite 구조를 사용하여 P3, P4, P5 다중 스케일 피처를 생성하고, YOLOHeadLite를 통해 각 스케일에서 클래스, 신뢰도, 바운딩박스 예측을 수행합니다.

---

## 3️⃣ 전체 구조도

```text
입력 이미지 (예: 640×640)
   │
   ▼
① AnomalyAwareStem (CNN)
   ├─ 3개의 conv-bn-act 블록 (stride=2) → (B, Cs, H/8, W/8)
   ├─ 고주파 특징 추출: 원본 - Gaussian blur → 고주파 성분 분리
   ├─ 로컬 특징과 고주파 특징 융합
   └─ 가시성 맵(visibility map) 생성 (옵션)
   │
   ▼
② PatchEmbed1x1
   ├─ CNN 특징의 채널수 Cs → ViT 임베딩차원 D로 1×1 conv
   └─ 공간 크기 유지 (H/8, W/8)
   │
   ▼
③ Positional Embedding
   ├─ 2D 공간 정보를 1D 시퀀스에 추가
   └─ 이미지 크기(640x640) 기준으로 사전 학습된 위치 임베딩
   │
   ▼
④ ViT Encoder
   ├─ CNN feature를 flatten → (B, N=Ht×Wt, D)
   ├─ Multihead Self-Attention으로 전역 문맥 학습
   ├─ Transformer 블록으로 구성 (LayerNorm + Attention + MLP)
   ├─ 출력 토큰 (B, N, D)
   │
   ▼
⑤ FeedbackAdapter
   ├─ ViT 토큰을 reshape → (B, D, Ht, Wt)
   ├─ 1×1 conv로 (γ, β) 생성
   ├─ CNN 출력 보정:
     f_fb = f_stem × (1 + tanh(γ)) + β
   └─ CNN의 지역 특징을 ViT가 본 전역 문맥으로 재조정
   │
   ▼
⑥ PatchEmbed1x1 (다시)
   ├─ 보정된 f_fb를 ViT 차원 D로 재매핑
   │
   ▼
⑦ 반복 (iters 지정 횟수만큼)
   ├─ ViT 처리 → Feedback 적용 → Neck/Head 예측
   └─ 다음 반복을 위한 토큰 준비
   │
   ▼
⑧ PANLite (neck)
   ├─ P3 (80×80), P4 (40×40), P5 (20×20) 멀티스케일 생성
   ├─ top-down & bottom-up 피처 융합 구조
   └─ 최종 멀티스케일 피처맵 반환
   │
   ▼
⑨ YOLOHeadLite
   ├─ P3, P4, P5 각각에 대해 (cls, obj, box) 예측
   ├─ 3x3 conv stem + 1x1 conv head 블록
   └─ 각 스케일에서 클래스, 신뢰도, 바운딩박스 예측
   │
   ▼
⑩ Focal Loss + GIoU 손실
   ├─ 클래스 및 신뢰도 예측: Focal Loss
   ├─ 바운딩박스 예측: GIoU Loss
   └─ 다중 스케일 예측 통합 손실
   │
   ▼
출력
   ├─ 예측: [(cls,obj,box)_P3, P4, P5]
   └─ 중간 feature: {'P3', 'P4', 'P5', 'V'}

```

---

## 환경 및 라이브러리

- **Python >= 3.8**
- **torch >= 1.10**
- **torchvision**
- **numpy**
- **opencv-python**
- **tqdm**
- **pillow**
- **matplotlib**
- **roboflow** (데이터셋 다운로드용)

*`model.ipynb` 기준이며, 데이터 증강, 후처리 등에 따라 다른 라이브러리가 필요할 수 있습니다.*

---

## 학습 및 평가

모델 학습은 YOLO 스타일의 손실 함수를 사용하여 진행됩니다:
- **클래스 예측**: Focal Loss
- **신뢰도 예측**: Focal Loss
- **바운딩박스 예측**: GIoU Loss

각 스케일(P3, P4, P5)에서 독립적으로 예측을 수행하고, 모든 스케일의 예측을 통합하여 최종 손실을 계산합니다.

학습된 모델은 mAP@0.5 지표를 사용하여 검증 및 테스트 데이터셋에서 평가됩니다.

