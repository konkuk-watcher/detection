# HybridTwoWay 피드백 기반 하이브리드 모델

> **프로젝트 유형:** 전장 인식 연구 / 객체 탐지
> **프레임워크:** PyTorch
> **모델 구조:** Anomaly-Aware CNN Stem → ViT Encoder → 반복적 피드백 → PANLite Neck → YOLOHead
> **목표:** CNN의 국소 특징과 ViT의 전역 문맥을 반복적 피드백 루프로 결합하여 탐지 정확도 향상


> **사용파일:** model.ipynb
> **모델 구조 개선:** Flash Attention 적용, Dynamic Positional Embedding resizing, torch.compile 지원


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
   └─ 다이나믹 리사이징 지원 (bicubic interpolation)
   │
   ▼
④ ViT Encoder
   ├─ CNN feature를 flatten → (B, N=Ht×Wt, D)
   ├─ Multihead Self-Attention으로 전역 문맥 학습
   ├─ Flash Attention 적용 (scaled_dot_product_attention)
   ├─ Transformer 블록으로 구성 (LayerNorm + Attention + MLP)
   ├─ 출력 토큰 (B, N, D)
   │
   ▼
⑤ FeedbackAdapter
   ├─ ViT 토큰을 reshape → (B, D, Ht, Wt)
   ├─ 1×1 conv로 (γ, β) 생성 (c_stem * 2 채널)
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
   ├─ ViT 처리 → Feedback 적용 (detach_feedback 옵션)
   ├─ Neck/Head 예측
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
   ├─ obj 레이어 bias 초기값: -4.59 (confidence 맞춤)
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

## 모델 구성 파라미터

- **in_ch**: 3 (입력 채널 수)
- **stem_base**: 32 (AnomalyAwareStem 기본 채널 수)
- **embed_dim**: 256 (ViT 임베딩 차원)
- **vit_depth**: 4 (ViT 인코더 블록 수)
- **vit_heads**: 4 (Multihead Attention 헤드 수)
- **num_classes**: 3 (클래스 수)
- **iters**: 1 (반복 횟수)
- **detach_feedback**: True (피드백 토큰 detach 여부)
- **img_size**: 640 (입력 이미지 크기)

---

## 환경 및 라이브러리

- **Python >= 3.8**
- **torch >= 2.0** (Flash Attention, torch.compile 지원)
- **torchvision**
- **numpy**
- **opencv-python**
- **tqdm**
- **pillow**
- **matplotlib**
- **roboflow** (데이터셋 다운로드용)
- **albumentations** (데이터 증강용 - 선택적)

*`model.ipynb` 기준이며, 데이터 증강, 후처리 등에 따라 다른 라이브러리가 필요할 수 있습니다.*

---

## 학습 및 평가

모델 학습은 YOLO 스타일의 손실 함수를 사용하여 진행됩니다:
- **클래스 예측**: Focal Loss (alpha=0.25, gamma=2.0)
- **신뢰도 예측**: Focal Loss (alpha=0.25, gamma=2.0)
- **바운딩박스 예측**: GIoU Loss

각 스케일(P3, P4, P5)에서 독립적으로 예측을 수행하고, 모든 스케일의 예측을 통합하여 최종 손실을 계산합니다.

학습 설정:
- **옵티마이저**: Adam (lr=1e-4)
- **에포크 수**: 5
- **배치 크기**: 8
- **AMP (Automatic Mixed Precision)**: 활성화 (torch.amp.autocast)
- **Gradient Scaler**: torch.cuda.amp.GradScaler
- **num_workers**: 2
- **pin_memory**: True

torch.compile 지원:
- **torch.compile**: 모델 컴파일 옵션 (PyTorch 2.0+ 지원)
- **Flash Attention**: torch.nn.functional.scaled_dot_product_attention 사용

학습된 모델은 mAP@0.5 지표를 사용하여 검증 및 테스트 데이터셋에서 평가됩니다.

평가 지표 및 방법:
- **mAP@0.5**: 0.5 IoU 임계값 기준 평균 정밀도
- **Confidence threshold**: 0.25
- **NMS IoU threshold**: 0.5
- **클래스 수**: 3 (num_classes 파라미터에 따라 조정 가능)
- **이미지 크기**: 640 (img_size 파라미터에 따라 조정 가능)
- **예측 디코딩**: NMS (Non-Maximum Suppression) 적용
- **성능 저장**: 최고 성능 모델은 'hybrid_two_way_best.pt'로 저장

---

# HybridTwoWay 피드백 기반 하이브리드 모델 (Advanced)

> **프로젝트 유형:** 전장 인식 연구 / 객체 탐지
> **프레임워크:** PyTorch + Timm
> **모델 구조:** Anomaly-Aware CNN Stem → Pretrained ViT Encoder → 반복적 피드백 → PANLite Neck → YOLOHead
> **목표:** 사전학습된 ViT의 표현력을 활용하고, CNN의 국소 특징과 ViT의 전역 문맥을 반복적 피드백 루프로 결합하여 탐지 정확도 향상

> **사용파일:** advanced_model.ipynb
> **모델 구조 개선:** Timm 사전학습 모델 적용, Task Aligned Assignment (TAL) Loss, Mosaic Augmentation, torch.compile 지원

---

## 1️⃣ 프로젝트 배경

전장 환경의 객체 탐지는 위장, 가림, 작은 크기 등 비정형적 특징 때문에 기존 모델로 어렵습니다. CNN은 텍스처 등 국소 정보에 강하지만 전체적인 맥락 파악이 어렵고, ViT는 전역 관계 추론에 유리하지만 세밀한 공간 정보를 놓칠 수 있습니다.

Advanced 모델은 사전학습된 ViT 모델을 사용하여 더 강력한 전역 문맥 표현을 추출하고, ViT가 파악한 **전역 문맥을 다시 CNN 특징맵에 주입(Feedback)**하여 국소 특징을 재조정하는 반복적 구조를 통해 탐지 성능을 극대화하는 것을 목표로 합니다. 특히 Anomaly-Aware 특징 추출과 반복적 피드백 메커니즘, Task Aligned Assignment (TAL) 손실을 통해 전장 환경의 도전적인 조건에 강건한 탐지 모델을 개발합니다.

---

## 2️⃣ 설계 철학

- **Timm Pretrained ViT Integration:** 사전학습된 ViT 모델(vit_base_patch16_224.augreg_in21k_ft_in1k 등)을 통합하여 강력한 특징 인코딩 능력을 확보합니다. Positional Embedding을 동적으로 리사이즈하여 다양한 입력 크기에 대응합니다.
- **Anomaly-Aware Stem:** 초기 CNN 단계에서 고주파(High-Frequency) 특징(에지, 질감)을 별도 분기로 추출하여 일반 특징과 융합함으로써, 위장 객체나 비정형 특징에 대한 민감도를 높입니다. Gaussian blur를 적용한 원본과의 차이를 통해 고주파 성분을 추출합니다.
- **Global Context Encoding:** 사전학습된 ViT를 통해 이미지 전체의 관계성을 모델링하고, 객체와 배경, 객체와 객체 간의 전역 문맥 정보를 추출합니다. Positional Embedding을 추가하여 공간 정보를 보존합니다.
- **Iterative Feedback:** ViT가 추출한 전역 문맥을 **Feedback Adapter**를 통해 CNN 특징맵에 다시 주입합니다. 이 과정을 통해 국소 특징이 전역 문맥에 맞게 보정됩니다. 이 피드백은 반복적으로 수행되어 점진적인 특징 개선을 달성합니다.
- **Task Aligned Assignment (TAL):** YOLOv8 스타일의 TAL을 구현하여 더 정확한 예측-정답 할당을 통해 학습 효율성과 성능을 향상시킵니다. 클래스 점수와 IoU 지표를 결합하여 후보 앵커를 평가하고 할당합니다.
- **Mosaic Augmentation:** 4장의 이미지를 조합하여 학습 데이터의 다양성을 확보하고, 소규모 객체 탐지 성능을 향상시킵니다.
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
② ViT Dimension Adapter
   ├─ CNN 특징의 채널수 Cs → ViT 임베딩차원 D로 1×1 conv
   └─ BatchNorm + SiLU 적용
   │
   ▼
③ Positional Embedding (Dynamic)
   ├─ Timm 사전학습 모델의 2D 공간 정보를 현재 입력 크기로 보간
   └─ Bicubic interpolation을 사용한 동적 리사이징
   │
   ▼
④ Pretrained ViT Encoder (from Timm)
   ├─ 사전학습된 ViT 블록 (Transformer blocks)
   ├─ Multihead Self-Attention으로 전역 문맥 학습
   ├─ ImageNet-21k에서 사전학습된 모델 사용
   ├─ Fine-tuning용으로 ImageNet-1k 데이터로 파인튜닝됨
   ├─ 출력 토큰 (B, N, D)
   │
   ▼
⑤ FeedbackAdapter
   ├─ ViT 토큰을 reshape → (B, D, Ht, Wt)
   ├─ 1×1 conv로 (γ, β) 생성 (c_stem * 2 채널)
   ├─ CNN 출력 보정:
     f_fb = f_stem × (1 + tanh(γ)) + β
   └─ CNN의 지역 특징을 ViT가 본 전역 문맥으로 재조정
   │
   ▼
⑥ Neck Dimension Adapter
   ├─ 보정된 f_fb를 Neck 차원으로 재매핑 (예: 128→256)
   │
   ▼
⑦ 반복 (iters 지정 횟수만큼)
   ├─ ViT 처리 → Feedback 적용 (detach_feedback 옵션)
   ├─ Neck/Head 예측
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
   ├─ obj 레이어 bias 초기값: -4.59 (confidence 맞춤)
   └─ 각 스케일에서 클래스, 신뢰도, 바운딩박스 예측
   │
   ▼
⑩ Task Aligned Assignment (TAL) 손실
   ├─ 클래스 점수와 IoU 지표를 결합 (s^alpha * u^beta)
   ├─ Top-k 후보 선택 및 가장 많이 겹치는 GT 포함 보장
   ├─ Anchor-free 방식이 아닌 Anchor-based 방식
   └─ 다중 스케일 예측 통합 손실
   │
   ▼
출력
   ├─ 예측: [(cls,obj,box)_P3, P4, P5]
   └─ 중간 feature: {'P3', 'P4', 'P5', 'V'}

```

---

## 모델 구성 파라미터

- **in_ch**: 3 (입력 채널 수)
- **stem_base**: 64 (AnomalyAwareStem 기본 채널 수 - Advanced 모델에서는 증가)
- **embed_dim**: 768 (ViT Base 임베딩 차원 - 사전학습 모델에 따라 조정)
- **vit_model_name**: 'vit_base_patch16_224.augreg_in21k_ft_in1k' (Timm 사전학습 모델명)
- **num_classes**: 3 (클래스 수)
- **iters**: 1 (반복 횟수 - Notebook에서는 Safe Mode로 1로 설정)
- **detach_feedback**: False (피드백 토큰 detach 여부 - Advanced 모델에서는 False)
- **img_size**: 512 (입력 이미지 크기)

---

## 환경 및 라이브러리

- **Python >= 3.8**
- **torch >= 2.0** (Flash Attention, torch.compile 지원)
- **torchvision**
- **timm** (사전학습 ViT 모델용)
- **numpy**
- **opencv-python**
- **albumentations** (데이터 증강용)
- **tqdm**
- **pillow**
- **matplotlib**
- **roboflow** (데이터셋 다운로드용)

*`advanced_model.ipynb` 기준이며, 데이터 증강, 후처리 등에 따라 다른 라이브러리가 필요할 수 있습니다.*

---

## 학습 및 평가

Advanced 모델 학습은 YOLOv8 스타일의 Task Aligned Assignment (TAL) 손실 함수를 사용하여 진행됩니다:
- **클래스 예측**: Task Aligned Assignment 기반 BCE 손실
- **신뢰도 예측**: TAL 기반 할당으로 대체 (Objectness 학습 없음)
- **바운딩박스 예측**: GIoU Loss (1 - IoU)

각 스케일(P3, P4, P5)에서 독립적으로 예측을 수행하고, Task Aligned Assignment를 통해 예측-정답 할당을 최적화한 후 손실을 계산합니다.

학습 설정:
- **옵티마이저**: AdamW (lr=2e-5, weight_decay=0.05) - Notebook "Safe Mode" 설정
- **에포크 수**: 15 (전체 학습을 위해 50~100 필요)
- **배치 크기**: 4 (VRAM 절약을 위해 감소)
- **AMP (Automatic Mixed Precision)**: 활성화 (torch.amp.autocast, float16)
- **Gradient Scaler**: torch.cuda.amp.GradScaler
- **num_workers**: 2
- **pin_memory**: True
- **Gradient Clipping**: max_norm=0.5 (ViT 학습에서 중요)
- **Gradient Accumulation**: 4스텝 (실제 배치 크기 16으로 증가)
- **Learning Rate Scheduler**: OneCycleLR (max_lr=2e-5, pct_start=0.3)

torch.compile 지원:
- **torch.compile**: 모델 컴파일 옵션 (PyTorch 2.0+ 지원)
- **Flash Attention**: torch.nn.functional.scaled_dot_product_attention 사용

학습된 모델은 mAP@0.5 지표를 사용하여 검증 및 테스트 데이터셋에서 평가됩니다.

평가 지표 및 방법:
- **mAP@0.5**: 0.5 IoU 임계값 기준 평균 정밀도
- **Confidence threshold**: 0.001 (TAL에서는 낮은 임계값이 일반적)
- **NMS IoU threshold**: 0.5
- **클래스 수**: 3 (num_classes 파라미터에 따라 조정 가능)
- **이미지 크기**: 640 (img_size 파라미터에 따라 조정 가능)
- **예측 디코딩**: NMS (Non-Maximum Suppression) 적용
- **성능 저장**: 최고 성능 모델은 'hybrid_two_way_best.pt'로 저장

---

# SAM 기반 객체 탐지 모델 (SAMDetector)

> **프로젝트 유형:** 전장 인식 연구 / 객체 탐지
> **프레임워크:** PyTorch
> **모델 구조:** Frozen SAM Encoder → FPN-like Adapter → Detection Heads
> **목표:** Segment Anything Model(SAM)의 강력한 사전 학습 특징 추출기를 객체 탐지(Object Detection)에 적용하여, 적은 학습 비용으로 높은 성능을 달성.
> **사용파일:** sam_model.ipynb
> **모델 구조 특징:** SAM의 ViT-B 인코더를 동결(freeze)하여 백본으로 사용하고, 가벼운 어댑터와 탐지 헤드만 학습.

---

## 1️⃣ 프로젝트 배경

Segment Anything Model(SAM)은 놀라운 제로샷(zero-shot) 분할 성능을 보여주며 강력한 시각적 특징을 학습했음을 입증했습니다. 본 프로젝트는 이 SAM의 인코더를 특징 추출기로 활용하여 객체 탐지 문제를 해결하고자 합니다. 백본 전체를 재학습하는 대신, 사전 학습된 SAM의 가중치는 고정한 채 간단한 FPN 스타일의 어댑터와 탐지 헤드만을 추가하여 학습함으로써 효율성과 성능을 동시에 추구합니다.

---

## 2️⃣ 설계 철학

- **Frozen Backbone:** SAM의 이미지 인코더(ViT-B)를 동결하여 사용하여, 거대한 모델을 학습하는 데 필요한 막대한 계산 리소스를 절약하고 과적합(overfitting)을 방지합니다.
- **Lightweight Adapter:** SAM 인코더가 출력한 단일 스케일의 특징 맵을 입력받아, 간단한 컨볼루션과 업샘플링을 통해 FPN(Feature Pyramid Network)과 유사한 다중 스케일(P3, P4, P5) 특징을 생성합니다. 이를 통해 다양한 크기의 객체를 탐지할 수 있습니다.
- **Simple Detection Head:** 각 스케일의 특징 맵에 대해 1x1 컨볼루션으로 구성된 간단한 탐지 헤드를 적용하여 클래스, 객체 존재 여부(objectness), 그리고 바운딩 박스를 예측합니다. 구조가 간단하여 학습이 빠릅니다.

---

## 3️⃣ 전체 구조도

```text
입력 이미지 (1024x1024)
   │
   ▼
① SAM Image Encoder (vit_b, Frozen)
   ├─ 그래디언트 계산 비활성화 (torch.no_grad())
   └─ 특징 맵 출력 (B, 256, 64, 64) → P4의 입력으로 사용됨
   │
   ▼
② FPN-like Adapter (학습 대상)
   ├─ P4 경로: SAM 특징에 ConvNormAct 적용 → P4 (B, 256, 64, 64)
   ├─ P3 경로: P4를 업샘플링(Upsample) → ConvNormAct 적용 → P3 (B, 256, 128, 128)
   └─ P5 경로: P4에 ConvNormAct(stride=2) 적용 → P5 (B, 256, 32, 32)
   │
   ▼
③ Detection Heads (학습 대상)
   ├─ P3, P4, P5 각 스케일에 1x1 Conv 헤드 적용
   └─ 각 헤드는 (클래스 수 + 5) 채널을 예측 (클래스, 신뢰도, 박스 좌표)
   │
   ▼
④ 손실 계산 (ComputeLoss)
   ├─ 박스 예측: IoU Loss
   ├─ 클래스/신뢰도 예측: BCEWithLogitsLoss
   └─ YOLO와 유사한 방식의 타겟 할당 및 손실 계산
```

---

## 모델 구성 파라미터

- **SAM checkpoint**: "sam_vit_b_01ec64.pth"
- **num_classes**: 3
- **img_size**: 1024
- **batch_size**: 4

---

## 환경 및 라이브러리

- **torch, torchvision**
- **opencv-python, numpy, matplotlib**
- **roboflow** (데이터셋)
- **segment-anything** (SAM 모델)
- **torchmetrics** (mAP 계산)

---

## 학습 및 평가

- **손실 함수**: IoU Loss(박스)와 BCE Loss(클래스/신뢰도)를 결합한 커스텀 손실
- **옵티마이저**: AdamW (lr=1e-4)
- **평가 지표**: torchmetrics를 사용한 mAP@0.5
- **학습 전략**: SAM 백본은 동결하고 어댑터와 헤드만 학습.

---

# YOLOv8m + CBAM 모델

> **프로젝트 유형:** 전장 인식 연구 / 객체 탐지
> **프레임워크:** PyTorch, Ultralytics YOLOv8
> **모델 구조:** YOLOv8m Backbone + CBAM → YOLOv8 PAN Head → YOLOv8 Detect Head
> **목표:** YOLOv8 아키텍처에 CBAM(Convolutional Block Attention Module)을 통합하여, 채널과 공간적 특징의 중요도를 학습하고 전반적인 탐지 성능을 향상.
> **사용파일:** yolo_cbam_fix.ipynb
> **모델 구조 특징:** Ultralytics 프레임워크를 활용하여 커스텀 모듈(CBAM)을 YOLOv8m 백본에 주입. YAML 파일을 통해 아키텍처를 유연하게 정의.

---

## 1️⃣ 프로젝트 배경

YOLOv8은 빠른 속도와 높은 정확도를 자랑하는 객체 탐지 모델이지만, 모든 특징을 동등하게 처리하는 경향이 있습니다. CBAM은 채널 주의(Channel Attention)와 공간 주의(Spatial Attention) 메커니즘을 순차적으로 적용하여, "무엇을" 볼지(채널)와 "어디에" 집중할지(공간)를 모델이 스스로 학습하게 합니다. 이 프로젝트는 YOLOv8 백본의 주요 특징 추출 단계 이후에 CBAM을 적용하여 특징 표현력을 강화하고, 결과적으로 탐지 정확도를 높이는 것을 목표로 합니다.

---

## 2️⃣ 설계 철학

- **Dynamic Module Injection:** Ultralytics YOLOv8의 유연성을 활용하여, `CBAM_Universal`이라는 커스텀 모듈을 파이썬 코드 레벨에서 정의하고 시스템에 등록합니다. 이후 YAML 설정 파일을 통해 원하는 위치에 CBAM 모듈을 동적으로 삽입합니다.
- **Attention on Key Features:** 백본(Backbone)에서 다운샘플링이 일어나는 세 군데의 C2f 블록 바로 다음에 CBAM 모듈을 배치하여, 해상도가 바뀌기 직전의 특징 맵을 정제(refine)하도록 설계했습니다. 이를 통해 각기 다른 스케일의 특징들이 어텐션 메커니즘의 혜택을 받도록 합니다.
- **Multi-Stage Training:** 처음에는 비교적 작은 이미지 크기(640x640)로 모델을 안정적으로 학습시킨 후, 더 큰 이미지 크기(800x800, 1280x1280)로 점진적으로 학습을 이어갑니다. 이는 모델이 다양한 스케일의 객체에 더 잘 적응하도록 돕는 효과적인 fine-tuning 전략입니다.

---

## 3️⃣ 전체 구조도

```text
입력 이미지 (640, 800, 1280 등)
   │
   ▼
① YOLOv8m Backbone (CBAM 삽입)
   ├─ ...
   ├─ C2f 블록 (idx 4)
   ├─ CBAM 모듈 1 (채널 192)
   ├─ ...
   ├─ C2f 블록 (idx 7)
   ├─ CBAM 모듈 2 (채널 384)
   ├─ ...
   ├─ C2f 블록 (idx 10)
   ├─ CBAM 모듈 3 (채널 576)
   └─ SPPF
   │
   ▼
② YOLOv8 PAN Head
   ├─ Top-down & Bottom-up 특징 융합
   └─ 다중 스케일 특징맵 (P3, P4, P5) 생성
   │
   ▼
③ YOLOv8 Detect Head
   ├─ 각 스케일에서 (클래스, 박스) 예측
   └─ Ultralytics 프레임워크의 내장 손실 함수 사용
```

---

## 모델 구성 파라미터

- **base_model**: YOLOv8m
- **custom_yaml**: yolov8m_cbam_real_final.yaml
- **img_size**: 640 -> 800 -> 1280 (단계적 학습)
- **optimizer**: AdamW

---

## 환경 및 라이브러리

- **ultralytics** (YOLOv8 프레임워크)
- **roboflow** (데이터셋)
- **pyyaml** (YAML 설정 파일 생성)
- **torch, torchvision**

---

## 학습 및 평가

- **손실 함수**: YOLOv8 기본 손실 함수 (Box: CIoU Loss, Cls: VFL, DFL)
- **옵티마이저**: AdamW
- **학습 전략**: 사전 학습된 `yolov8m.pt` 가중치에서 시작. 640x640 크기로 1차 학습 후, 800x800, 1280x1280 크기로 2, 3차 학습 진행.
- **평가 지표**: mAP50-95, mAP50


