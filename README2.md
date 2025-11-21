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
- **iters**: 2 (반복 횟수 - Advanced 모델에서는 증가)
- **detach_feedback**: False (피드백 토큰 detach 여부 - Advanced 모델에서는 False)
- **img_size**: 640 (입력 이미지 크기)

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
- **옵티마이저**: AdamW (lr=5e-5, weight_decay=0.05) - Weight Decay로 일반화 성능 향상
- **에포크 수**: 15 (전체 학습을 위해 50~100 필요)
- **배치 크기**: 4 (VRAM 절약을 위해 감소)
- **AMP (Automatic Mixed Precision)**: 활성화 (torch.amp.autocast, float16)
- **Gradient Scaler**: torch.cuda.amp.GradScaler
- **num_workers**: 2
- **pin_memory**: True
- **Gradient Clipping**: max_norm=0.5 (ViT 학습에서 중요)
- **Gradient Accumulation**: 4스텝 (실제 배치 크기 16으로 증가)
- **Learning Rate Scheduler**: OneCycleLR (max_lr=1e-4, pct_start=0.2, warmup 20%)

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

</content>