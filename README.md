# CNN-ViT 피드백 기반 하이브리드 모델

> **프로젝트 유형:** 전장 인식 연구 / 객체 탐지  
> **프레임워크:** PyTorch  
> **모델 구조:** CNN Stem → ViT Encoder → 피드백 → Detection Neck/Head  
> **목표:** CNN의 국소 특징과 ViT의 전역 문맥을 피드백 루프로 결합하여 탐지 정확도 향상


> **사용파일:** model.ipynb 


---

## 1️⃣ 프로젝트 배경

전장 환경의 객체 탐지는 위장, 가림, 작은 크기 등 비정형적 특징 때문에 기존 모델로 어렵습니다. CNN은 텍스처 등 국소 정보에 강하지만 전체적인 맥락 파악이 어렵고, ViT는 전역 관계 추론에 유리하지만 세밀한 공간 정보를 놓칠 수 있습니다.

본 프로젝트는 이 두 장점을 결합하고, ViT가 파악한 **전역 문맥을 다시 CNN 특징맵에 주입(Feedback)**하여 국소 특징을 재조정하는 양방향 구조를 통해 탐지 성능을 극대화하는 것을 목표로 합니다.

---

## 2️⃣ 설계 철학

- **Anomaly-Aware Stem:** 초기 CNN 단계에서 고주파(High-Frequency) 특징(에지, 질감)을 별도 분기로 추출하여 일반 특징과 융합함으로써, 위장 객체나 비정형 특징에 대한 민감도를 높입니다.
- **Global Context Encoding:** CNN 특징맵을 ViT에 입력하여 이미지 전체의 관계성을 모델링하고, 객체와 배경, 객체와 객체 간의 문맥 정보를 추출합니다.
- **Iterative Feedback:** ViT가 추출한 전역 문맥을 **FiLM(Feature-wise Linear Modulation)과 유사한 방식의 Adapter**를 통해 초기 CNN 특징맵에 다시 주입합니다. 이 과정을 통해 국소 특징이 전역 문맥에 맞게 보정됩니다. 이 피드백은 반복적으로 수행될 수 있습니다.

---

## 3️⃣ 전체 구조도

```text
입력 이미지 (예: 640×640)
   │
   ▼
① AnomalyAwareStem (CNN)
   ├─ stride=8로 다운샘플 → (B, Cs, H/8, W/8)
   ├─ 로컬 특징 추출 (에지, 질감, 패턴)
   └─ 가시성 맵(visibility map) 생성 (옵션)
   │
   ▼
② PatchEmbed1x1
   ├─ 채널수 Cs → ViT 임베딩차원 D로 1×1 conv
   └─ 공간 크기 유지 (H/8, W/8)
   │
   ▼
③ ViT Encoder
   ├─ CNN feature를 flatten → (B, N=Ht×Wt, D)
   ├─ Self-Attention으로 전역 문맥(global context) 학습
   ├─ 출력 토큰 (B, N, D)
   │
   ▼
④ FeedbackAdapter
   ├─ ViT 토큰을 reshape → (B, D, Ht, Wt)
   ├─ 1×1 conv로 (γ, β) 생성
   ├─ CNN 출력 보정:  
     f_fb = f_stem × (1 + tanh(γ)) + β
   └─ CNN의 지역 feature를 ViT가 본 전역 문맥으로 재조정
   │
   ▼
⑤ PatchEmbed1x1 (다시)
   ├─ 보정된 f_fb를 ViT 차원 D로 재매핑
   │
   ▼
⑥ PANLite (neck)
   ├─ P3 (80×80), P4 (40×40), P5 (20×20) 멀티스케일 생성
   ├─ top-down & bottom-up 피처 융합
   │
   ▼
⑦ YOLOHeadLite
   ├─ 각 스케일별 (cls, obj, box) 예측
   ├─ 바운딩박스 좌표는 stride(8/16/32) 기반
   │
   ▼
⑧ 손실(Loss)
   ├─ YOLO 방식 (box + class + objectness)
   ├─ 역전파로 ViT, CNN까지 gradient 전달
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
- **numpy**

*`basic2.py` 기준이며, `torchvision` 등 다른 라이브러리는 데이터셋/후처리 등에 따라 필요할 수 있습니다.*