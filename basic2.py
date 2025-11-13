import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 0. Utils
# -----------------------------
def conv_bn_act(in_ch, out_ch, k=3, s=1, p=1, act=True):
    m = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
         nn.BatchNorm2d(out_ch)]
    if act:
        m.append(nn.SiLU(inplace=True))
    return nn.Sequential(*m)



# -----------------------------
# 1. Anomaly-Aware CNN Stem (Option1: Standard Conv + High-Freq branch)
# 출력 해상도: 입력의 1/8 (stride 2 × 3회)
# -----------------------------
class FixedGaussianBlur(nn.Module):
    def __init__(self, channels, k=5, sigma=1.0):
        super().__init__()
        grid = torch.arange(k).float() - (k-1)/2
        gauss = torch.exp(- (grid**2) / (2*sigma**2))
        kernel1d = gauss / gauss.sum()
        kernel2d = torch.outer(kernel1d, kernel1d)
        weight = kernel2d[None, None, :, :].repeat(channels, 1, 1, 1)
        self.register_buffer('weight', weight)
        self.groups = channels
        self.k = k
    def forward(self, x):
        pad = (self.k//2,)*4
        return F.conv2d(F.pad(x, pad, mode='reflect'), self.weight, groups=self.groups)

class AnomalyAwareStem(nn.Module):
    def __init__(self, in_ch=3, base_ch=48):
        super().__init__()
        C1, C2, C3 = base_ch, base_ch*2, base_ch*4  # 최종 C = 4*base_ch
        self.stem = nn.Sequential(
            conv_bn_act(in_ch, C1, 3, 2, 1),
            conv_bn_act(C1, C2, 3, 2, 1),
            conv_bn_act(C2, C3, 3, 2, 1),
        )
        self.blur = FixedGaussianBlur(in_ch, k=5, sigma=1.0)
        self.anom = nn.Sequential(                      # 얕은 에지/질감 분기
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, C3//4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C3//4), nn.SiLU(inplace=True),
        )
        self.fuse = nn.Conv2d(C3 + C3//4, C3, 1, 1, 0, bias=False)
        self.fuse_bn = nn.BatchNorm2d(C3)
        self.vis_head = nn.Conv2d(C3, 1, 1, 1, 0)

    @property
    def out_channels(self):
        return 4*48  # default with base_ch=48 (수정 시 주의)

    def forward(self, x):
        f_main = self.stem(x)                     # (B,C,H/8,W/8)
        blurred = self.blur(x)
        high = x - blurred
        high_ds = F.interpolate(high, size=f_main.shape[-2:], mode='bilinear', align_corners=False)
        f_anom = self.anom(high_ds)
        f = torch.cat([f_main, f_anom], dim=1)
        f = self.fuse_bn(self.fuse(f))
        f = F.silu(f, inplace=True)
        v = torch.sigmoid(self.vis_head(f_main))  # 가시성 맵 (옵션: ViT에서 쓸 수 있음)
        return f, v


# -----------------------------
# 2. ViT Encoder (기본 MSA) + Patch Embedding
# 입력: (B,C,H,W)  → 토큰 (B,N,D)
# -----------------------------
class PatchEmbed1x1(nn.Module):
    """CNN 출력 채널 C → ViT 임베딩 D 로 맞추는 1x1 conv, 해상도 유지"""
    def __init__(self, in_ch, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(embed_dim)
    def forward(self, x):
        x = self.proj(x)
        x = self.bn(x)
        x = F.silu(x, inplace=True)
        return x

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x); return x

class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,h,N,d)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1,2).reshape(B, N, C)
        out = self.proj(out); out = self.proj_drop(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadSelfAttention(dim, num_heads, drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViTEncoder(nn.Module):
    def __init__(self, embed_dim=512, depth=8, num_heads=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=4.0, drop=0.0)
        for _ in range(depth)])
    def forward(self, tokens):
        # tokens: (B,N,D)
        for blk in self.blocks:
            tokens = blk(tokens)
        return tokens
    
# -----------------------------
# ViT 토큰(B,N,D)을 Stem feature(B,Cs,H/8,W/8)에 '실제'로 재주입하는 어댑터.
# tokens -> (B,D,Ht,Wt)로 복원 후 1x1 conv로 (gamma, beta) 생성
# f_fb = f_stem * (1 + gamma) + beta  (FiLM-like gating)
# -----------------------------
class FeedbackAdapter(nn.Module):
    def __init__(self, d_token: int, c_stem: int, use_bn: bool = True):
        super().__init__()
        layers = [nn.Conv2d(d_token, c_stem * 2, 1, 1, 0, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(c_stem * 2))
        layers.append(nn.SiLU(inplace=True))
        self.adapter = nn.Sequential(*layers)

    def forward(self, tokens: torch.Tensor, Ht: int, Wt: int, f_stem: torch.Tensor):
        # tokens: (B, N, D), f_stem: (B, Cs, Ht, Wt)
        B, N, D = tokens.shape
        t2d = tokens.transpose(1, 2).reshape(B, D, Ht, Wt)  # (B,D,Ht,Wt)
        ab = self.adapter(t2d)                               # (B, 2*Cs, Ht, Wt)
        Cs = f_stem.shape[1]
        gamma, beta = torch.split(ab, Cs, dim=1)             # (B,Cs,Ht,Wt) each
        return f_stem * (1 + torch.tanh(gamma)) + beta



# -----------------------------
# 3. PAN-Lite Neck (P3/P4/P5)
# 입력: ViT 출력 2D (P3) → 다운샘플로 P4/P5 생성 + 경량 융합
# -----------------------------
class PANLite(nn.Module):
    def __init__(self, in_ch=512, mid=256):
        super().__init__()
        # 채널 정합
        self.lateral = conv_bn_act(in_ch, mid, 1, 1, 0)
        # P4/P5 생성
        self.down4 = conv_bn_act(mid, mid, 3, 2, 1)  # P3 -> P4
        self.down5 = conv_bn_act(mid, mid, 3, 2, 1)  # P4 -> P5
        # 상향 융합
        self.up4 = conv_bn_act(mid+mid, mid, 3, 1, 1)
        self.up3 = conv_bn_act(mid+mid, mid, 3, 1, 1)
        # 하향 보강
        self.down_f4 = conv_bn_act(mid, mid, 3, 2, 1)
        self.fuse4 = conv_bn_act(mid+mid, mid, 3, 1, 1)
        self.down_f5 = conv_bn_act(mid, mid, 3, 2, 1)
        self.fuse5 = conv_bn_act(mid+mid, mid, 3, 1, 1)

    def forward(self, p3):
        # p3: (B, C=embed_dim, H, W)  ex) 80x80 for input 640
        p3 = self.lateral(p3)               # (B,256,H,W)
        p4 = self.down4(p3)                 # (B,256,H/2,W/2)
        p5 = self.down5(p4)                 # (B,256,H/4,W/4)

        # top-down
        p4u = F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        p4 = self.up4(torch.cat([p4, p4u], dim=1))
        p3u = F.interpolate(p4, size=p3.shape[-2:], mode='nearest')
        p3 = self.up3(torch.cat([p3, p3u], dim=1))

        # bottom-up
        p4b = self.down_f4(p3)
        p4 = self.fuse4(torch.cat([p4, p4b], dim=1))
        p5b = self.down_f5(p4)
        p5 = self.fuse5(torch.cat([p5, p5b], dim=1))

        return p3, p4, p5  # strides ~8,16,32


# -----------------------------
# 4. YOLO-style Decoupled Head (anchor-free)
# 각 피처맵에서 (cls + obj + bbox) 예측
# -----------------------------
class YOLOHeadLite(nn.Module):
    def __init__(self, in_ch=256, num_classes=1, reg_max=0):  # reg_max=0 → 직접 (tx,ty,tw,th)
        super().__init__()
        c = in_ch
        # 분기 공유 블록
        self.stem3 = conv_bn_act(c, c, 3, 1, 1)
        self.stem4 = conv_bn_act(c, c, 3, 1, 1)
        self.stem5 = conv_bn_act(c, c, 3, 1, 1)
        # decoupled heads
        self.cls3 = nn.Conv2d(c, num_classes, 1, 1, 0)
        self.obj3 = nn.Conv2d(c, 1,           1, 1, 0)
        self.box3 = nn.Conv2d(c, 4,           1, 1, 0)

        self.cls4 = nn.Conv2d(c, num_classes, 1, 1, 0)
        self.obj4 = nn.Conv2d(c, 1,           1, 1, 0)
        self.box4 = nn.Conv2d(c, 4,           1, 1, 0)

        self.cls5 = nn.Conv2d(c, num_classes, 1, 1, 0)
        self.obj5 = nn.Conv2d(c, 1,           1, 1, 0)
        self.box5 = nn.Conv2d(c, 4,           1, 1, 0)

    def forward_single(self, x, stem, cls, obj, box):
        f = stem(x)
        return cls(f), obj(f), box(f)  # (B,C,H,W)

    def forward(self, p3, p4, p5):
        c3, o3, b3 = self.forward_single(p3, self.stem3, self.cls3, self.obj3, self.box3)
        c4, o4, b4 = self.forward_single(p4, self.stem4, self.cls4, self.obj4, self.box4)
        c5, o5, b5 = self.forward_single(p5, self.stem5, self.cls5, self.obj5, self.box5)
        # 출력은 리스트로 반환 (후처리는 별도 모듈에서)
        return [(c3, o3, b3), (c4, o4, b4), (c5, o5, b5)]


# -----------------------------
# 5. 전체 모델: HybridOneWay
# -----------------------------
class HybridTwoWay(nn.Module):
    """
    실제 양방향(피드백) 버전:
    Stem → ViT → (FeedbackAdapter로 Stem 재조정) → Neck → Head
    - iters>1 로 두면 간단한 반복(Iterative) 갱신도 가능
    - detach_feedback=True 로 하면 안정화를 위해 피드백 경로의 gradient 차단
    """
    def __init__(self,
                 in_ch=3,
                 stem_base=48,
                 embed_dim=512,
                 vit_depth=8,
                 vit_heads=8,
                 num_classes=1,
                 iters=1,
                 detach_feedback=False):
        super().__init__()
        assert iters >= 1
        self.iters = iters
        self.detach_feedback = detach_feedback

        self.stem = AnomalyAwareStem(in_ch=in_ch, base_ch=stem_base)
        c_stem = stem_base * 4

        self.patch = PatchEmbed1x1(c_stem, embed_dim)
        self.vit = ViTEncoder(embed_dim=embed_dim, depth=vit_depth, num_heads=vit_heads)
        self.feedback = FeedbackAdapter(embed_dim, c_stem, use_bn=True)

        self.neck = PANLite(in_ch=embed_dim, mid=256)
        self.head = YOLOHeadLite(in_ch=256, num_classes=num_classes)

    def forward_once(self, x):
        # 1) Stem
        f_stem, vis = self.stem(x)                    # (B,Cs,H/8,W/8), (B,1,H/8,W/8)

        # 2) Patch -> Tokens -> ViT
        p = self.patch(f_stem)                        # (B,D,H/8,W/8)
        Ht, Wt = p.shape[-2:]
        tokens = p.flatten(2).transpose(1, 2)         # (B,N,D)
        tokens = self.vit(tokens)                     # (B,N,D)

        # 3) Feedback: ViT -> Stem 재조정
        toks_for_fb = tokens.detach() if self.detach_feedback else tokens
        f_fb = self.feedback(toks_for_fb, Ht, Wt, f_stem)   # (B,Cs,H/8,W/8)

        # 4) 병목을 피하려면, 재조정된 stem로 neck/head 진행
        p3_in = self.patch(f_fb)                      # (B,D,H/8,W/8)
        p3 = p3_in
        p3, p4, p5 = self.neck(p3)                    # (B,256, ...)
        preds = self.head(p3, p4, p5)
        aux = {'P3': p3, 'P4': p4, 'P5': p5, 'V': vis}
        return preds, aux, f_fb

    def forward(self, x):
        # 1회면 일반 피드백, 2회 이상이면 간단한 iterative refinement
        preds, aux, f_fb = self.forward_once(x)
        for _ in range(self.iters - 1):
            # 반복 시에는 f_fb를 새 입력처럼 간주하지 않고, 원 입력 x로 다시 지나가되
            # 첫 패스의 tokens 특성을 반영한 f_fb가 초기분포를 이미 조정했다고 해석.
            # (더 강한 반복을 원하면 forward_once를 변형해 f_fb를 바로 사용하도록 확장 가능)
            preds, aux, f_fb = self.forward_once(x)
        return preds, aux




if __name__ == "__main__":
    # --- 실제 양방향(피드백) ---
    model = HybridTwoWay(in_ch=3, stem_base=48, embed_dim=512, vit_depth=8, vit_heads=8,
                          num_classes=1, iters=1, detach_feedback=True)  # 안정화를 위해 초기엔 detach 권장
    x = torch.randn(2, 3, 640, 640)
    preds, aux = model(x)
    for i, (c, o, b) in enumerate(preds, start=3):
        print(f"[TwoWay] P{i} cls:{list(c.shape)} obj:{list(o.shape)} box:{list(b.shape)}")
