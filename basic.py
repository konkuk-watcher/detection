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

class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        x = self.dw(x); x = self.pw(x); x = self.bn(x); return self.act(x)

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
class HybridOneWay(nn.Module):
    """
    입력 640x640 → Stem(1/8) → ViT → P3/P4/P5 → Head
    """
    def __init__(self,
                 in_ch=3,
                 stem_base=48,
                 embed_dim=512,
                 vit_depth=8,
                 vit_heads=8,
                 num_classes=1):
        super().__init__()
        self.stem = AnomalyAwareStem(in_ch=in_ch, base_ch=stem_base)
        c_stem = stem_base*4
        self.patch = PatchEmbed1x1(c_stem, embed_dim)
        self.vit = ViTEncoder(embed_dim=embed_dim, depth=vit_depth, num_heads=vit_heads)
        self.neck = PANLite(in_ch=embed_dim, mid=256)
        self.head = YOLOHeadLite(in_ch=256, num_classes=num_classes)

    def forward(self, x):
        """
        반환:
          preds: [(cls,obj,box)_P3, ... P4, P5] 각 (B,C,H,W)
          aux:   {'P3':(B,embed_dim,H,W), 'V':(B,1,H,W)} 등
        """
        B, _, H, W = x.shape
        # 1) stem
        f, v = self.stem(x)                        # (B,Cs,H/8,W/8), (B,1,H/8,W/8)
        # 2) patch → tokens
        p = self.patch(f)                          # (B, D, H/8, W/8)
        Ht, Wt = p.shape[-2:]
        tokens = p.flatten(2).transpose(1, 2)      # (B, N=Ht*Wt, D)
        # 3) ViT
        tokens = self.vit(tokens)                  # (B, N, D)
        # 4) 2D 복원 → neck
        p3 = tokens.transpose(1, 2).reshape(B, -1, Ht, Wt)  # (B,D,H/8,W/8)
        p3, p4, p5 = self.neck(p3)                 # (B,256, H/8, H/16, H/32 ...)
        # 5) head
        preds = self.head(p3, p4, p5)
        return preds, {'P3': p3, 'P4': p4, 'P5': p5, 'V': v}


# -----------------------------
# 6. 간단한 테스트
# -----------------------------
if __name__ == "__main__":
    model = HybridOneWay(in_ch=3, stem_base=48, embed_dim=512, vit_depth=8, vit_heads=8, num_classes=1)
    x = torch.randn(2, 3, 640, 640)
    preds, aux = model(x)
    for i, (c, o, b) in enumerate(preds, start=3):
        print(f"P{i} cls:{list(c.shape)} obj:{list(o.shape)} box:{list(b.shape)}")
    for k, v in aux.items():
        print(k, list(v.shape))
