#!/bin/bash
./hnsw -k100 -t40 -w 1 >& tout1.txt
가능해. 아래처럼 잡으면 꽤 제대로 된 **CNN + Cross-Attention 기반 정상/불량 비교 모델**이 돼.

이번 코드는 다음 흐름이야.

```text
정상 이미지 ─┐
             ├─ Shared ConvNeXtV2 Backbone
검사 이미지 ─┘
       │
       ├─ Stage 1 feature
       │      └─ 직접 feature difference
       │
       └─ Stage 2 feature
              └─ Cross-Attention
                    │
                    ├─ 검사 feature
                    ├─ matched 정상 feature
                    └─ difference
                          ↓
                       Fusion
                          ↓
                ┌─────────┴─────────┐
                ↓                   ↓
           Defect Map          정상/불량 분류
```

`timm`은 현재 `features_only=True`, `out_indices`, `feature_info.channels()`로 CNN의 중간 feature를 공식적으로 꺼낼 수 있어서 이 구조에 잘 맞는다. ([Hugging Face][1]) 예시는 실제 timm에 존재하는 `convnextv2_tiny.fcmae_ft_in1k`를 기준으로 하겠다. ([Hugging Face][2])

## 1. Cross-Attention부터

일단 이 부분이 핵심이야.

나는 여기서는 `nn.MultiheadAttention`을 그대로 쓰기보다 **정상/검사 feature가 동일한 latent space에서 직접 비교되도록** scaled dot-product attention을 구현할게.

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
```

```python
class CrossAttention2D(nn.Module):
    """
    test_feature   : [B, C, H, W]
    normal_feature : [B, C, H, W]

    test의 각 위치가 normal의 어느 위치를 참고해야 하는지 찾고,
    matched normal feature와 test feature의 차이를 반환.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 192,
        num_heads: int = 6,
        dropout: float = 0.0,
        max_tokens: int = 1024,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_tokens = max_tokens
        self.dropout = dropout

        # normal / test 둘 다 동일 projection 사용
        # -> 같은 latent space에 놓기 위함
        self.feature_proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=1,
            bias=False,
        )

        self.norm = nn.LayerNorm(embed_dim)

        # attention weight를 계산하기 위한 Q, K projection
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _pool_if_needed(self, x):
        """
        Full attention의 token 수가 너무 커지는 것을 방지.
        """
        B, C, H, W = x.shape

        num_tokens = H * W

        if num_tokens <= self.max_tokens:
            return x

        ratio = math.sqrt(self.max_tokens / num_tokens)

        new_h = max(1, int(H * ratio))
        new_w = max(1, int(W * ratio))

        return F.adaptive_avg_pool2d(
            x,
            (new_h, new_w),
        )

    def _split_heads(self, x):
        """
        [B, N, D]
            ->
        [B, heads, N, head_dim]
        """
        B, N, D = x.shape

        x = x.view(
            B,
            N,
            self.num_heads,
            self.head_dim,
        )

        return x.transpose(1, 2)

    def _merge_heads(self, x):
        """
        [B, heads, N, head_dim]
            ->
        [B, N, D]
        """
        B, H, N, D = x.shape

        x = x.transpose(1, 2).contiguous()

        return x.view(
            B,
            N,
            H * D,
        )

    def forward(
        self,
        test_feature,
        normal_feature,
        return_attention=False,
    ):
        # -----------------------------------
        # 1. 같은 latent space로 projection
        # -----------------------------------

        test_feature = self.feature_proj(test_feature)
        normal_feature = self.feature_proj(normal_feature)

        # Attention 계산량 제한
        test_feature = self._pool_if_needed(test_feature)
        normal_feature = self._pool_if_needed(normal_feature)

        B, C, Ht, Wt = test_feature.shape
        _, _, Hn, Wn = normal_feature.shape

        # -----------------------------------
        # 2. CNN feature -> tokens
        # -----------------------------------

        test_tokens = (
            test_feature
            .flatten(2)
            .transpose(1, 2)
        )

        normal_tokens = (
            normal_feature
            .flatten(2)
            .transpose(1, 2)
        )

        # [B, Nt, C]
        # [B, Nn, C]

        test_tokens = self.norm(test_tokens)
        normal_tokens = self.norm(normal_tokens)

        # -----------------------------------
        # 3. Query / Key
        # -----------------------------------

        q = self.q_proj(test_tokens)
        k = self.k_proj(normal_tokens)

        # Value는 normal feature 자체.
        # test/normal이 동일 latent space에 있으므로
        # 이후 직접 difference 계산 가능.
        v = normal_tokens

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # -----------------------------------
        # 4. Cross Attention
        # -----------------------------------

        if return_attention:
            scale = 1.0 / math.sqrt(self.head_dim)

            attention = torch.matmul(
                q,
                k.transpose(-2, -1),
            ) * scale

            attention = torch.softmax(
                attention,
                dim=-1,
            )

            matched_normal = torch.matmul(
                attention,
                v,
            )

            # visualization용으로 head 평균
            attention_map = attention.mean(dim=1)

        else:
            matched_normal = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
            )

            attention_map = None

        # -----------------------------------
        # 5. head 결합
        # -----------------------------------

        matched_normal = self._merge_heads(
            matched_normal
        )

        # -----------------------------------
        # 6. Difference
        # -----------------------------------

        difference = torch.abs(
            test_tokens - matched_normal
        )

        # 다시 CNN feature map으로
        test_map = (
            test_tokens
            .transpose(1, 2)
            .reshape(B, C, Ht, Wt)
        )

        matched_map = (
            matched_normal
            .transpose(1, 2)
            .reshape(B, C, Ht, Wt)
        )

        difference_map = (
            difference
            .transpose(1, 2)
            .reshape(B, C, Ht, Wt)
        )

        return {
            "test": test_map,
            "matched_normal": matched_map,
            "difference": difference_map,
            "attention": attention_map,
        }
```

실질적으로 여기서 하는 게:

[
Q=W_QF_{test}
]

[
K=W_KF_{normal}
]

[
V=F_{normal}
]

[
F_{matched}
===========

softmax
\left(
\frac{QK^T}{\sqrt d}
\right)V
]

그리고

[
D =
|F_{test}-F_{matched}|
]

야.

PyTorch의 scaled dot-product attention도 현재 공식 API로 제공되고 있다. ([PyTorch Docs][3])

---

# 2. 전체 불량 검출 모델

이제 여기에 ConvNeXtV2를 붙이자.

중요한 설계는 **Cross-Attention을 너무 고해상도 feature에는 사용하지 않는 것**이야.

ConvNeXt 계열은 대략:

```text
Input
 ↓
Stage 0   높은 해상도
 ↓
Stage 1
 ↓
Stage 2
 ↓
Stage 3   낮은 해상도
```

가 되므로 여기서는

```text
Stage 0 → direct difference
Stage 1 → cross attention
```

으로 해보자.

```python
import timm


class DefectCrossAttentionModel(nn.Module):

    def __init__(
        self,
        backbone_name="convnextv2_tiny.fcmae_ft_in1k",
        pretrained=True,
        embed_dim=192,
        num_heads=6,
        fusion_dim=256,
        max_attention_tokens=1024,
    ):
        super().__init__()

        # =============================================
        # Backbone
        # =============================================

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,

            # ConvNeXt의 앞쪽 두 stage 사용
            out_indices=(0, 1),
        )

        channels = self.backbone.feature_info.channels()
        reductions = self.backbone.feature_info.reduction()

        print("Backbone channels :", channels)
        print("Backbone reductions:", reductions)

        high_channels = channels[0]
        mid_channels = channels[1]

        # =============================================
        # High-resolution local branch
        # =============================================

        self.high_projection = nn.Sequential(
            nn.Conv2d(
                high_channels,
                fusion_dim // 2,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(fusion_dim // 2),
            nn.GELU(),
        )

        # =============================================
        # Cross-Attention branch
        # =============================================

        self.cross_attention = CrossAttention2D(
            in_channels=mid_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_tokens=max_attention_tokens,
        )

        # test
        # matched normal
        # |test - normal|
        # test * normal
        #
        # 총 embed_dim * 4

        self.cross_fusion = nn.Sequential(
            nn.Conv2d(
                embed_dim * 4,
                fusion_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(fusion_dim),
            nn.GELU(),

            nn.Conv2d(
                fusion_dim,
                fusion_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(fusion_dim),
            nn.GELU(),
        )

        # =============================================
        # High + Cross fusion
        # =============================================

        total_channels = fusion_dim + fusion_dim // 2

        self.final_fusion = nn.Sequential(
            nn.Conv2d(
                total_channels,
                fusion_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(fusion_dim),
            nn.GELU(),

            nn.Conv2d(
                fusion_dim,
                fusion_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(fusion_dim),
            nn.GELU(),
        )

        # =============================================
        # Defect localization head
        # =============================================

        self.defect_head = nn.Sequential(
            nn.Conv2d(
                fusion_dim,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                128,
                1,
                kernel_size=1,
            ),
        )

        # =============================================
        # Classification head
        # =============================================

        self.classifier = nn.Sequential(
            nn.Linear(
                fusion_dim * 2,
                256,
            ),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(
                256,
                1,
            ),
        )

    def forward(
        self,
        normal_image,
        test_image,
        return_attention=False,
    ):
        input_size = test_image.shape[-2:]

        # =============================================
        # 1. Shared Backbone
        # =============================================

        normal_features = self.backbone(normal_image)
        test_features = self.backbone(test_image)

        normal_high = normal_features[0]
        normal_mid = normal_features[1]

        test_high = test_features[0]
        test_mid = test_features[1]

        # =============================================
        # 2. High-resolution direct comparison
        # =============================================

        normal_high = self.high_projection(normal_high)
        test_high = self.high_projection(test_high)

        high_diff = torch.abs(
            test_high - normal_high
        )

        # =============================================
        # 3. Cross Attention
        # =============================================

        cross = self.cross_attention(
            test_mid,
            normal_mid,
            return_attention=return_attention,
        )

        test_cross = cross["test"]
        matched_normal = cross["matched_normal"]
        cross_diff = cross["difference"]

        # cosine-like interaction
        cross_product = (
            test_cross * matched_normal
        )

        cross_feature = torch.cat(
            [
                test_cross,
                matched_normal,
                cross_diff,
                cross_product,
            ],
            dim=1,
        )

        cross_feature = self.cross_fusion(
            cross_feature
        )

        # =============================================
        # 4. High resolution으로 복원
        # =============================================

        cross_feature = F.interpolate(
            cross_feature,
            size=high_diff.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # =============================================
        # 5. Final fusion
        # =============================================

        fused = torch.cat(
            [
                high_diff,
                cross_feature,
            ],
            dim=1,
        )

        fused = self.final_fusion(fused)

        # =============================================
        # 6. Defect map
        # =============================================

        defect_logits = self.defect_head(fused)

        # 원본 이미지 크기로
        defect_logits = F.interpolate(
            defect_logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        # =============================================
        # 7. Classification
        # =============================================

        avg_feature = F.adaptive_avg_pool2d(
            fused,
            1,
        ).flatten(1)

        max_feature = F.adaptive_max_pool2d(
            fused,
            1,
        ).flatten(1)

        global_feature = torch.cat(
            [
                avg_feature,
                max_feature,
            ],
            dim=1,
        )

        class_logits = self.classifier(
            global_feature
        ).squeeze(1)

        return {
            "logits": class_logits,
            "defect_logits": defect_logits,
            "attention": cross["attention"],
        }
```

---

# 3. 실제로 한번 돌려보기

```python
device = torch.device("cuda")

model = DefectCrossAttentionModel(
    backbone_name="convnextv2_tiny.fcmae_ft_in1k",
    pretrained=True,

    embed_dim=192,
    num_heads=6,

    fusion_dim=256,

    # Cross attention 최대 token 수
    max_attention_tokens=1024,
)

model = model.to(device)
```

테스트:

```python
normal = torch.randn(
    4,
    3,
    224,
    224,
    device=device,
)

test = torch.randn(
    4,
    3,
    224,
    224,
    device=device,
)

out = model(
    normal,
    test,
)

print(out["logits"].shape)
print(out["defect_logits"].shape)
```

대략:

```text
logits
torch.Size([4])

defect_logits
torch.Size([4, 1, 224, 224])
```

가 나와.

---

# 4. 불량 확률과 결함 영역 얻기

모델 출력은 전부 **logit**이야.

따라서:

```python
out = model(
    normal,
    test,
)

defect_probability = torch.sigmoid(
    out["logits"]
)

defect_map = torch.sigmoid(
    out["defect_logits"]
)
```

그러면:

```text
defect_probability

[0.02,
 0.94,
 0.07,
 0.81]
```

같이 이미지 전체의 불량 확률이 나오고,

```python
defect_map.shape
# [B, 1, H, W]
```

에는 위치별 불량 score가 들어가게 돼.

예를 들어:

```python
mask = defect_map > 0.5
```

하면 임시 binary defect mask도 만들 수 있고.

---

# 5. Loss

여기서 **결함 mask annotation이 있느냐 없느냐**에 따라 상당히 달라져.

### 이미지별 정상/불량 label만 있는 경우

```python
classification_loss_fn = nn.BCEWithLogitsLoss()
```

```python
def compute_loss_without_mask(
    output,
    label,
):
    cls_loss = F.binary_cross_entropy_with_logits(
        output["logits"],
        label.float(),
    )

    return cls_loss
```

그런데 이것만 쓰면 문제가 있어.

`defect_head`에는 직접적인 supervision이 없잖아.

그래서 defect map도 정상/불량 판단에 관여시키는 게 좋아.

예를 들어 **top-k MIL pooling**을 쓸 수 있어.

```python
def topk_map_pool(
    defect_logits,
    ratio=0.05,
):
    """
    defect_logits:
        [B, 1, H, W]

    가장 이상한 상위 5% 영역 평균을
    이미지 단위 defect logit으로 사용.
    """

    B = defect_logits.size(0)

    x = defect_logits.flatten(1)

    k = max(
        1,
        int(x.size(1) * ratio),
    )

    topk = torch.topk(
        x,
        k=k,
        dim=1,
    ).values

    return topk.mean(dim=1)
```

그리고:

```python
def compute_weakly_supervised_loss(
    output,
    label,
):
    label = label.float()

    # 전체 feature 기반 classification
    global_loss = (
        F.binary_cross_entropy_with_logits(
            output["logits"],
            label,
        )
    )

    # defect map 기반 classification
    map_logits = topk_map_pool(
        output["defect_logits"],
        ratio=0.05,
    )

    map_loss = (
        F.binary_cross_entropy_with_logits(
            map_logits,
            label,
        )
    )

    loss = (
        global_loss
        + 0.5 * map_loss
    )

    return loss
```

이렇게 하면 `defect_map`도 최소한

> "불량 이미지 어딘가에는 높은 score를 만들어야 한다."

라는 학습 신호를 받게 돼.

다만 **segmentation mask 없이 만든 map을 진짜 segmentation 결과라고 생각하면 안 돼.**

약지도 localization이기 때문에 실제 결함 전체가 아니라 **분류에 가장 유용한 일부 영역만 강조할 수도 있어.**

---

# 6. 실제 결함 Mask가 있는 경우

이 경우는 훨씬 좋지.

```python
def dice_loss(
    logits,
    target,
    eps=1e-6,
):
    pred = torch.sigmoid(logits)

    pred = pred.flatten(1)
    target = target.flatten(1)

    intersection = (
        pred * target
    ).sum(dim=1)

    dice = (
        2 * intersection + eps
    ) / (
        pred.sum(dim=1)
        + target.sum(dim=1)
        + eps
    )

    return 1 - dice.mean()
```

전체 loss:

```python
def compute_supervised_loss(
    output,
    label,
    mask,
):
    label = label.float()

    # classification
    cls_loss = (
        F.binary_cross_entropy_with_logits(
            output["logits"],
            label,
        )
    )

    # segmentation
    bce_seg = (
        F.binary_cross_entropy_with_logits(
            output["defect_logits"],
            mask.float(),
        )
    )

    d_loss = dice_loss(
        output["defect_logits"],
        mask.float(),
    )

    loss = (
        cls_loss
        + bce_seg
        + d_loss
    )

    return loss
```

이쪽이면 정말로

```text
정상/불량 여부

+

실제 불량 영역
```

두 개를 같이 학습시킬 수 있어.

---

# 7. AMP까지 넣은 학습 Loop

네가 AMP를 이미 쓰고 있으니까 그것까지 합치면:

```python
device = torch.device("cuda")

use_amp = True

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.05,
)
```

현재 PyTorch 공식 AMP 예제도 `torch.autocast(device_type=...)`와 `torch.amp.GradScaler("cuda")` 형식을 사용하고 있다. ([PyTorch Docs][4])

학습:

```python
model.train()

for normal, test, label in train_loader:

    normal = normal.to(
        device,
        non_blocking=True,
    )

    test = test.to(
        device,
        non_blocking=True,
    )

    label = label.to(
        device,
        non_blocking=True,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=use_amp,
    ):
        output = model(
            normal,
            test,
        )

        loss = compute_weakly_supervised_loss(
            output,
            label,
        )

    scaler.scale(loss).backward()

    scaler.step(optimizer)

    scaler.update()
```

---

# 8. 네가 이미 학습한 모델 가중치를 가져온다면

이 부분도 중요해.

기존 모델이 대략:

```python
class OldClassifier(nn.Module):
    def __init__(self):
        self.backbone = ...
        self.head = ...
```

였다면 checkpoint에서 backbone만 가져오는 게 좋아.

예를 들어 checkpoint가

```python
checkpoint = torch.load(
    "old_model.pt",
    map_location="cpu",
)

state_dict = checkpoint["model"]
```

라고 하면:

```python
backbone_state = {}

for key, value in state_dict.items():

    if key.startswith("backbone."):
        new_key = key.replace(
            "backbone.",
            "",
            1,
        )

        backbone_state[new_key] = value


result = model.backbone.load_state_dict(
    backbone_state,
    strict=False,
)

print("Missing:", result.missing_keys)
print("Unexpected:", result.unexpected_keys)
```

형태로 가져올 수 있어.

다만 **기존 네 모델에서 backbone을 어떤 방식으로 감쌌는지에 따라 state_dict key 이름은 확인해야 해.**

---

# 9. 처음에는 backbone을 freeze

난 이걸 먼저 추천해.

```python
for param in model.backbone.parameters():
    param.requires_grad = False
```

그러면 처음에는:

```text
기존 학습된 Backbone
        ↓ freeze

Cross Attention   ← 학습
Fusion            ← 학습
Defect Head       ← 학습
Classifier        ← 학습
```

만 한다.

Optimizer도:

```python
optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters(),
    ),
    lr=1e-4,
    weight_decay=0.05,
)
```

로.

Cross-Attention 부분이 어느 정도 안정된 후에 backbone을 풀고:

```python
for param in model.backbone.parameters():
    param.requires_grad = True
```

optimizer를 새로:

```python
optimizer = torch.optim.AdamW(
    [
        {
            "params": model.backbone.parameters(),
            "lr": 1e-5,
        },
        {
            "params": model.cross_attention.parameters(),
            "lr": 1e-4,
        },
        {
            "params": model.cross_fusion.parameters(),
            "lr": 1e-4,
        },
        {
            "params": model.final_fusion.parameters(),
            "lr": 1e-4,
        },
        {
            "params": model.defect_head.parameters(),
            "lr": 1e-4,
        },
        {
            "params": model.classifier.parameters(),
            "lr": 1e-4,
        },
    ],
    weight_decay=0.05,
)
```

처럼 가면 돼.

---

## 그리고 이 구조에서 내가 가장 중요하게 보는 부분

네 문제에서는 이 두 branch가 서로 역할이 달라.

```text
             작은 결함 / 세밀한 차이
                      ↓
Stage 0 ─────── |Test - Normal|
                      │
                      │
                      ├──────┐
                      │      │
                      │    Fusion → Defect map
                      │      │
                      ├──────┘
                      │
Stage 1 ───── Cross Attention
                      ↑
                위치가 조금 달라도
              대응되는 정상 구조 검색
```

즉 **direct difference와 Cross-Attention을 같이 쓰는 이유**가 있어.

Cross-Attention만 쓰면 정상 feature를 굉장히 잘 찾아와서 오히려 작은 defect까지 정상 패턴으로 설명해버리는 문제가 생길 수 있어.

반대로 direct difference만 쓰면:

```text
카메라 2pixel 이동
        ↓
회로 edge 전체가 difference 발생
```

할 수 있고.

둘을 같이 주면,

> **정확한 local 차이 + 정상 구조와의 semantic correspondence**

를 동시에 모델한테 줄 수 있어.

특히 반복적인 회로 패턴이나 작은 이물질을 상대한다면 이 조합을 첫 실험으로 꽤 추천해.

그리고 **정상/검사 이미지에 geometric augmentation을 줄 때는 같은 crop, flip, affine을 적용하는 게 중요해.** 정상만 왼쪽으로 translate하고 검사 이미지만 오른쪽으로 translate해버리면 high-resolution difference branch를 스스로 망가뜨리게 되니까.

마지막으로 이 코드는 아직 한 단계 개선할 여지가 큰데, 그 다음 버전에서는 **Cross-Attention에 2D 위치정보를 넣어서 “비슷하게 생겼지만 완전히 다른 회로 위치”로 attention이 튀는 현상을 막는 것**까지 넣는 게 좋아. PCB처럼 같은 패턴이 반복되는 영상이라면 그 부분이 꽤 중요해질 수 있어.

[1]: https://huggingface.co/docs/timm/feature_extraction "Feature Extraction · Hugging Face"
[2]: https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in1k?utm_source=chatgpt.com "timm/convnextv2_tiny.fcmae_ft_in1k"
[3]: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html?utm_source=chatgpt.com "torch.nn.functional.scaled_dot_product_attention"
[4]: https://docs.pytorch.org/docs/stable/amp.html?utm_source=chatgpt.com "Automatic Mixed Precision package - torch.amp"
