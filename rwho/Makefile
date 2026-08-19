호오BIN = client server
GEN = who_clnt.c who_svc.c who.h
RPCCOM = rpcgen
CC = gcc
CFLAGS = -g -I/usr/include/tirpc
LDFLAGS = -ltirpc

all: $(GEN) $(BIN)

client: who_clnt.o
	$(CC) -o $@ who_clnt.o $(LDFLAGS)

server: who_svc.o
	$(CC) -o $@ who_svc.o $(LDFLAGS)

%.o: %.c who.h
	$(CC) $(CFLAGS) -c $< -o $@

$(GEN): who.x
	$(RPCCOM) who.x

clean:
	rm -f *.o $(BIN)



응. **분류만 목표라면 지금까지 만든 것보다 훨씬 단순하게 가는 게 낫다.** 나는 아래 구조로 시작해볼 것 같아.

```text
Normal ─┐
        ├─ Shared CNN Backbone
Test ───┘
         │
         ├── Mid feature 1 ─ Cross Attention ─ Comparison ─ Pool ─┐
         │                                                       │
         └── Mid feature 2 ─ Cross Attention ─ Comparison ─ Pool ─┤
                                                                 ↓
                                                               MLP
                                                                 ↓
                                                           Normal / Defect
```

Attention weight나 segmentation map은 전혀 만들지 않는다. `F.scaled_dot_product_attention()`만 사용한다. PyTorch는 CUDA에서 SDPA에 적절한 backend를 런타임에 선택할 수 있으므로 직접 attention matrix를 만들 필요가 없다. ([PyTorch Documentation][1])

그리고 timm의 `features_only=True`, `out_indices`를 이용해 중간 feature만 받는다. ([Hugging Face][2])

## 1. Cross-Attention 비교 블록

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionCompare2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_heads: int,
        max_tokens: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.max_tokens = max_tokens
        self.dropout = dropout

        # Normal/Test에 동일하게 사용하는 projection
        self.feature_proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=1,
            bias=False,
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Q/K만 별도 projection
        self.q_proj = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )

        self.k_proj = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )

        # 비교 feature fusion
        #
        # abs diff
        # squared diff
        # element-wise similarity
        #
        # -> embed_dim * 3
        self.compare_fusion = nn.Sequential(
            nn.Conv2d(
                embed_dim * 3,
                embed_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GELU(),

            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GELU(),
        )

    def _limit_tokens(
        self,
        test,
        normal,
    ):
        """
        Attention 계산량이 너무 커지는 것을 막는다.
        """

        _, _, H, W = test.shape

        num_tokens = H * W

        if num_tokens <= self.max_tokens:
            return test, normal

        ratio = math.sqrt(
            self.max_tokens / num_tokens
        )

        new_h = max(
            1,
            int(H * ratio),
        )

        new_w = max(
            1,
            int(W * ratio),
        )

        size = (new_h, new_w)

        test = F.adaptive_avg_pool2d(
            test,
            size,
        )

        normal = F.adaptive_avg_pool2d(
            normal,
            size,
        )

        return test, normal

    def _split_heads(self, x):
        """
        [B, N, C]
            ->
        [B, heads, N, head_dim]
        """

        B, N, C = x.shape

        x = x.reshape(
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
        [B, N, C]
        """

        B, H, N, D = x.shape

        x = (
            x.transpose(1, 2)
            .contiguous()
        )

        return x.reshape(
            B,
            N,
            H * D,
        )

    def forward(
        self,
        test_feature,
        normal_feature,
    ):
        # -----------------------------------------
        # 1. Shared latent space
        # -----------------------------------------

        test = self.feature_proj(
            test_feature
        )

        normal = self.feature_proj(
            normal_feature
        )

        # token 수 제한
        test, normal = self._limit_tokens(
            test,
            normal,
        )

        B, C, H, W = test.shape

        # -----------------------------------------
        # 2. CNN -> Token
        # -----------------------------------------

        test_tokens = (
            test.flatten(2)
            .transpose(1, 2)
        )

        normal_tokens = (
            normal.flatten(2)
            .transpose(1, 2)
        )

        test_tokens = self.norm(
            test_tokens
        )

        normal_tokens = self.norm(
            normal_tokens
        )

        # -----------------------------------------
        # 3. Q K V
        # -----------------------------------------

        q = self.q_proj(
            test_tokens
        )

        k = self.k_proj(
            normal_tokens
        )

        # V에는 normal feature 자체 사용
        v = normal_tokens

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # -----------------------------------------
        # 4. Cross Attention
        # -----------------------------------------

        matched_normal = (
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=(
                    self.dropout
                    if self.training
                    else 0.0
                ),
            )
        )

        matched_normal = self._merge_heads(
            matched_normal
        )

        # -----------------------------------------
        # 5. Difference
        # -----------------------------------------

        difference = (
            test_tokens
            - matched_normal
        )

        abs_diff = difference.abs()

        squared_diff = difference.square()

        # 비슷하면 같은 방향/부호의 activation
        similarity = (
            test_tokens
            * matched_normal
        )

        # -----------------------------------------
        # Auxiliary MSE용 distance
        # -----------------------------------------

        # [B, N, C]
        token_mse = squared_diff.mean(
            dim=-1
        )

        # [B]
        feature_distance = token_mse.mean(
            dim=-1
        )

        # -----------------------------------------
        # CNN 형태로 복구
        # -----------------------------------------

        def to_map(x):
            return (
                x.transpose(1, 2)
                .reshape(B, C, H, W)
            )

        abs_diff = to_map(abs_diff)
        squared_diff = to_map(squared_diff)
        similarity = to_map(similarity)

        # -----------------------------------------
        # 6. Comparison feature
        # -----------------------------------------

        comparison = torch.cat(
            [
                abs_diff,
                squared_diff,
                similarity,
            ],
            dim=1,
        )

        comparison = self.compare_fusion(
            comparison
        )

        # -----------------------------------------
        # 7. Classification용 pooling
        # -----------------------------------------

        avg_feature = (
            F.adaptive_avg_pool2d(
                comparison,
                1,
            )
            .flatten(1)
        )

        max_feature = (
            F.adaptive_max_pool2d(
                comparison,
                1,
            )
            .flatten(1)
        )

        pooled = torch.cat(
            [
                avg_feature,
                max_feature,
            ],
            dim=1,
        )

        return pooled, feature_distance
```

여기서 핵심적으로 나온 건 두 개야.

```python
pooled
# [B, embed_dim * 2]

feature_distance
# [B]
```

`pooled`는 최종 classifier로 보내고, `feature_distance`는 원하면 auxiliary MSE loss에 사용한다.

참고로 SDPA의 `dropout_p`는 평가 모드라고 자동으로 0이 되는 게 아니므로 코드처럼 `self.training`에 따라 직접 `0.0`을 넘기는 게 맞다. ([PyTorch Documentation][1])

---

# 2. 전체 모델

ConvNeXtV2를 예로 들겠다.

나는 마지막 stage보다는 **중간 feature 2개**를 쓰겠다.

```python
import timm


class NormalDefectCrossAttentionClassifier(nn.Module):
    def __init__(
        self,
        backbone_name="convnextv2_tiny.fcmae_ft_in1k",
        pretrained=True,

        out_indices=(1, 2),

        embed_dims=(128, 256),
        num_heads=(4, 8),

        max_tokens=1024,

        classifier_dim=256,
        dropout=0.2,
    ):
        super().__init__()

        assert (
            len(out_indices)
            == len(embed_dims)
            == len(num_heads)
        )

        # =========================================
        # Backbone
        # =========================================

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

        channels = (
            self.backbone
            .feature_info
            .channels()
        )

        print(
            "Feature channels:",
            channels,
        )

        # =========================================
        # Cross Attention blocks
        # =========================================

        self.compare_blocks = nn.ModuleList()

        for in_ch, dim, heads in zip(
            channels,
            embed_dims,
            num_heads,
        ):
            block = CrossAttentionCompare2d(
                in_channels=in_ch,
                embed_dim=dim,
                num_heads=heads,
                max_tokens=max_tokens,
            )

            self.compare_blocks.append(
                block
            )

        # 각각 avg + max가 있으므로 * 2
        classifier_input_dim = sum(
            dim * 2
            for dim in embed_dims
        )

        # =========================================
        # Classification Head
        # =========================================

        self.classifier = nn.Sequential(
            nn.LayerNorm(
                classifier_input_dim
            ),

            nn.Linear(
                classifier_input_dim,
                classifier_dim,
            ),
            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                classifier_dim,
                1,
            ),
        )

    def forward(
        self,
        normal_image,
        test_image,
    ):
        B = normal_image.size(0)

        # =========================================
        # Backbone을 한 번만 호출
        # =========================================
        #
        # normal/test를 batch에 합쳤다가
        # feature 추출 후 다시 분리
        #
        # [2B, 3, H, W]

        pair = torch.cat(
            [
                normal_image,
                test_image,
            ],
            dim=0,
        )

        features = self.backbone(
            pair
        )

        pooled_features = []
        feature_distances = []

        # =========================================
        # Multi-scale Cross Attention
        # =========================================

        for feature, compare_block in zip(
            features,
            self.compare_blocks,
        ):
            normal_feature = feature[:B]
            test_feature = feature[B:]

            pooled, distance = compare_block(
                test_feature,
                normal_feature,
            )

            pooled_features.append(
                pooled
            )

            feature_distances.append(
                distance
            )

        # =========================================
        # Scale fusion
        # =========================================

        x = torch.cat(
            pooled_features,
            dim=1,
        )

        # =========================================
        # Classification
        # =========================================

        logits = (
            self.classifier(x)
            .squeeze(1)
        )

        # [B, number_of_scales]
        distances = torch.stack(
            feature_distances,
            dim=1,
        )

        return {
            "logits": logits,
            "distances": distances,
        }
```

`timm`의 feature backbone은 이런 식으로 여러 hierarchy level의 feature를 반환하도록 사용할 수 있다. ([Hugging Face][2])

---

# 3. 실제 forward

```python
device = torch.device("cuda")

model = NormalDefectCrossAttentionClassifier(
    backbone_name="convnextv2_tiny.fcmae_ft_in1k",
    pretrained=True,

    out_indices=(1, 2),

    embed_dims=(128, 256),
    num_heads=(4, 8),

    max_tokens=1024,
)

model = model.to(device)
```

그리고:

```python
normal = torch.randn(
    8,
    3,
    224,
    224,
    device=device,
)

test = torch.randn(
    8,
    3,
    224,
    224,
    device=device,
)

output = model(
    normal,
    test,
)

print(
    output["logits"].shape
)
# [8]

print(
    output["distances"].shape
)
# [8, 2]
```

추론은:

```python
probability = torch.sigmoid(
    output["logits"]
)
```

끝.

---

# 4. Loss는 이렇게 하겠다

여기서는

```text
0 = 정상
1 = 불량
```

이라고 하자.

분류 loss는 그냥 BCE.

그리고 중간 Cross-Attention feature에 MSE auxiliary loss를 넣는데, **정상 pair에만 적용한다.**

```python
def compute_loss(
    output,
    label,
    aux_weight=0.05,
):
    label = label.float()

    # =====================================
    # Classification loss
    # =====================================

    cls_loss = (
        F.binary_cross_entropy_with_logits(
            output["logits"],
            label,
        )
    )

    # =====================================
    # Feature auxiliary loss
    # =====================================

    # scale 평균
    #
    # [B, 2]
    # ->
    # [B]

    distance = (
        output["distances"]
        .mean(dim=1)
    )

    normal_mask = (
        label < 0.5
    )

    if normal_mask.any():
        feature_loss = (
            distance[normal_mask]
            .mean()
        )
    else:
        # graph를 유지하면서 zero
        feature_loss = (
            output["logits"].sum()
            * 0.0
        )

    # =====================================
    # Total
    # =====================================

    loss = (
        cls_loss
        + aux_weight * feature_loss
    )

    return {
        "loss": loss,
        "cls_loss": cls_loss,
        "feature_loss": feature_loss,
    }
```

즉 loss는

[
L =
L_{\text{BCE}}
+
\lambda L_{\text{normal-MSE}}
]

이고,

[
L_{\text{normal-MSE}}
=====================

\operatorname{MSE}
(F_{test}, F^{matched}_{normal})
]

야.

### 왜 불량에는 MSE margin을 안 걸었냐

이건 의도적이야.

작은 결함이라면 이미지 전체 feature 중에서 실제로 달라야 하는 영역이 극히 일부일 수 있어.

그런데

```python
defect MSE > margin
```

을 전체 feature에 강제로 걸어버리면 모델이

> "불량 이미지라면 전체 feature를 정상과 다르게 만들어야겠네."

라고 학습할 위험이 있어.

그건 작은 결함 검출에서는 원하지 않는 동작이야.

그래서 일단은:

```text
정상
→ Cross-Attention 후 feature가 잘 맞도록 MSE

불량
→ 별도의 MSE 제약 없음
→ BCE가 필요한 차이를 찾아내게 함
```

으로 가는 걸 추천해.

---

# 5. 왜 `avg + max pooling` 둘 다 쓰냐

이 구조에서는 이 부분이 꽤 중요해.

예를 들어 전체 비교 feature가

```text
28 × 28
```

인데 결함이 겨우 몇 pixel/token에만 나타났다고 하자.

Global Average Pooling만 하면

```text
엄청 큰 defect response
       ↓
전체 784개 위치 평균
       ↓
희석
```

될 수 있어.

반면 max pooling은

```text
"가장 수상한 위치가 얼마나 강한가?"
```

를 가져간다.

그래서:

```python
avg_feature
max_feature
```

를 concatenate했다.

**작은 결함 분류라는 조건에서는 이 부분은 그대로 두는 걸 추천해.**

---

# 6. 학습은 처음에 이렇게

기존에 학습해둔 backbone을 가져오는 상황이라면 처음에는 backbone을 freeze하는 게 안전해.

```python
for p in model.backbone.parameters():
    p.requires_grad = False
```

그리고:

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

새로운 Cross-Attention과 classifier를 먼저 학습한다.

그다음 backbone을 풀어서:

```python
for p in model.backbone.parameters():
    p.requires_grad = True
```

```python
optimizer = torch.optim.AdamW(
    [
        {
            "params": model.backbone.parameters(),
            "lr": 1e-5,
        },
        {
            "params": model.compare_blocks.parameters(),
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

정도로 fine-tuning하는 식.

---

## 이번 구조가 앞의 모델보다 내가 더 추천하는 이유

이제 목적이 **순수하게 정상/불량 분류**라면 localization head는 학습을 복잡하게만 만들 가능성이 있어.

이번 모델은 사실 상당히 명확해.

```text
                 같은 CNN
                   │
        ┌──────────┴──────────┐
        ↓                     ↓
    정상 feature          검사 feature
        │                     │
        │                 Query
        │                     │
      Key/Value ───── Cross Attention
                              │
                              ↓
                      Matched Normal
                              │
            ┌─────────────────┴──────────────┐
            │                                │
          |차이|                           차이²
            │                                │
            └──────────────┬─────────────────┘
                           ↓
                      Conv Fusion
                           ↓
                    AVG Pool + MAX Pool
                           ↓
                       Classifier
                           ↓
                     정상 / 불량
```

그리고 개인적으로는 **첫 실험에서 `aux_weight=0`도 반드시 같이 해볼 것** 같아.

즉,

```python
loss = BCE
```

만으로 먼저 학습해서 Cross-Attention comparison 자체가 분류에 도움이 되는지 보고,

그다음

```python
loss = BCE + 0.05 * normal_MSE
```

를 비교하는 거야.

MSE는 이 모델에 반드시 필요한 요소가 아니야. **Cross-Attention과 difference까지 모두 BCE gradient만으로 end-to-end 학습 가능**하다. MSE는 정상 feature correspondence를 조금 더 강제하는 auxiliary regularization이라고 보는 게 정확해. ([PyTorch Documentation][1])

그리고 네 데이터처럼 작은 결함이 중요한 경우라면, 나는 처음부터 MSE를 강하게 주기보다 **`0 → 0.01 → 0.05` 정도를 비교하면서** Recall이 어떻게 움직이는지 보는 쪽을 택하겠다.

[1]: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html?utm_source=chatgpt.com "torch.nn.functional.scaled_dot_product_attention"
[2]: https://huggingface.co/docs/timm/feature_extraction?utm_source=chatgpt.com "Feature Extraction"


















맞아. 이제 **정상도 하나의 클래스**로 두고, 불량 유형들이 각각 별도 클래스라면 `BCEWithLogitsLoss`가 아니라 **`CrossEntropyLoss`**로 가는 게 자연스러워.

예를 들어 클래스가:

```text
0 = Normal
1 = Scratch
2 = Particle
3 = Stain
4 = Crack
```

이라면 모델 출력은 `[B, 5]`가 되고, label은 `[B]`의 정수형 클래스 인덱스가 돼.

가장 먼저 classifier 마지막 층을 바꾸면 돼.

```python
class NormalDefectCrossAttentionClassifier(nn.Module):
    def __init__(
        self,
        backbone_name="convnextv2_tiny.fcmae_ft_in1k",
        pretrained=True,
        num_classes=5,

        out_indices=(1, 2),
        embed_dims=(128, 256),
        num_heads=(4, 8),

        max_tokens=1024,
        classifier_dim=256,
        dropout=0.2,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

        channels = self.backbone.feature_info.channels()

        self.compare_blocks = nn.ModuleList()

        for in_ch, dim, heads in zip(
            channels,
            embed_dims,
            num_heads,
        ):
            self.compare_blocks.append(
                CrossAttentionCompare2d(
                    in_channels=in_ch,
                    embed_dim=dim,
                    num_heads=heads,
                    max_tokens=max_tokens,
                )
            )

        classifier_input_dim = sum(
            dim * 2
            for dim in embed_dims
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),

            nn.Linear(
                classifier_input_dim,
                classifier_dim,
            ),
            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                classifier_dim,
                num_classes,
            ),
        )

    def forward(
        self,
        normal_image,
        test_image,
    ):
        B = normal_image.size(0)

        pair = torch.cat(
            [normal_image, test_image],
            dim=0,
        )

        features = self.backbone(pair)

        pooled_features = []
        feature_distances = []

        for feature, compare_block in zip(
            features,
            self.compare_blocks,
        ):
            normal_feature = feature[:B]
            test_feature = feature[B:]

            pooled, distance = compare_block(
                test_feature,
                normal_feature,
            )

            pooled_features.append(pooled)
            feature_distances.append(distance)

        x = torch.cat(
            pooled_features,
            dim=1,
        )

        logits = self.classifier(x)

        distances = torch.stack(
            feature_distances,
            dim=1,
        )

        return {
            "logits": logits,
            "distances": distances,
        }
```

이제:

```python
output["logits"].shape
```

는

```text
[B, num_classes]
```

가 돼.

예를 들어 batch가 16, class가 5개면:

```text
[16, 5]
```

---

## Loss는 CE로

기본 classification loss는 아주 단순해.

```python
cls_loss = F.cross_entropy(
    output["logits"],
    label,
)
```

중요한 건 `label`이 float가 아니라 **LongTensor**여야 한다는 거야.

```python
label = label.to(
    device,
    dtype=torch.long,
)
```

예:

```python
label
# tensor([0, 3, 1, 0, 4, 2, ...])
```

---

## 그럼 기존 normal MSE auxiliary loss는?

그대로 사용할 수 있어.

정상 class를 `0`으로 두었다고 하면:

```python
normal_mask = (label == 0)
```

으로 정상 sample만 골라서 Cross-Attention 후 feature distance를 작게 만들면 돼.

```python
def compute_loss(
    output,
    label,
    normal_class=0,
    aux_weight=0.05,
):
    # ---------------------------------
    # Multi-class classification
    # ---------------------------------

    cls_loss = F.cross_entropy(
        output["logits"],
        label,
    )

    # ---------------------------------
    # Cross-Attention feature distance
    # ---------------------------------

    # [B, num_scales] -> [B]
    distance = (
        output["distances"]
        .mean(dim=1)
    )

    normal_mask = (
        label == normal_class
    )

    if normal_mask.any():
        feature_loss = (
            distance[normal_mask]
            .mean()
        )
    else:
        feature_loss = (
            output["logits"].sum()
            * 0.0
        )

    loss = (
        cls_loss
        + aux_weight * feature_loss
    )

    return {
        "loss": loss,
        "cls_loss": cls_loss,
        "feature_loss": feature_loss,
    }
```

그래서 전체 loss는:

[
L =
CE(y,\hat y)
+
\lambda D_{\text{normal}}
]

가 돼.

---

## 왜 불량 종류별로 MSE를 따로 안 거냐?

여기서 CE가 그 역할을 해.

Cross-Attention에서 만들어지는 comparison feature가 예를 들어:

```text
정상
→ 정상 reference와 거의 동일

Scratch
→ 얇고 길게 차이

Particle
→ 작은 국소 차이

Stain
→ 넓고 약한 차이

Crack
→ 구조적인 차이
```

처럼 서로 다른 패턴을 갖게 되고,

그걸 마지막 classifier가:

```text
comparison feature
        ↓
      MLP
        ↓
Normal / Scratch / Particle / Stain / Crack
```

로 구분하도록 CE가 학습시킨다.

즉 Cross-Attention은 주로

> **정상 기준에서 무엇이 다른가?**

를 계산하고,

CE는

> **그 차이가 어떤 종류의 불량인가?**

를 학습하는 셈이야.

이 역할 분리가 꽤 깔끔해.

---

## 추론도 조금 달라져

BCE 때는 sigmoid였지만 이제는 `softmax`야.

```python
output = model(
    normal_image,
    test_image,
)

prob = torch.softmax(
    output["logits"],
    dim=1,
)

pred = prob.argmax(
    dim=1,
)
```

예를 들어:

```python
prob[0]
```

가

```text
Normal    0.02
Scratch   0.03
Particle  0.91
Stain     0.02
Crack     0.02
```

라면:

```python
pred[0]
# 2
```

가 되는 거지.

---

## 한 가지 더: 정상 검출을 중요하게 본다면

다중 클래스에서는 이런 문제가 생길 수 있어.

```text
실제 Particle
→ Scratch로 예측
```

은 불량 종류는 틀렸지만 어쨌든 **불량으로는 잡은 것**이잖아.

반면

```text
실제 Particle
→ Normal
```

은 네 목적에서는 훨씬 심각한 오류일 수 있어.

그래서 나는 학습 자체는 CE로 하되, 평가 metric은 두 층으로 나눠 볼 것 같아.

```text
1. Normal vs Any Defect Recall
2. Defect subtype accuracy / macro F1
```

즉 5-class CE를 쓰면서도:

```python
is_defect_true = label != 0
is_defect_pred = pred != 0
```

로 별도의 binary defect recall을 계산하는 거야.

그리고 필요하면 CE 자체에도 class weight를 줄 수 있어.

예를 들어 정상으로 놓치는 걸 더 강하게 벌주고 싶다면:

```python
weights = torch.tensor(
    [
        1.0,  # Normal
        2.0,  # Scratch
        2.0,  # Particle
        2.0,  # Stain
        2.0,  # Crack
    ],
    device=device,
)

cls_loss = F.cross_entropy(
    output["logits"],
    label,
    weight=weights,
)
```

처럼 defect class 전체의 weight를 높일 수도 있어.

다만 **처음부터 weight를 넣지는 않고**, 먼저 plain CE로 confusion matrix를 보고 `불량 → 정상` 오류가 실제로 많은지 확인한 뒤 넣는 걸 추천해.

정리하면, 이제 구조는 이렇게 보면 돼:

```text
정상 Reference + 검사 이미지
          ↓
   Shared Backbone
          ↓
   Cross Attention
          ↓
 Normal과의 차이 feature
          ↓
     Multi-scale Pool
          ↓
        MLP
          ↓
 ┌────────┼───────────────┐
Normal  Scratch  Particle  ...
```

그리고 loss는 우선

```python
loss = CE + 0.05 * NormalFeatureMSE
```

이 정도가 가장 무난한 시작점이야.

특히 **정상 이미지를 class 0으로 고정**해두면 auxiliary loss나 defect-vs-normal metric 계산도 굉장히 편해져.

