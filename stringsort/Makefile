all:
	g++ -g -O2 -o seqsort sequential_sort.cpp
	g++ -g -o textgen text_gen.cpp

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
