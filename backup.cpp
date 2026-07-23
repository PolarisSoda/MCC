#include <bits/stdc++.h>

using namespace std;

int main() {
	int* a[10]; int* b[10];
}
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

import torch
from torch import nn


def make_convnextv2_groupwise_param_groups(
    model: nn.Module,
    *,
    backbone: nn.Module | None = None,
    backbone_prefix: str = "",
    layer_decay: float = 0.6,
    weight_decay: float = 0.05,
    blocks_per_group: int = 3,
) -> list[dict[str, Any]]:
    """
    timm ConvNeXt/ConvNeXtV2용 group-wise layer decay.

    Group 구성:
      0                    : stem
      1 ... N              : 연속된 ConvNeXt blocks
      N + 1                : final norm / classifier / custom head

    wrapper 없이 timm 모델 자체를 넘길 때:
        model=model
        backbone=None
        backbone_prefix=""

    wrapper가 다음과 같을 때:
        self.backbone = timm.create_model(...)
        self.head = ...

    다음처럼 호출:
        model=full_model
        backbone=full_model.backbone
        backbone_prefix="backbone."
    """
    if not 0.0 < layer_decay <= 1.0:
        raise ValueError("layer_decay must be in (0, 1].")

    if blocks_per_group <= 0:
        raise ValueError("blocks_per_group must be positive.")

    backbone = model if backbone is None else backbone

    if not hasattr(backbone, "stages"):
        raise TypeError("backbone must be a timm ConvNeXt model with .stages.")

    depths = [len(stage.blocks) for stage in backbone.stages]

    # 각 stage의 첫 번째 LR group ID
    # Base/Large [3, 3, 27, 3]이면 [1, 2, 3, 12]
    stage_first_group: list[int] = []
    next_group_id = 1

    for depth in depths:
        stage_first_group.append(next_group_id)
        next_group_id += math.ceil(depth / blocks_per_group)

    # stem=0, block groups=1..N, final norm/head=N+1
    head_group_id = next_group_id
    max_group_id = head_group_id

    def remove_backbone_prefix(name: str) -> str | None:
        if not backbone_prefix:
            return name

        if name.startswith(backbone_prefix):
            return name[len(backbone_prefix):]

        # backbone 바깥의 custom head
        return None

    def get_layer_id(full_name: str) -> int:
        name = remove_backbone_prefix(full_name)

        # wrapper의 custom head 등
        if name is None:
            return head_group_id

        if name.startswith("stem."):
            return 0

        downsample_match = re.match(
            r"^stages\.(\d+)\.downsample(?:\.|$)",
            name,
        )
        if downsample_match:
            stage_id = int(downsample_match.group(1))

            if not 0 <= stage_id < len(depths):
                raise ValueError(f"Invalid stage ID in parameter: {full_name}")

            # 해당 stage의 첫 block group과 동일한 LR
            return stage_first_group[stage_id]

        block_match = re.match(
            r"^stages\.(\d+)\.blocks\.(\d+)(?:\.|$)",
            name,
        )
        if block_match:
            stage_id = int(block_match.group(1))
            block_id = int(block_match.group(2))

            if not 0 <= stage_id < len(depths):
                raise ValueError(f"Invalid stage ID in parameter: {full_name}")

            if not 0 <= block_id < depths[stage_id]:
                raise ValueError(f"Invalid block ID in parameter: {full_name}")

            return (
                stage_first_group[stage_id]
                + block_id // blocks_per_group
            )

        # head.norm, head.fc, norm_pre, 외부 custom head 등
        return head_group_id

    grouped_params: dict[tuple[int, bool], dict[str, Any]] = {}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        layer_id = get_layer_id(name)
        lr_scale = layer_decay ** (max_group_id - layer_id)

        no_decay = (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or ".grn." in name
        )

        key = (layer_id, no_decay)

        if key not in grouped_params:
            grouped_params[key] = {
                "params": [],
                "weight_decay": 0.0 if no_decay else weight_decay,
                "lr_scale": lr_scale,
                "layer_id": layer_id,
                "param_names": [],
            }

        grouped_params[key]["params"].append(parameter)
        grouped_params[key]["param_names"].append(name)

    return [
        grouped_params[key]
        for key in sorted(grouped_params)
    ]
