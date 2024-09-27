import json
from collections.abc import Callable
from pathlib import Path

import numpy.typing as npt
import torch
from torchvision.transforms import v2

from nisseq.utils.functions import arr_to_tensor


def get_resize(size: int) -> list[v2.Transform]:
    return [
        v2.Resize(size),
        v2.CenterCrop(size),
    ]


def get_val_transforms(
    normalization: v2.Transform,
    resize_to: int,
) -> Callable[[npt.NDArray], torch.Tensor]:
    resize = get_resize(resize_to)
    return v2.Compose([arr_to_tensor, normalization, *resize])


def get_normalization(
    stats_json_path: Path,
) -> v2.Transform:
    with stats_json_path.open("r") as f:
        stats = json.load(f)
        return v2.Normalize(mean=[stats["mean"]], std=[stats["std"]])


def get_transforms(
    size: int,
    stats_json_path: Path,
) -> Callable[[npt.NDArray], torch.Tensor]:
    return get_val_transforms(
        normalization=get_normalization(stats_json_path=stats_json_path),
        resize_to=size,
    )
