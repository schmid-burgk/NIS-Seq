from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, WeightedRandomSampler

DEBUG_RUN_BATCHES = 4


class ProteinImageDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        sample_ids: list[str],
        transform: Callable[[npt.NDArray], torch.Tensor],
        mask_by_mem: bool,
        use_channels: tuple[str, ...],
        sample_labels: Optional[list[int]] = None,
        internal_max_normalization: bool = True,
        max_norm_after_memmask: bool = False,
    ):
        self.images_dir = images_dir
        self.sample_ids = sample_ids
        self.labels = sample_labels
        self.transform = transform
        self.mask_by_mem = mask_by_mem
        self.use_channels = use_channels
        self.internal_max_normalization = internal_max_normalization
        self.max_norm_after_memmask = max_norm_after_memmask

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        sample_id = self.sample_ids[idx]
        if self.labels is not None:
            sample_label = self.labels[idx]
        else:
            sample_label = -1
        image = self.load_image(sample_id)
        image = self.transform(image)
        return image, sample_label

    def load_image(self, sample_id: str) -> npt.NDArray:
        images_channels: dict[str, npt.NDArray] = {}

        for channel in self.use_channels:
            image_path = self.images_dir / f"{sample_id}_{channel}.tif"
            image = np.array(Image.open(image_path)).astype(np.float32)
            images_channels[channel] = image

        if not self.max_norm_after_memmask:
            if self.internal_max_normalization:
                for channel, image in images_channels.items():
                    images_channels[channel] = image / image.max()

        if self.mask_by_mem:
            mask_path = self.images_dir / f"{sample_id}_maskmem.tif"
            mask = np.array(Image.open(mask_path))
            for channel, image in images_channels.items():
                images_channels[channel] = np.where(mask, image, 0)
                # images_channels[channel] = image * mask

        if self.max_norm_after_memmask:
            if self.internal_max_normalization:
                for channel, image in images_channels.items():
                    images_channels[channel] = image / image.max()

        return np.stack([images_channels[channel] for channel in self.use_channels])

    def compute_balanced_weights(self) -> list[float]:
        if self.labels is None:
            raise ValueError("Labels are not provided")

        class_counts = np.bincount(self.labels)
        class_weights = 1 / class_counts
        weights = class_weights[self.labels]
        return [float(x) for x in weights]


def dataloader_from_samples(
    sample_ids: list[str],
    raw_data_directory: Path,
    batch_size: int,
    transforms: Callable[[npt.NDArray], torch.Tensor],
    use_channels: tuple[str, ...],
    sample_labels: Optional[list[int]] = None,
    num_workers: int = 0,
    sampling_strategy: str = "none",
    internal_max_normalization: bool = True,
    debug_run: bool = False,
) -> DataLoader:
    dataset = ProteinImageDataset(
        images_dir=raw_data_directory,
        sample_ids=sample_ids,
        sample_labels=sample_labels,
        transform=transforms,
        mask_by_mem=True,
        use_channels=use_channels,
        internal_max_normalization=internal_max_normalization,
        max_norm_after_memmask=False,
    )

    if sampling_strategy == "balanced":
        samples_weights = dataset.compute_balanced_weights()
        num_samples = len(samples_weights) if not debug_run else DEBUG_RUN_BATCHES * batch_size
        sampler = WeightedRandomSampler(
            weights=samples_weights,
            num_samples=num_samples,
            replacement=True,
        )
        shuffle = None
    elif sampling_strategy == "none" and not debug_run:
        sampler = None
        shuffle = True
    elif sampling_strategy == "none" and debug_run:
        sampler = WeightedRandomSampler(
            weights=[1] * len(dataset),
            num_samples=DEBUG_RUN_BATCHES * batch_size,
            replacement=True,
        )
        shuffle = None
    elif sampling_strategy == "eval" and not debug_run:
        sampler = None
        shuffle = False
    elif sampling_strategy == "eval" and debug_run:
        sampler = SubsetRandomSampler(
            indices=range(DEBUG_RUN_BATCHES * batch_size // 4),
            generator=None,
        )
        shuffle = None
    else:
        raise ValueError(f"Sampling strategy {sampling_strategy} not recognized")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=True,
    )
