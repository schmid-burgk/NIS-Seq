from pathlib import Path
from typing import Any

import pandas as pd

from nisseq.config import DEVICE
from nisseq.eval import eval_model_on_dataloader
from nisseq.model.dataset import dataloader_from_samples
from nisseq.model.model import model_factory
from nisseq.model.transforms import get_transforms


def compute_embeddings_for_model(
    model_config: dict[str, Any],
    data_dir: Path,
    samples_df: pd.DataFrame,
    stats_json_path: Path,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    use_channels: list[str] = model_config["use_channels"]

    # Load model and transforms
    model, input_size = model_factory(num_classes=0, pretrained=model_config["pretrained"])

    # Load transforms
    transforms = get_transforms(
        size=input_size,
        stats_json_path=stats_json_path,
    )

    # Load data
    dataloader = dataloader_from_samples(
        sample_ids=samples_df["sample_id"].tolist(),
        sample_labels=None,
        raw_data_directory=data_dir,
        batch_size=batch_size,
        transforms=transforms,
        num_workers=num_workers,
        sampling_strategy="eval",
        use_channels=tuple(use_channels),
    )

    # Compute embeddings
    outputs = eval_model_on_dataloader(model=model, dataloader=dataloader, device=DEVICE)["outputs"]

    # Create a dataframe with sample ids as indices
    return pd.DataFrame(outputs, index=samples_df["sample_id"])


def load_all_embeddings(dir: Path) -> pd.DataFrame:
    all_embeddings_file = dir / "embeddings.parquet"

    return pd.read_parquet(all_embeddings_file)
