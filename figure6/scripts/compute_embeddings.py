from pathlib import Path
from typing import Any

import typer

from nisseq.config import ROOT_DIR
from nisseq.embedding.embeddings import compute_embeddings_for_model
from nisseq.utils.cli import load_csv_or_parquet, save_csv_or_parquet, wrap_command_with_config


def main(
    config: dict[str, Any],
) -> None:
    model_config: dict[str, Any] = config["model"]
    config_embedding: dict[str, Any] = config["embeddings"]

    # Retrieve the arguments
    save_dir: Path = ROOT_DIR / config_embedding["save_dir"]
    recompute: bool = config_embedding.get("recompute", False)
    debug_run: bool = config_embedding.get("debug_run", False)

    # Retrieve optional parameters
    batch_size: int = config_embedding["batch_size"]
    num_workers: int = config_embedding["num_workers"]

    # Get data config
    data_config: dict[str, Any] = config["data"]
    raw_data_directory: Path = ROOT_DIR / data_config["images_path"]
    samples_dataframe: Path = ROOT_DIR / data_config["samples_df_path"]
    stats_json_path_str: str = data_config["stats_json_path"]
    stats_json_path: Path = ROOT_DIR / stats_json_path_str

    # Load the samples dataframe
    samples_df = load_csv_or_parquet(samples_dataframe)

    if debug_run:
        samples_df = samples_df[: min(1000, len(samples_df))]

    output_file = save_dir / "embeddings.parquet"

    # If embeddings have been computed, skip
    if output_file.exists() and not recompute:
        print("Embeddings already computed, skipping")
        return

    # Compute the embeddings
    embeddings = compute_embeddings_for_model(
        model_config=model_config,
        data_dir=raw_data_directory,
        samples_df=samples_df,
        stats_json_path=stats_json_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Save embeddings
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_csv_or_parquet(embeddings, output_file, index=True)


command = wrap_command_with_config(main)

if __name__ == "__main__":
    typer.run(command)
