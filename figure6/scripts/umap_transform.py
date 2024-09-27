from pathlib import Path
from typing import Any

import typer
import umap

from nisseq.config import ROOT_DIR
from nisseq.embedding.embeddings import load_all_embeddings
from nisseq.embedding.umap import load_umap, umap_transform
from nisseq.utils.cli import wrap_command_with_config


def main(config: dict[str, Any]) -> None:
    # Load UMAP model
    embeddings_dir: Path = ROOT_DIR / config["embeddings"]["save_dir"]
    save_file: Path = embeddings_dir / config["umap"]["save_file"]
    umap_model: umap.UMAP = load_umap(save_file)

    # Should we recompute if file is found?
    recompute: bool = config["umap"].get("recompute", True)

    print("Loading embeddings")

    # Load embeddings
    embeddings_df = load_all_embeddings(embeddings_dir)

    # Compute UMAP
    umap_csv = embeddings_dir / "umap.csv"

    # Check if computed already
    if umap_csv.exists() and not recompute:
        print("UMAP embeddings already computed, skipping")
        return

    # Transform embeddings
    print("Transforming embeddings")
    umap_results = umap_transform(umap_model, embeddings_df)
    umap_results.to_csv(umap_csv, index=True)


command = wrap_command_with_config(main)

if __name__ == "__main__":
    typer.run(command)
