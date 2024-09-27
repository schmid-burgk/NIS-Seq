from pathlib import Path
from typing import Any, Optional

import typer

from nisseq.config import ROOT_DIR
from nisseq.embedding.embeddings import load_all_embeddings
from nisseq.embedding.umap import serialize_umap, umap_fit
from nisseq.utils.cli import wrap_command_with_config


def main(
    config: dict[str, Any],
) -> None:
    embeddings_dir: Path = ROOT_DIR / config["embeddings"]["save_dir"]

    num_samples = config["umap"].get("num_samples", None)
    umap_seed: Optional[int] = config["umap"].get("random_seed", 42)

    save_file: Path = embeddings_dir / config["umap"]["save_file"]

    print("Loading embeddings")

    # Load embeddings
    embeddings_df = load_all_embeddings(embeddings_dir)

    if num_samples is not None:
        embeddings_df = embeddings_df.sample(
            min(num_samples, len(embeddings_df)), random_state=umap_seed if umap_seed is not None else 42
        )

    # Fit UMAP
    print("Fitting UMAP")

    umap_model = umap_fit(embeddings_df, random_state=umap_seed)
    serialize_umap(umap_model, save_file)


command = wrap_command_with_config(main)

if __name__ == "__main__":
    typer.run(command)
