import pickle
from pathlib import Path
from typing import Any

import typer

from nisseq.clustering import (
    fit_predict_kmeans,
    kmeans_hierarchy,
    predict_kmeans,
)
from nisseq.config import ROOT_DIR
from nisseq.embedding.embeddings import load_all_embeddings
from nisseq.utils.cli import wrap_command_with_config


def main(config: dict[str, Any]) -> None:
    config_embedding: dict[str, Any] = config["embeddings"]
    config_clustering: dict[str, Any] = config["clustering"]

    embeddings_dir: Path = ROOT_DIR / config_embedding["save_dir"]

    n_clusters: int = config_clustering["n_clusters"]

    # Load embeddings
    embeddings_df = load_all_embeddings(embeddings_dir)

    clustering_dir = embeddings_dir / "clustering"
    clustering_dir.mkdir(exist_ok=True)

    # Check if the file already exists
    save_file = clustering_dir / f"kmeans_{n_clusters}_clusters.pkl"
    save_file_clusters = clustering_dir / f"kmeans_{n_clusters}_clusters.csv"
    if save_file.exists() and save_file_clusters.exists():
        print(f"Kmeans with {n_clusters} clusters already computed, reusing for inference")
        kmeans = pickle.load(save_file.open("rb"))  # noqa: S301

        kmeans_clusters = predict_kmeans(kmeans, embeddings_df)
    else:
        # Actually compute the kmeans
        print(f"Computing kmeans with {n_clusters} clusters")
        kmeans, kmeans_clusters = fit_predict_kmeans(embeddings_df, n_clusters)

        # Save model
        with save_file.open("wb") as f:
            pickle.dump(kmeans, f)

    # Save clusters
    kmeans_clusters.to_csv(save_file_clusters, index=False)

    # Prepare hierarchical clustering, its dendrogram and human-friendly cluster names
    kmeans_hierarchy(kmeans, clustering_dir, n_clusters)


command = wrap_command_with_config(main)

if __name__ == "__main__":
    typer.run(command)
