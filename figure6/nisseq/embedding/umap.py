import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
import umap


def umap_transform(umap_model: umap.UMAP, embeddings_df: pd.DataFrame) -> pd.DataFrame:
    umap_results = umap_model.transform(embeddings_df.to_numpy())

    return pd.DataFrame(umap_results, columns=["UMAP1", "UMAP2"], index=embeddings_df.index)


def umap_fit(embeddings_df: pd.DataFrame, random_state: Optional[int] = 42) -> umap.UMAP:
    if random_state is None:
        umap_model = umap.UMAP(n_components=2)
    else:
        umap_model = umap.UMAP(n_components=2, random_state=random_state)

    embeddings_arr = embeddings_df.to_numpy()
    umap_model.fit(embeddings_arr)

    return umap_model


def umap_fit_transform(embeddings_df: pd.DataFrame) -> tuple[umap.UMAP, pd.DataFrame]:
    umap_model = umap_fit(embeddings_df)
    results_df = umap_transform(umap_model, embeddings_df)
    return umap_model, results_df


def serialize_umap(umap_model: umap.UMAP, save_file: Path):
    with save_file.open("wb") as f:
        pickle.dump(umap_model, f)


def load_umap(save_file: Path):
    with save_file.open("rb") as f:
        return pickle.load(f)  # noqa: S301
