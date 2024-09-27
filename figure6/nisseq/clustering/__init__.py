import string
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from sklearn.cluster import KMeans


def predict_kmeans(kmeans: KMeans, embeddings_df: pd.DataFrame) -> pd.DataFrame:
    clusters = kmeans.predict(embeddings_df)
    return pd.DataFrame({"sample_id": list(embeddings_df.index), "cluster": clusters})


def fit_predict_kmeans(embeddings_df: pd.DataFrame, n_clusters: int) -> tuple[KMeans, pd.DataFrame]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans = kmeans.fit(embeddings_df)
    return kmeans, predict_kmeans(kmeans, embeddings_df)


def kmeans_hierarchy(
    kmeans_model: KMeans,
    save_dir: Path,
    n_clusters: int,
):
    """
    Computes the agglomerative clustering for the provided KMeans clusters,
    names clusters accordingly and saves a dendrogram.
    We compute 12 major clusters.
    """
    # Save files
    dendrogram_save_path = save_dir / f"kmeans_{n_clusters}_dendrogram.png"
    dendrogram_big_save_path = save_dir / f"kmeans_{n_clusters}_dendrogram_big.png"
    cluster_naming_save_file = save_dir / f"kmeans_{n_clusters}_clusters_naming.csv"

    # Colors for dendrogram
    color_list = [f"C{i}" for i in range(1, 10)] + ["orangered", "yellowgreen", "goldenrod"]
    color_to_bigcluster = {color: string.ascii_uppercase[i] for i, color in enumerate(color_list)}
    set_link_color_palette(color_list)

    # Carry out agglomerative clustering on the model
    kmeans_linkage = linkage(kmeans_model.cluster_centers_, method="ward")
    color_threshold = kmeans_linkage[-11, 2]

    # Dendrogram
    _ = plt.figure(figsize=(30, 10))
    dn_full = dendrogram(kmeans_linkage, color_threshold=color_threshold)

    # Cluster naming computation
    big_cluster_dict = {}
    sub_cluster_dict = {}
    full_cluster_name_dict = {}
    dd_cluster_name_dict = {}
    num_subclusters = {}
    all_children = {}

    num_clusters_in_big_clusters = {cluster: 0 for cluster in color_to_bigcluster.values()}

    for i, color in zip(dn_full["leaves"], dn_full["leaves_color_list"], strict=True):
        big_cluster = color_to_bigcluster[color]
        big_cluster_dict[i] = big_cluster

        sub_cluster_number = num_clusters_in_big_clusters[big_cluster]
        sub_cluster_dict[i] = sub_cluster_number
        num_clusters_in_big_clusters[big_cluster] += 1

        full_cluster_name_dict[i] = f"{big_cluster}{sub_cluster_number}"
        dd_cluster_name_dict[i] = full_cluster_name_dict[i]

        num_subclusters[i] = 1
        all_children[i] = [i]

    def get_sub_minmax(a):
        if isinstance(a, int):
            return a, a
        a_str = a.split("-")
        return int(a_str[0]), int(a_str[1])

    def get_big_minmax(a):
        if len(a) == 1:
            return a, a
        a_str = a.split("-")
        return a_str[0], a_str[1]

    def join_sub_clusters(a, b):
        amin, amax = get_sub_minmax(a)
        bmin, bmax = get_sub_minmax(b)
        return f"{min(amin, bmin)}-{max(amax, bmax)}"

    def join_big_clusters(a, b):
        amin, amax = get_big_minmax(a)
        bmin, bmax = get_big_minmax(b)
        bigmin = min(amin, bmin)
        bigmax = max(amax, bmax)
        if bigmin == bigmax:
            return bigmin
        return f"{min(amin, bmin)}-{max(amax, bmax)}"

    for i, linkage_vec in enumerate(kmeans_linkage):
        node_num = i + len(kmeans_model.cluster_centers_)

        num_subclusters[node_num] = num_subclusters[int(linkage_vec[0])] + num_subclusters[int(linkage_vec[1])]
        all_children[node_num] = [node_num] + all_children[int(linkage_vec[0])] + all_children[int(linkage_vec[1])]

        big_cluster_dict[node_num] = join_big_clusters(
            big_cluster_dict[int(linkage_vec[0])], big_cluster_dict[int(linkage_vec[1])]
        )
        sub_cluster_dict[node_num] = join_sub_clusters(
            sub_cluster_dict[int(linkage_vec[0])], sub_cluster_dict[int(linkage_vec[1])]
        )

        if len(big_cluster_dict[node_num]) == 1:
            if num_clusters_in_big_clusters[big_cluster_dict[node_num]] == num_subclusters[node_num]:
                full_cluster_name_dict[node_num] = big_cluster_dict[node_num]
                dd_cluster_name_dict[node_num] = big_cluster_dict[node_num]
            else:
                full_cluster_name_dict[node_num] = f"{big_cluster_dict[node_num]}{sub_cluster_dict[node_num]}"
                dd_cluster_name_dict[node_num] = sub_cluster_dict[node_num]
        else:
            full_cluster_name_dict[node_num] = big_cluster_dict[node_num]
            dd_cluster_name_dict[node_num] = big_cluster_dict[node_num]

    # Actual dendrogram plot
    _ = plt.figure(figsize=(30, 10))
    dn_full = dendrogram(
        kmeans_linkage, color_threshold=color_threshold, leaf_label_func=lambda x: full_cluster_name_dict[x]
    )

    all_links_coords = zip(dn_full["icoord"], dn_full["dcoord"], strict=True)
    all_links_coords = sorted(all_links_coords, key=(lambda x: x[1][1]))

    # Add labels to the links
    for n, (i, d) in enumerate(all_links_coords):
        x = 0.5 * sum(i[1:3])
        y = d[1] - 1.5
        plt.text(x, y, dd_cluster_name_dict[n + len(kmeans_model.cluster_centers_)], va="center", ha="center")

    plt.savefig(dendrogram_save_path)

    _ = plt.figure(figsize=(100, 10))
    dn_full = dendrogram(kmeans_linkage, color_threshold=57, leaf_label_func=lambda x: full_cluster_name_dict[x])

    all_links_coords = zip(dn_full["icoord"], dn_full["dcoord"], strict=True)
    all_links_coords = sorted(all_links_coords, key=(lambda x: x[1][1]))

    # Add labels to the links
    for n, (i, d) in enumerate(all_links_coords):
        x = 0.5 * sum(i[1:3])
        y = d[1] - 1.5
        plt.text(x, y, dd_cluster_name_dict[n + len(kmeans_model.cluster_centers_)], va="center", ha="center")

    plt.savefig(dendrogram_big_save_path)

    # Prepare labels for various levels of clustering
    cluster_name_dict_by_level = {
        0: full_cluster_name_dict,
    }

    for n, _ in enumerate(kmeans_linkage):
        node_num = n + len(kmeans_model.cluster_centers_)
        cluster_name_dict_by_level[n + 1] = cluster_name_dict_by_level[n].copy()

        for subcluster in all_children[node_num]:
            cluster_name_dict_by_level[n + 1][subcluster] = full_cluster_name_dict[node_num]

    df_cluster_naming = pd.DataFrame(
        {f"name{lvl}": [cluster_name_dict_by_level[lvl][i] for i in range(398)] for lvl in range(200)}
    )
    df_cluster_naming[:200].to_csv(cluster_naming_save_file, index=False)
