# prettyProteins

To reproduce or investigate results used to produce `figure6`, start by cloning this repo using SSH:
```
git clone https://github.com/schmid-burgk/NIS-Seq.git
cd NIS-Seq/figure6
```

## Data

### Download the embeddings, clustering, UMAP

The clustering and UMAP-transformed embeddings are available in this repository in the `embeddings` folder.

To obtain embeddings and the fitted UMAP model, download `embeddings.zip` from [here (will be shared via Zenodo)]() and unzip it in this directory.


### Download the raw data

Download the `nisseq_hela_data.zip` from [here (will be shared via Zenodo)]().

Then unzip it into `data/` using `unzip nisseq_hela_data.zip`, or
```bash
ripunzip unzip-file nisseq_hela_data.zip
```
if you have `ripunzip` installed.

All images should be placed in `data/nisseq/hela_data` folder,
or the value of `images_path` can be changed in `config.yaml` accordingly.

Note: unzipped image files will take about `250GB` of disk space.


#### ripunzip

Unzipping the images takes quite a long time, you can install `ripunzip` to speed it up by using multiple cores. For example, on a Debian-based machine, run:
```bash
curl https://sh.rustup.rs -sSf | sh
```
then restart your shell and execute:
```bash
sudo apt install libssl-dev
cargo install ripunzip
```



## Using the repository

To run scripts within this repository, create a conda environment and install required packages:
```bash
conda create -n nisseq python=3.10
conda activate nisseq
pip install .
```

The steps to reproduce all of the results are outlined below.


### Reproduce embeddings, clustering, UMAP

Relying on pre-processed data in the `metadata` folder,
after you [download the data](#download-the-raw-data),
you can run:
```bash
python3 scripts/compute_embeddings.py
python3 scripts/clustering.py
python3 scripts/umap_fit.py
python3 scripts/umap_transform.py
```
to reproduce the full pipeline (the embeddings, clustering and UMAP).


### Compute embeddings

You can [download precomputed embeddings](#download-the-embeddings-clustering-umap).

If you wish to instead reproduce the embeddings, run:
```bash
python scripts/compute_embeddings.py
```

The embeddings will be saved to `embeddings/embeddings.parquet` file.


### Compute clustering

To compute K-Means clusters and agglomerative clustering using embeddings (either [downloaded](#download-the-embeddings-clustering-umap) or [computed](#compute-embeddings)), run
```bash
python scripts/clustering.py
```
and the clustering results will be saved to `embeddings/clustering`.


### Fit/compute UMAP

The UMAP model can be retrieved by [downloading precomputed embeddings](#download-the-model-and-embeddings),
and by default can be found at `embeddings/umap.pkl`.

If you wish to reproduce fitting the model, run
```bash
python scripts/umap_fit.py
```
The fitted UMAP model will be saved to `embeddings/umap.pkl`.

If you wish to reproduce the UMAP embeddings, run
```bash
python scripts/umap_transform.py
```
The UMAP-transformed embeddings will be saved to `embeddings/umap.csv`.


### Using a manually specified config

You might want to copy `config.yaml` and edit it.
You can then run any of the above scripts using the new config by adding
`--config-file PATH_TO_NEW_CONFIG`.


## Repository structure

The repository has the following folder structure:
- `data`: folder for all the data.
- `embeddings`: embeddings are saved here.
  - `clustering`: contains results of clustering.
- `nisseq`: the main package.
  - `clustering`: carrying out clustering computations.
  - `embedding`: contains submodules for computing embeddings using the model, training and evaluating UMAP, finding nearest neighbours.
  - `eval`: in development, supposed to contain metrics for quality of location separation.
  - `model`: loading data and computer vision models.
  - `utils`: various utility functions.
- `scripts`: scripts for executing the full pipeline of data preparation, training, computing embeddings etc.
- `config.yaml`: the config file for controling the full pipeline.
- `README.md`: this file.
- `pyproject.toml`: pyproject file for installing the package and reproducing the environment.
- `.gitignore`: gitignore file.



# LICENSE

This work is shared under 
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
