# NIS-Seq

This repository contains the analysis files for figures presented in [*Cell Type-Agnostic Optical Perturbation Screening Using Nuclear In-Situ Sequencing (NIS-Seq)*](https://www.biorxiv.org/content/10.1101/2024.01.18.576210v1).

## Image analysis pipeline

Raw image analysis files are located [*here*](./image_analysis/). For a detailed description how to use the image analysis tools, check [*Supplementary Protocol 3*](https://www.biorxiv.org/content/10.1101/2024.01.18.576210v1).

#### Short description:

The image analysis section provides tools for processing raw images from up to 14 NIS-Seq cycles. The tools align images using FFT-accelerated cross-correlation of nuclear staining, detect spots by summing sequencing channels from the first three cycles, and apply high-pass filtering, local maximum detection, and brightness thresholding. Sequencing information for each spot is aggregated over 5x5 pixels, followed by high-pass filtering and removal of negative values. Channel unmixing is achieved by multiplying the channel vector of each cycle by the inverse matrix of average base-wise intensities. Non-G bases are identified by the maximum of unmixed channels, while Gs are identified when all unmixed intensities are less than 20% of the maximum spot intensity across all cycles. The tools match sequences to a known dictionary (Brunello sgRNA sequences, reverse complemented), allowing for zero or one mismatch, and assign sequences to nuclei using [*Cellpose's*](https://doi.org/10.1038/s41592-020-01018-x) [*(Github)*](https://github.com/MouseLand/cellpose) "nuclei" model, with strict criteria for sequence dominance and signal intensity thresholds.

## Further analysis
#### The Jupyter Notebooks provided here were used to generate the following figures:

- [**Figure 1E**](./figure1/NIS_Figure1E.ipynb): Cell type panel with comparing mapped nuclei between *NIS-Seq* and [*Optical Pooled Screens*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6886477/) [*(Github)*](https://github.com/feldman4/OpticalPooledScreens_2019).
- [**Figure 1F**](./figure1/NIS_Figure1F.ipynb): Mixing experiment (GFP positive cells and NIS-barcode containing cells) to determine Specificity and Sensitivity of the method for PFA fixation and methanol fixation.
- [**Figure 2A**](./figure2/NIS_Figure2A.ipynb): Genome-wide NIS-Seq analysis in IL-1b-stimulated HeLa-Cas9-p65-mNeonGreen cells, calculating fold-changes from Pearson correlation between mNeonGreen and nuclear staining, with statistical testing via Wilcoxon-Mann-Whitney and Benjamini-Hochberg correction.
- [**Figure 2D**](./figure2/NIS_Figure2D.ipynb): Genome-wide NIS-Seq analysis in  TNF-a-stimulated HeLa-Cas9-p65-mNeonGreen cells, calculating fold-changes from Pearson correlation between mNeonGreen and nuclear staining, with statistical testing via Wilcoxon-Mann-Whitney and Benjamini-Hochberg correction.
- [**Figure 3A**](./figure3/NIS_Figure3A.ipynb): Genome-wide NIS-Seq analysis in Nigericin-stimulated THP1-Cas9-ASC-GFP-CASP1/8DKO cells, focusing on fold-changes from the high-frequency filtered GFP signal per cell. Statistical testing was performed using the Wilcoxon-Mann-Whitney test with Benjamini-Hochberg correction. 
- [**Figure 3E**](./figure3/NIS_Figure3E.ipynb): Genome-wide NIS-Seq analysis in PrgI + PA-stimulated THP1-Cas9-ASC-GFP-CASP1/8DKO cells, focusing on fold-changes from the high-frequency filtered GFP signal per cell. Statistical testing was performed using the Wilcoxon-Mann-Whitney test with Benjamini-Hochberg correction.


## Getting started

1. Install [Miniconda](https://conda.io/miniconda.html)
2. Clone this repository:
    open a terminal and run:
    ```bash
    git clone https://github.com/schmid-burgk/NIS-Seq.git
    cd NIS-Seq
    ```
3. Open the notebooks:
    ```bash
    conda activate
    jupyter lab
    ```
