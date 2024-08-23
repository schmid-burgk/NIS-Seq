# NIS-Seq

## Analysis for figure 1E and 1F
The Jupyter Notebooks provided here were used to generate the figures presented in [*Cell Type-Agnostic Optical Perturbation Screening Using Nuclear In-Situ Sequencing (NIS-Seq)*](https://www.biorxiv.org/content/10.1101/2024.01.18.576210v1).

--- 
#### The figures include:
- [**Figure 1E**](./figure1/NIS_Figure1E.ipynb): Cell type panel with comparing mapped nuclei between *NIS-Seq* and [*Optical Pooled Screens*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6886477/) [*(Github)*](https://github.com/feldman4/OpticalPooledScreens_2019).
- [**Figure 1F**](./figure1/NIS_Figure1F.ipynb): Mixing experiment (GFP positive cells and NIS-barcode containing cells) to determine Specificity and Sensitivity of the method for PFA fixation and methanol fixation.

---
#### Getting started

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
