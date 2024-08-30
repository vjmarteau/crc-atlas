# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python [conda env:.conda-2024-scanpy]
#     language: python
#     name: conda-env-.conda-2024-scanpy-py
# ---

# %% [markdown]
# # Annotate metastasis seed

# %%
# ensure reproducibility -> set numba multithreaded mode
import os
from collections import OrderedDict
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sc_atlas_helpers as ah
import scanpy as sc
from nxfvars import nxfvars
from threadpoolctl import threadpool_limits

cpus = nxfvars.get("cpus", 2)
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

sc.settings.set_figure_params(
    dpi=200,
    facecolor="white",
    frameon=False,
)

# %% tags=["parameters"]
adata_path = nxfvars.get(
    "adata_path",
    "/data/projects/2022/CRCA/results/v1/artifacts/build_atlas/load_datasets/harmonize_datasets/artifacts/metastasis-adata.h5ad",
)
artifact_dir = nxfvars.get(
    "artifact_dir",
    "/data/scratch/marteau/downloads/seed/",
)
marker_genes_path = nxfvars.get(
    "marker_genes_path",
    "/data/projects/2022/CRCA/tables/cell_type_markers.csv",
)

# %%
adata = sc.read_h5ad(adata_path)
marker_genes = pd.read_csv(marker_genes_path)

# %% [markdown]
# ### Apply basic QC

# %%
# Basic QC (was set to "n_genes" < 100 for scAR denoising)
adata.obs["is_droplet"] = np.where(
    (adata.obs["n_genes"] < 200)
    | (adata.obs["total_counts"] < 400),
    "empty droplet",
    "cell"
)

# %%
# Plate-based protocols
adata_plate = adata[adata.obs["platform"].isin(['Microwell-seq', 'SMARTer (C1)', 'Smart-seq2', 'scTrio-seq2_Dong_protocol', 'scTrio-seq2_Tang_protocol'])].copy()
ah.pp.reprocess_adata_subset_scvi(adata_plate, leiden_res=2, n_neighbors=15, use_rep="X_scVI")
adata_plate.write(f"{artifact_dir}/plate-adata.h5ad", compression="lzf")

# %%
# Remove droplets
adata = adata[adata.obs["is_droplet"] == "cell"].copy()
ah.pp.reprocess_adata_subset_scvi(adata, leiden_res=2, n_neighbors=15, use_rep="X_scVI")

# %%
ah.pl.umap_qc_metrics(adata, vmax_total_counts=20000, vmax_n_genes=2000, save_plot=True, prefix="tumor", save_dir=artifact_dir)

# %%
ah.pl.umap_covariates(
    adata,
    covariates=[
        "sample_type",
        "sample_tissue",
        "tissue_cell_state",
        "treatment_status_before_resection",
        "sex",
        "enrichment_cell_types",
        "phase",
        "dataset",
    ],
    save_plot=True,
    prefix="tumor",
    save_dir=artifact_dir,
)

# %% [markdown]
# ## 2. Get coarse cell annotation

# %%
#marker_plot_dict = ah.pl.umap_marker_genes(
#    adata,
#    marker_genes,
#    col_gene_id="symbol",
#    save_plot=True,
#    save_dir=artifact_dir,
#    prefix="tumor_",
#    file_ext="png",
#)

# %%
cluster_annot = {
    "Epithelial cell": [0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 38, 47, 49, 50, 51], # 20 -> Hepatocyte
    "B cell": [27],
    "Plasma cell": [28, 42, 44],
    "T cell": [23, 24, 29, 30, 31, 32, 33, 34, 35, 36, 37],
    "Stromal cell": [21, 25, 26, 41, 46, 48],
    "Myeloid cell": [13, 14, 22, 39, 40, 43, 45],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}

adata.obs["cell_type_coarse"] = (
    adata.obs["leiden"].map(cluster_annot).fillna(adata.obs["leiden"])
)

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    fig = sc.pl.umap(
        adata,
        color="cell_type_coarse",
        legend_loc="on data",
        legend_fontoutline=1,
        legend_fontsize="xx-small",
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/metastasis-cell_type_coarse.png", bbox_inches="tight")

# %% [markdown]
# ## Reprocess major cell types

# %%
adata.obs["is_outlier"] = "cell"
adata.obs["cell_type_seed"] = "unknown"
adata.obs["cell_type_fine"] = "unknown"

# %% [markdown]
# ## Myeloid compartment

# %%
adata_myeloid = adata[adata.obs["cell_type_coarse"].isin(["Myeloid cell"])].copy()
ah.pp.reprocess_adata_subset_scvi(adata_myeloid, leiden_res=2, n_neighbors=15, use_rep="X_scVI")

# %% [markdown]
# ### Myeloid marker genes

# %%
myeloid_marker =     {
         "cDC1": {
        "positive": [
            ["CLEC9A", "XCR1", "CLNK", "CADM1", "WDFY4"],
        ],
        "negative": [["VCAN", "CD14"], ["EPCAM"]]
        },
        "cDC2": {
        "positive": [
            ["CD1C", "FCER1A"],
            ["CLEC10A"],
        ],
        "negative": [["VCAN", "CD14"], ["EPCAM"]]
        },
        "DC3": {
            "positive": [
                ["CD1C"],
                ["CD163", "CD14", "LYZ", "VCAN"],
            ],
            "negative": [["FCER1A"]]
        },
        "DC mature": {
            "positive": [
                ["CCL22"],
                ["CCR7"],
                ["LAMP3"],
            ],
            "negative": [[]]
        },
        "Macrophage": {
            "positive": [
                ["C1QA", "C1QB"],
            ],
            "negative": [["VCAN"], ["EPCAM"]],
        },
        "Monocyte": {
            "positive": [
                ["VCAN", "CD14"],
            ],
            "negative": [
                ["CD3E"],
                ["C1QA", "C1QB"],
                ["FCGR3B"],
                ["EPCAM"],
            ],
        },
    }

# %%
adata_myeloid.obs["cell_type_seed"] = ah.pp.score_seeds(
    adata_myeloid,
    myeloid_marker,
    pos_cutoff=0.5,
    neg_cutoff=0.5,
)

# %%
cluster_annot = {
    "Epithelial cell": [31],
    "T cell": [39],
    "Macrophage": [1, 4, 6, 7, 10, 11, 12, 13, 14, 15, 17, 20, 24, 26, 28, 40],
    "Monocyte": [3, 5, 8, 9, 16, 19, 21, 25, 27, 32],
    "Granulocyte progenitor": [],
    "Eosinophil": [],
    "Neutrophil": [34, 35, 36, 37, 38],
    "Mast cell": [29],
    "cDC progenitor": [18, 22, 23, 30],
    "cDC1": [33],
    "cDC2": [0, 2],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}

adata_myeloid.obs["cell_type_fine"] = (
    adata_myeloid.obs["leiden"].map(cluster_annot).fillna(adata_myeloid.obs["leiden"])
)

# %%
adata_myeloid.obs["cell_type_fine"] = adata_myeloid.obs["cell_type_fine"].astype(str)

adata_myeloid.obs.loc[
    (adata_myeloid.obs["cell_type_seed"] != "unknown")
  & ~(adata_myeloid.obs["leiden"].isin(["18", "22", "23", "29", "30", "31", "34", "35", "36", "37", "38"])),
    "cell_type_fine",
] = adata_myeloid.obs["cell_type_seed"]

adata_myeloid.obs["cell_type_seed"] = ah.pp.score_seeds(
    adata_myeloid,
    {
        "T cell": {
            "positive": [["CD3E", "TRAC"]],
            "negative": [["VCAN", "CD14"]],
        },
    },
    pos_cutoff=0.1,
    neg_cutoff=0,
)

adata_myeloid.obs.loc[
    (adata_myeloid.obs["cell_type_seed"] != "unknown")
  & (adata_myeloid.obs["leiden"].isin(["3"])),
    "cell_type_fine",
] = adata_myeloid.obs["cell_type_seed"]

# %%
# Seperate cycling cells
adata_myeloid.obs["cell_type_fine2"] = ah.pp.score_seeds(
    adata_myeloid,
    {
        "Cycling": {
            "positive": [["MKI67"]],
            "negative": [[]],
        },
    },
    pos_cutoff=0.1,
    neg_cutoff=0,
)

# %%
adata_myeloid.obs["cell_type_fine"] = adata_myeloid.obs["cell_type_fine"].astype(str)

adata_myeloid.obs.loc[
    (adata_myeloid.obs["cell_type_fine2"] == "Cycling")
  & (adata_myeloid.obs["cell_type_fine"].isin(["Monocyte", "Macrophage"])),
    "cell_type_fine",
] = "Macrophage cycling"

# %%
# Remove high mito + low counts/genes
adata_myeloid.obs["is_droplet"] = np.where(
      (adata_myeloid.obs["pct_counts_mito"] > 25)
    | (adata_myeloid.obs["n_genes"] < 200)
    | (adata_myeloid.obs["total_counts"] < 400),
    "empty droplet",
    "cell",
)

# %%
adata_myeloid.obs.loc[
    (adata_myeloid.obs["is_droplet"] == "empty droplet")
    & ~(adata_myeloid.obs["platform"].isin(["Microwell-seq", "SMARTer (C1)", "Smart-seq2", "scTrio-seq2_Dong_protocol", "scTrio-seq2_Tang_protocol", "TruDrop"])),
    "cell_type_fine",
] = "empty droplet"

# %%
# Transfer updated coarse annotation to cell_type_coarse column for correct subsetting
annot = {
    "T cell": "T cell",
    "Epithelial cell": "Epithelial cell",
}

adata_myeloid.obs["cell_type_coarse"] = (
    adata_myeloid.obs["cell_type_fine"].map(annot).fillna(adata_myeloid.obs["cell_type_coarse"])
)

# %%
ah.pp.integrate_back(adata, adata_myeloid, variable="cell_type_fine")
ah.pp.integrate_back(adata, adata_myeloid, variable="cell_type_coarse")

# %%
# Get clean myeloid adata
adata_myeloid = adata_myeloid[~adata_myeloid.obs["cell_type_fine"].isin(["T cell", "Epithelial cell", "empty droplet"])].copy()
ah.pp.reprocess_adata_subset_scvi(adata_myeloid, leiden_res=2, n_neighbors=15, use_rep="X_scVI")

# %%
adata_myeloid.write(f"{artifact_dir}/metastasis-myeloid-adata.h5ad", compression="lzf")

# %%
adata_myeloid.obs["cell_type_fine"] = pd.Categorical(
    adata_myeloid.obs["cell_type_fine"],
    categories=["Monocyte", "Macrophage", "Macrophage cycling", "cDC progenitor", "cDC1", "cDC2", "DC3", "DC mature", "Neutrophil", "Mast cell"],
    ordered=True,
)

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    fig = sc.pl.umap(
        adata_myeloid,
        color="cell_type_fine",
        #legend_loc="on data",
        legend_fontoutline=1,
        legend_fontsize="xx-small",
        groups=["Monocyte", "Macrophage", "Macrophage cycling", "cDC progenitor", "cDC1", "cDC2", "DC3", "DC mature", "Neutrophil", "Mast cell"],
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/metastasis-umap_myeloid_seeds.png", bbox_inches="tight")

# %% [markdown]
# ### Myeloid seeds dotplot

# %%
marker_genes = OrderedDict(
    [
        ("Monocyte", ["VCAN", "CD14", "CD163", "FCGR3A"]),
        ("Macrophage", ["C1QA", "C1QB"]),
        ("Pan-DC", ["FLT3"]),
        ("cDC progenitor", ["AXL", "MRC1"]),
        ("cDC1", ["CLEC9A", "XCR1", "CLNK", "CADM1", "WDFY4"]),
        ("cDC2", ["CD1C", "CD1E", "CLEC10A", "FCER1A"]),
        ("pDC", ["IL3RA", "CLEC4C", "CCDC50", "ITM2C", "GZMB"]),
        ("DC mature", ["CCL22", "CCR7", "LAMP3"]),
        ("Granulocyte progenitor", ["CD24", "CEACAM8", "ITGAM", "ARG1"]),
        ("Eosinophil", ["CLC", "CCR3", "ADGRE1", "ALOX15"]),
        ("Neutrophil", ["CXCR2", "FCGR3B", "HCAR3", "FFAR2", "PROK2"]),
        ("Mast cell", ["TPSB2", "CPA3", "TPSAB1", "GATA2"]),
        ("Pan-Cycling", ["MKI67", "CDK1"]),
    ]
)

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    dp = sc.pl.dotplot(
        adata_myeloid,
        groupby="cell_type_fine",
        var_names=marker_genes,
        layer="log1p_norm",
        standard_scale="var",  # standard scale: normalize each gene to range from 0 to 1
        return_fig=True,
    )
    axs = dp.add_totals(color="lightgrey").style(cmap="Reds").show(return_axes=True)
    ax = axs["mainplot_ax"]
    # Loop through ticklabels and make them italic
    for l in ax.get_xticklabels():
        l.set_style("italic")
        g = l.get_text()
        # Change settings (e.g. color) of certain ticklabels based on their text (here gene name)
        if g == "HLA-DRA":
            l.set_color("#d7191c")
            
    plt.savefig(f"{artifact_dir}/metastasis-dotplot_myeloid_seeds.pdf", bbox_inches="tight")

# %%
adata_myeloid.obs["cell_type_fine"].value_counts()

# %% [markdown]
# ## T cell compartment

# %%
adata_t = adata[adata.obs["cell_type_coarse"].isin(["T cell"])].copy()
ah.pp.reprocess_adata_subset_scvi(adata_t, leiden_res=3, n_neighbors=15, use_rep="X_scVI")

# %% [markdown]
# ### T cell marker genes

# %%
t_cell_marker = {
    "CD8": {
        "positive": [
            ["CD8A", "CD8B"],
            ["CD3E", "TRAC", "TRBC1", "TRBC2"],
        ],
        "negative": [
            ["CD4"],
            ["FCGR3A", "TYROBP"],  # NK marker
            ["TRDC", "TRGC2"], # gamma-delta
            ["VCAN", "CD14"], # Monocyte marker
            ["MKI67"],  # cycling -> for seed annot seperately
        ],
    },
    "CD8 cycling": {
        "positive": [
            ["CD8A", "CD8B"],
            ["CD3E", "TRAC", "TRBC1", "TRBC2"],
            ["MKI67"]
        ],
        "negative": [
            ["CD4"],
            ["FCGR3A", "TYROBP"],
            ["TRDC", "TRGC2"],
            ["VCAN", "CD14"],
        ],
    },
    "CD4": {
        "positive": [
            ["CD4"],
            ["CD3E", "TRAC", "TRBC1", "TRBC2"],
        ],
        "negative": [
            ["CD8A", "CD8B"],
            ["FOXP3"],  # Score Tregs seperately
            ["FCGR3A", "TYROBP"],  # NK markers
            ["VCAN", "CD14"],
            ["MKI67"],  # cycling -> for seed annot seperately
        ],
    },
    "CD4 cycling": {
        "positive": [
            ["CD4"],
            ["CD3E", "TRAC", "TRBC1", "TRBC2"],
            ["MKI67"]
        ],
        "negative": [
            ["CD8A", "CD8B"],
            ["VCAN", "CD14"],
            ["FCGR3A", "TYROBP"],
        ],
    },
    "Treg": {
        "positive": [
            ["FOXP3"], ["IL2RA"],
        ],
        "negative": [
            ["CD8A", "CD8B"],
            ["FCGR3A", "TYROBP"],
            ["VCAN", "CD14"],
            ["MKI67"],
        ],
    },
    "NK": {
        "positive": [
            ["NKG7", "FCGR3A", "TYROBP", "S1PR5"],
        ],
        "negative": [
            ["CD8A", "CD8B"],
            ["CD3E", "TRAC", "TRBC1", "TRBC2"],
            ["TRDC", "TRDV1", "TRGC1", "TRGC2"],
            ["IL4I1", "KIT", "LST1", "RORC"],
            ["VCAN", "CD14"],
            ["MKI67"]
        ],
    },
    "NK cycling": {
        "positive": [
            ["NKG7", "FCGR3A", "TYROBP", "S1PR5"],
            ["MKI67"]
        ],
        "negative": [
            ["CD8A", "CD8B"],
            ["TRDC", "TRDV1", "TRGC1", "TRGC2"],
            ["IL4I1", "KIT", "LST1", "RORC"],
            ["CD3E", "TRAC", "TRBC1", "TRBC2"],
            ["VCAN", "CD14"],
        ],
    },
    "gamma-delta": {
        "positive": [
            ["TRDC", "TRDV1", "TRGC1", "TRGC2"],
            ["CD3E", "TRAC"],
        ],
        "negative": [
            ["CD8A", "CD8B"],
            ["FOXP3", "CD4"],
            ["TRBC1", "TRBC2"],
            ["FCGR3A", "TYROBP"],
            ["VCAN", "CD14"],
            ["MKI67"],
        ],
    },
    "ILC": {
        "positive": [
            ["IL4I1", "KIT", "LST1", "RORC", "IL22", "IL23R"],
        ],
        "negative": [
            ["CD8A", "CD8B"],
            ["FOXP3", "CD4"],
            ["FCGR3A", "TYROBP"],
            ["TRDC", "TRDV1", "TRGC1", "TRGC2"],
            ["CD3E", "TRAC", "TRBC1", "TRBC2"],
            ["VCAN", "CD14"],
            ["MKI67"],
        ],
    },
    # NKT cells are tricky as they express a combination of markers
    "NKT": {
        "positive": [
            ["SLAMF1"],
            ["CD4"],
            ["CD28"],
            ["NKG7"],
            ["GZMA"],
            ["CD3E", "TRAC", "TRBC1", "TRBC2"],
        ],
        "negative": [
            ["FOXP3"],
            ["TRDC", "TRDV1", "TRGC1", "TRGC2"],
            ["VCAN", "CD14"],
            ["MKI67"]
        ],
    },
    "NKT": {
        "positive": [
            ["SLAMF6"],
            ["CD4"],
            ["CD28"],
            ["NKG7"],
            ["GZMA"],
            ["CD3E", "TRAC", "TRBC1", "TRBC2"],
        ],
        "negative": [
            ["FOXP3"],
            ["TRDC", "TRDV1", "TRGC1", "TRGC2"],
            ["VCAN", "CD14"],
            ["MKI67"]
        ],
    },
    "NKT": {
        "positive": [
            ["SLAMF1"],
            ["CD8A"],
            ["CD28"],
            ["NKG7"],
            ["GZMA"],
            ["CD3E", "TRAC", "TRBC1", "TRBC2"],
        ],
        "negative": [
            ["FOXP3"],
            ["TRDC", "TRDV1", "TRGC1", "TRGC2"],
            ["VCAN", "CD14"],
            ["MKI67"]
        ],
    },
    "NKT": {
        "positive": [
            ["SLAMF6"],
            ["CD8A"],
            ["CD28"],
            ["NKG7"],
            ["GZMA"],
            ["CD3E", "TRAC", "TRBC1", "TRBC2"],
        ],
        "negative": [
            ["FOXP3"],
            ["TRDC", "TRDV1", "TRGC1", "TRGC2"],
            ["VCAN", "CD14"],
            ["MKI67"]
        ],
    },
}

# %%
adata_t.obs["cell_type_seed"] = ah.pp.score_seeds(
    adata_t,
    t_cell_marker,
    pos_cutoff=0,
    neg_cutoff=0.1,
)

# %%
cluster_annot = {
    "CD8": [3, 5, 7, 8, 9, 10, 16, 19, 22, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 38, 42, 47, 51, 52, 53, 55, 56],
    "CD4": [4, 6, 11, 12, 13, 14, 15, 17, 18, 32, 33, 39, 43, 45, 46, 48, 50, 54],
    "Treg": [0, 1, 2, 40, 41, 44],
    "NK": [20, 21, 23],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}

adata_t.obs["cell_type_fine"] = (
    adata_t.obs["leiden"].map(cluster_annot).fillna(adata_t.obs["leiden"])
)

# %%
adata_sub = adata_t[adata_t.obs["leiden"] == "49"].copy()
sc.tl.leiden(adata_sub, 0.5, flavor="igraph")

adata_sub.obs["cell_type_fine"] = adata_sub.obs["cell_type_fine"].astype(str)
cluster_annot = {
    "Treg": [0],
    "CD8": [1, 4, 5],
    "CD4": [2, 3, 6],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}
adata_sub.obs["cell_type_fine"] = adata_sub.obs["leiden"].map(cluster_annot).fillna(adata_sub.obs["leiden"])

ah.pp.integrate_back(adata_t, adata_sub, variable="cell_type_fine")

# %%
adata_t.obs["cell_type_fine"] = adata_t.obs["cell_type_fine"].astype(str)

adata_t.obs.loc[
    (adata_t.obs["cell_type_seed"] != "unknown"),
    "cell_type_fine",
] = adata_t.obs["cell_type_seed"]

# %%
# Remove high mito + low counts/genes
adata_t.obs["is_droplet"] = np.where(
    (adata_t.obs["pct_counts_mito"] > 25)
    & (adata_t.obs["n_genes"] < 500)
    & (adata_t.obs["total_counts"] < 1000),
    "empty droplet",
    "cell",
)

# %%
adata_t.obs.loc[
    (adata_t.obs["is_droplet"] == "empty droplet")
    & ~(adata_t.obs["platform"].isin(["Microwell-seq", "SMARTer (C1)", "Smart-seq2", "scTrio-seq2_Dong_protocol", "scTrio-seq2_Tang_protocol", "TruDrop"])),
    "cell_type_fine",
] = "empty droplet"

# %%
ah.pp.integrate_back(adata, adata_t, variable="cell_type_fine")

# %%
adata_t.write(f"{artifact_dir}/metastasis-t_cell-adata.h5ad", compression="lzf")

# %%
adata_t = adata_t[~adata_t.obs["cell_type_fine"].isin(["empty droplet"])].copy()

adata_t.obs["cell_type_fine"] = pd.Categorical(
    adata_t.obs["cell_type_fine"],
    categories=["CD4", "Treg", "CD8", "NK", "ILC", "gamma-delta", "NKT", "CD4 cycling", "CD8 cycling", "NK cycling"],
    ordered=True,
)

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    fig = sc.pl.umap(
        adata_t,
        color="cell_type_fine",
        #legend_loc="on data",
        legend_fontoutline=1,
        legend_fontsize="xx-small",
        groups=["CD4", "Treg", "CD8", "NK", "ILC", "gamma-delta", "NKT", "CD4 cycling", "CD8 cycling", "NK cycling"],
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/metastasis-umap_t_cell_seeds.png", bbox_inches="tight")

# %% [markdown]
# ### T cell seeds dotplot

# %%
marker_genes = OrderedDict(
    [
        ("Pan-T cell", ["CD3E", "TRAC", "TRBC1", "TRBC2"]),
        ("CD4", ["CD4"]),
        ("Treg", ["FOXP3", "IL2RA"]),
        ("CD8", ["CD8A", "CD8B"]),
        ("NK", ["NKG7", "FCGR3A", "TYROBP", "S1PR5"]),
        ("gamma-delta", ["TRDC", "TRDV1", "TRGC1", "TRGC2"]),
        ("ILC", ["ICOS", "IL4I1", "KIT", "LST1", "RORC", "IL22", "IL1R1", "IL23R", "LMNA"]),
        ("NKT", ["SLAMF1", "SLAMF6", "CD28", "GZMA"]),
        ("Pan-Cycling", ["MKI67", "CDK1"]),
    ]
)

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    dp = sc.pl.dotplot(
        adata_t,
        groupby="cell_type_fine",
        var_names=marker_genes,
        layer="log1p_norm",
        standard_scale="var",  # standard scale: normalize each gene to range from 0 to 1
        return_fig=True,
    )
    axs = dp.add_totals(color="lightgrey").style(cmap="Reds").show(return_axes=True)
    ax = axs["mainplot_ax"]
    # Loop through ticklabels and make them italic
    for l in ax.get_xticklabels():
        l.set_style("italic")
        g = l.get_text()
        # Change settings (e.g. color) of certain ticklabels based on their text (here gene name)
        if g == "HLA-DRA":
            l.set_color("#d7191c")
            
    plt.savefig(f"{artifact_dir}/metastasis-dotplot_t_cell_seeds.pdf", bbox_inches="tight")

# %%
adata_t.obs["cell_type_fine"].value_counts()

# %% [markdown]
# ## Plasma/B cell compartment

# %%
adata_plasma = adata[adata.obs["cell_type_coarse"].isin(["B cell", "Plasma cell"])].copy()
ah.pp.reprocess_adata_subset_scvi(adata_plasma, leiden_res=2, n_neighbors=15, use_rep="X_scVI")

# %%
cluster_annot = {
    "Plasma IgA": [10],
    "Plasma IgG": [9, 11, 13, 14, 16, 19, 20, 22, 24, 28],
    "Plasmablast": [3, 21],
    "B cell naive": [7, 12, 15, 25, 26],
    "B cell memory": [1, 2],
    "B cell activated": [0, 4, 5, 6, 8, 17, 18, 23, 27],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}

adata_plasma.obs["cell_type_fine"] = (
    adata_plasma.obs["leiden"].map(cluster_annot).fillna(adata_plasma.obs["leiden"])
)

# %%
# Score IgG vs IgA with higher "pos_cutoff"
adata_plasma.obs["cell_type_seed"] = ah.pp.score_seeds(
    adata_plasma,
    { "Plasma IgG": {
        "positive": [["IGHG1", "IGHG2", "IGHG3", "IGHG4"]],
        "negative": [["IGHM"]]
    },
     "Plasma IgA": {
        "positive": [["IGHA1", "IGHA2"]],
        "negative": [["IGHM"]]
    },
     "Plasma IgM": {
        "positive": [["IGHM"]],
        "negative": [[]]
    },
    },
    pos_cutoff=1,
    neg_cutoff=0,
)

adata_plasma.obs.loc[
    (adata_plasma.obs["cell_type_seed"] != "unknown")
    & (adata_plasma.obs["leiden"].isin(["3", "9", "10", "11", "13", "14", "16", "19", "20", "22", "24", "28"])),
    "cell_type_fine",
] = adata_plasma.obs["cell_type_seed"]

# %%
# Score B cell subtypes
adata_plasma.obs["cell_type_seed2"] = ah.pp.score_seeds(
    adata_plasma,
    {
    "B cell memory": {
        "positive": [["CD27"], ["CD44"]],
        "negative": [["IGHM", "IGHD"]],
    },
    "B cell naive": {
        "positive": [["IGHD"], ["TCL1A"], ["FCER2"]],
        "negative": [],
    },
    "B cell activated": {
        "positive": [["CD69", "CD83"]],
        "negative": [["IGHD", "TCL1A", "FCER2"]],
    },
    },
    pos_cutoff=0.3,
    neg_cutoff=0.3,
)

# %%
adata_plasma.obs.loc[
    (adata_plasma.obs["cell_type_seed2"] != "unknown")
    & ~(adata_plasma.obs["leiden"].isin(["3", "9", "10", "11", "13", "14", "16", "19", "20", "21", "22", "24", "28"])),
    "cell_type_fine",
] = adata_plasma.obs["cell_type_seed2"]

# %%
# Remove high mito + low counts/genes
adata_plasma.obs["is_droplet"] = np.where(
      (adata_plasma.obs["pct_counts_mito"] > 25)
    | (adata_plasma.obs["n_genes"] < 200)
    | (adata_plasma.obs["total_counts"] < 400),
    "empty droplet",
    "cell",
)

# %%
adata_plasma.obs.loc[
    (adata_plasma.obs["is_droplet"] == "empty droplet")
    & ~(adata_plasma.obs["platform"].isin(["Microwell-seq", "SMARTer (C1)", "Smart-seq2", "scTrio-seq2_Dong_protocol", "scTrio-seq2_Tang_protocol", "TruDrop"])),
    "cell_type_fine",
] = "empty droplet"

# %%
adata_plasma.write(f"{artifact_dir}/metastasis-plasma_cell-adata.h5ad", compression="lzf")

# %%
adata_plasma = adata_plasma[~adata_plasma.obs["cell_type_fine"].isin(["empty droplet"])].copy()

adata_plasma.obs["cell_type_fine"] = pd.Categorical(
    adata_plasma.obs["cell_type_fine"],
    categories=["B cell naive", "B cell activated", "B cell memory", "Plasma IgA", "Plasma IgG", "Plasma IgM", "Plasmablast"],
    ordered=True,
)

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    fig = sc.pl.umap(
        adata_plasma,
        color="cell_type_fine",
        #legend_loc="on data",
        legend_fontoutline=1,
        legend_fontsize="xx-small",
        groups=["B cell naive", "B cell activated", "B cell memory", "Plasma IgA", "Plasma IgG", "Plasma IgM", "Plasmablast"],
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/metastasis-umap_plasma_seeds.png", bbox_inches="tight")

# %% [markdown]
# ### Plasma cell seeds dotplot

# %%
marker_genes = OrderedDict(
    [
        ("Pan-B cell", ["CD79B", "CD19", "MS4A1"]),
        ("GC B cell", ["MARCKSL1", "GMDS", "S100A10"]),
        ("B cell naive", ["IGHD", "TCL1A", "FCER2"]),
        ("B cell activated", ["CD69", "CD83"]),
        ("B cell memory", ["CD27", "CD44"]),
        ("Plasmablast", ["ENO1", "NME1", "HMGB2", "MYDGF"]),
        ("Pan-Plasma cell", ["JCHAIN"]),
        ("Plasma IgA", ["IGHA1", "IGHA2"]),
        ("Plasma IgG", ["IGHG1", "IGHG2", "IGHG3", "IGHG4"]),
        ("Plasma IgM", ["IGHM"]),
        ("Pan-Cycling", ["MKI67", "CDK1"]),
    ]
)

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    dp = sc.pl.dotplot(
        adata_plasma,
        groupby="cell_type_fine",
        var_names=marker_genes,
        layer="log1p_norm",
        standard_scale="var",  # standard scale: normalize each gene to range from 0 to 1
        return_fig=True,
    )
    axs = dp.add_totals(color="lightgrey").style(cmap="Reds").show(return_axes=True)
    ax = axs["mainplot_ax"]
    # Loop through ticklabels and make them italic
    for l in ax.get_xticklabels():
        l.set_style("italic")
        g = l.get_text()
        # Change settings (e.g. color) of certain ticklabels based on their text (here gene name)
        if g == "HLA-DRA":
            l.set_color("#d7191c")
            
    plt.savefig(f"{artifact_dir}/metastasis-dotplot_plasma_cell_seeds.pdf", bbox_inches="tight")

# %%
adata_plasma.obs["cell_type_fine"].value_counts()

# %% [markdown]
 ## Epithelial cell compartment

# %%
adata_epi = adata[adata.obs["cell_type_coarse"].isin(["Epithelial cell"])].copy()
ah.pp.reprocess_adata_subset_scvi(adata_epi, leiden_res=2, n_neighbors=15, use_rep="X_scVI")

# %% [markdown]
# ### Epithelial cell marker genes

# %%
epi_cell_marker = {
    "Tumor Goblet-like": {
        "positive": [["MUC2", "FCGBP", "ZG16", "ATOH1"]],
        "negative": [["FABP1", "CEACAM7", "ANPEP", "SI"]],
    },
    "Tumor Colonocyte-like": {
        "positive": [["FABP1", "CEACAM7", "ANPEP", "SI"]],
        "negative": [
            ["BEST4", "OTOP2", "SPIB"],
            ["TOP2A", "UBE2C", "PCLAF", "HELLS", "TK1"],
            ["LGR5", "SMOC2"],
        ],
    },
    "Tumor Crypt-like": {  # Includes cycling MKI67 pos cells
        "positive": [["LGR5", "SMOC2"]],
        "negative": [
            ["MUC2", "FCGBP", "ZG16", "ATOH1"],
            ["FABP1", "CEACAM7", "ANPEP", "SI"],
        ],
    },
    "Tumor TA-like": {  # TA: Transit-amplifying; Includes cycling MKI67 pos cells
        "positive": [["TOP2A", "UBE2C", "PCLAF", "HELLS", "TK1"]],
        "negative": [
            ["MUC2", "FCGBP", "ZG16", "ATOH1"],
            ["FABP1", "CEACAM7", "ANPEP", "SI"],
            ["LGR5", "SMOC2"],
        ],
    },
}

# %%
adata_epi.obs["cell_type_seed"] = ah.pp.score_seeds(
    adata_epi,
    epi_cell_marker,
    pos_cutoff=0.1,
    neg_cutoff=0,
)

# %%
cluster_annot = {
    "Tumor Colonocyte-like": [0, 5, 7, 8, 9, 10, 11, 14, 15, 16, 18, 22, 23, 24, 33, 34, 36, 41, 42, 46],
    "Tumor Goblet-like": [3, 25, 39, 45],
    "Tumor Crypt-like": [6, 13, 17, 26, 30, 37, 38, 40, 44],
    "Tumor TA-like": [1, 2, 4, 12, 20, 21, 27, 28, 29, 31, 32],
    "Hepatocyte": [43],
    "Tuft": [35], # 19 -> Not sure how Tuft cells get to the liver (metastasis? contamination?); no other marker genes match ...
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}

adata_epi.obs["cell_type_fine"] = (
    adata_epi.obs["leiden"].map(cluster_annot).fillna(adata_epi.obs["leiden"])
)

# %%
adata_sub = adata_epi[adata_epi.obs["leiden"].isin(["19"])].copy()
sc.tl.leiden(adata_sub, 0.3, flavor="igraph")

adata_sub.obs["cell_type_fine"] = adata_sub.obs["cell_type_fine"].astype(str)
cluster_annot = {
    "Tuft": [0],
    "Tumor Colonocyte-like": [1, 2, 3, 4],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}
adata_sub.obs["cell_type_fine"] = adata_sub.obs["leiden"].map(cluster_annot).fillna(adata_sub.obs["leiden"])

ah.pp.integrate_back(adata_epi, adata_sub, variable="cell_type_fine")

# %%
adata_epi.obs["cell_type_fine"] = adata_epi.obs["cell_type_fine"].astype(str)

adata_epi.obs.loc[
    (adata_epi.obs["cell_type_seed"] != "unknown"),
    "cell_type_fine",
] = adata_epi.obs["cell_type_seed"]

# %%
# Remove high mito + low counts/genes
adata_epi.obs["is_droplet"] = np.where(
    (adata_epi.obs["pct_counts_mito"] > 25)
    & (adata_epi.obs["n_genes"] < 1000)
    & (adata_epi.obs["total_counts"] < 1000),
    "empty droplet",
    "cell",
)

# %%
adata_epi.obs.loc[
    (adata_epi.obs["is_droplet"] == "empty droplet")
    & ~(adata_epi.obs["platform"].isin(["Microwell-seq", "SMARTer (C1)", "Smart-seq2", "scTrio-seq2_Dong_protocol", "scTrio-seq2_Tang_protocol", "TruDrop"])),
    "cell_type_fine",
] = "empty droplet"

# %%
ah.pp.integrate_back(adata, adata_epi, variable="cell_type_fine")

# %%
adata_epi.write(f"{artifact_dir}/metastasis-epi-adata.h5ad", compression="lzf")

# %%
adata_epi = adata_epi[~adata_epi.obs["cell_type_fine"].isin(["empty droplet"])].copy()

adata_epi.obs["cell_type_fine"] = pd.Categorical(
    adata_epi.obs["cell_type_fine"],
    categories=["Tumor Colonocyte-like", "Tumor Goblet-like", "Tumor Crypt-like", "Tumor TA-like", "Tuft", "Hepatocyte"],
    ordered=True,
)

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    fig = sc.pl.umap(
        adata_epi,
        color="cell_type_fine",
        #legend_loc="on data",
        legend_fontoutline=1,
        legend_fontsize="xx-small",
        groups=["Tumor Colonocyte-like", "Tumor Goblet-like", "Tumor Crypt-like", "Tumor TA-like", "Tuft", "Hepatocyte"],
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/metastasis-umap_epi_seeds.png", bbox_inches="tight")

# %% [markdown]
# ### Epithelial (Cancer cell) seeds dotplot

# %%
marker_genes = OrderedDict(
    [
        ("Pan-Cancer cell", ["EPCAM", "ELF3", "KRT8", "KRT18", "KRT19"]),
        ("Tumor Colonocyte-like", ["FABP1", "CEACAM7", "ANPEP", "SI"]),
        ("Tumor BEST4", ["BEST4", "OTOP2", "SPIB"]),
        ("Tumor Goblet-like", ["MUC2", "FCGBP", "ZG16", "ATOH1"]),
        ("Tumor Crypt-like", ["LGR5", "SMOC2"]),
        ("Tumor TA-like", ["TOP2A", "UBE2C", "PCLAF", "HELLS", "TK1"]),
        ("Hepatocyte", ["CYP3A4", "CYP2E1", "ALB",]),
        ("Tuft", ["IRAG2", "SH2D6", "ALOX5AP", "AVIL"]),
        ("Pan-Cycling", ["MKI67", "CDK1"]),
    ]
)

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    dp = sc.pl.dotplot(
        adata_epi,
        groupby="cell_type_fine",
        var_names=marker_genes,
        layer="log1p_norm",
        standard_scale="var",  # standard scale: normalize each gene to range from 0 to 1
        return_fig=True,
    )
    axs = dp.add_totals(color="lightgrey").style(cmap="Reds").show(return_axes=True)
    ax = axs["mainplot_ax"]
    # Loop through ticklabels and make them italic
    for l in ax.get_xticklabels():
        l.set_style("italic")
        g = l.get_text()
        # Change settings (e.g. color) of certain ticklabels based on their text (here gene name)
        if g == "HLA-DRA":
            l.set_color("#d7191c")
            
    plt.savefig(f"{artifact_dir}/metastasis-dotplot_epi_seeds.pdf", bbox_inches="tight")

# %%
adata_epi.obs["cell_type_fine"].value_counts()

# %% [markdown]
# ## Stromal cell compartment

# %%
adata_stromal = adata[adata.obs["cell_type_coarse"].isin(["Stromal cell"])].copy()
ah.pp.reprocess_adata_subset_scvi(adata_stromal, leiden_res=2, n_neighbors=15, use_rep="X_scVI")

# %% [markdown]
# ### Stromal marker genes

# %%
stromal_cell_marker = {
    "Endothelial lymphatic": {
        "positive": [["PECAM1", "CDH5", "VWF", "KDR", "FLT1"], ["PROX1", "TFF3"]],
        "negative": [
            ["HEY1", "GJA4", "GJA5", "SEMA3G"], # Endothelial arterial
            ["ACKR1", "MADCAM1", "SELP"], # Endothelial venous
            ["MKI67"],
        ],
    },
    "Endothelial venous": {
        "positive": [["PECAM1", "CDH5", "VWF", "KDR", "FLT1"], ["ACKR1", "MADCAM1", "SELP"]],
        "negative": [
            ["PROX1", "TFF3"], # Endothelial lymphatic
            ["HEY1", "GJA4", "GJA5", "SEMA3G"], # Endothelial arterial
            ["MKI67"],
        ],
    },
    "Endothelial arterial": {
        "positive": [["PECAM1", "CDH5", "VWF", "KDR", "FLT1"], ["HEY1", "GJA4", "GJA5", "SEMA3G"]],
        "negative": [
            ["PROX1", "TFF3"], # Endothelial lymphatic
            ["ACKR1", "MADCAM1", "SELP"], # Endothelial venous
            ["MKI67"],
        ],
    },
    "Pericyte": {
        "positive": [["NDUFA4L2", "RGS5"]],
        "negative": [["PECAM1","CDH5", "VWF", "KDR", "FLT1"], ["F3", "GREM1"], ["MKI67"]],
    },
    "Fibroblast S2": {
        "positive": [["F3", "AGT", "NRG1", "SOX6"]],
        "negative": [
            ["ADAMDEC1", "FABP5", "APOE"], # Fibroblast S1
            ["C7", "GREM1"], # Fibroblast S3
            ["S100B", "PLP1"], # Schwann cell
            ["NDUFA4L2", "RGS5"], # Pericyte
            ["PECAM1", "CDH5", "VWF", "KDR", "FLT1", "PROX1", "TFF3"], # Endothelial
            ["EPCAM"],
            ["MKI67"],
        ],
    },
    "Fibroblast S3": {
        "positive": [["C7", "ASPN", "CCDC80", "GREM1"]],
        "negative": [
            ["F3", "AGT", "NRG1", "SOX6"], # Fibroblast S2
            ["S100B", "PLP1"], # Schwann cell
            ["NDUFA4L2", "RGS5"], # Pericyte
            ["PECAM1","CDH5", "VWF", "KDR", "FLT1", "PROX1", "TFF3"], # Endothelial
            ["EPCAM"],
            ["MKI67"],
        ],
    },
}

# %%
adata_stromal.obs["cell_type_seed"] = ah.pp.score_seeds(
    adata_stromal,
    stromal_cell_marker,
    pos_cutoff=0.1,
    neg_cutoff=0,
)

# %%
cluster_annot = {
    "Epithelial cell": [24],
    "Pericyte": [1, 4, 6, 7, 11, 12, 15, 16, 25],
    "Endothelial lymphatic": [22],
    "Fibroblast S2": [34],
    "Fibroblast S3": [0, 2, 3, 8, 9, 10, 13, 14, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31, 32, 33, 35],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}

adata_stromal.obs["cell_type_fine"] = (
    adata_stromal.obs["leiden"].map(cluster_annot).fillna(adata_stromal.obs["leiden"])
)

# %%
adata_sub = adata_stromal[adata_stromal.obs["leiden"].isin(["5", "17", "18"])].copy()
sc.tl.leiden(adata_sub, 1, flavor="igraph")

adata_sub.obs["cell_type_fine"] = adata_sub.obs["cell_type_fine"].astype(str)
cluster_annot = {
    "Endothelial venous": [1, 3, 4, 7, 8, 9, 10, 11],
    "Endothelial arterial": [0, 2, 5, 6, 12],
    "Endothelial lymphatic": [13],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}
adata_sub.obs["cell_type_fine"] = adata_sub.obs["leiden"].map(cluster_annot).fillna(adata_sub.obs["leiden"])

ah.pp.integrate_back(adata_stromal, adata_sub, variable="cell_type_fine")

# %%
adata_stromal.obs["cell_type_fine"] = adata_stromal.obs["cell_type_fine"].astype(str)

adata_stromal.obs.loc[
    (adata_stromal.obs["cell_type_seed"] != "unknown"),
    "cell_type_fine",
] = adata_stromal.obs["cell_type_seed"]

# %%
# Remove high mito + low counts/genes
adata_stromal.obs["is_droplet"] = np.where(
    (adata_stromal.obs["pct_counts_mito"] > 25)
    & (adata_stromal.obs["n_genes"] < 1000)
    & (adata_stromal.obs["total_counts"] < 1000),
    "empty droplet",
    "cell",
)

# %%
adata_stromal.obs.loc[
    (adata_stromal.obs["is_droplet"] == "empty droplet")
    & ~(adata_stromal.obs["platform"].isin(["Microwell-seq", "SMARTer (C1)", "Smart-seq2", "scTrio-seq2_Dong_protocol", "scTrio-seq2_Tang_protocol", "TruDrop"])),
    "cell_type_fine",
] = "empty droplet"

# %%
# Transfer updated coarse annotation to cell_type_coarse column for correct subsetting
adata_stromal.obs["cell_type_coarse"] = (
    adata_stromal.obs["cell_type_fine"].map({"Epithelial cell": "Epithelial cell"}).fillna(adata_stromal.obs["cell_type_coarse"])
)

# %%
ah.pp.integrate_back(adata, adata_stromal, variable="cell_type_fine")
ah.pp.integrate_back(adata, adata_stromal, variable="cell_type_coarse")

# %%
adata_stromal.write(f"{artifact_dir}/metastasis-stromal-adata.h5ad", compression="lzf")

# %%
adata_stromal = adata_stromal[~adata_stromal.obs["cell_type_fine"].isin(["Epithelial cell", "empty droplet"])].copy()

adata_stromal.obs["cell_type_fine"] = pd.Categorical(
    adata_stromal.obs["cell_type_fine"],
    categories=["Endothelial venous", "Endothelial arterial", "Endothelial lymphatic", "Fibroblast S2", "Fibroblast S3", "Pericyte"],
    ordered=True,
)

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    fig = sc.pl.umap(
        adata_stromal,
        color="cell_type_fine",
        #legend_loc="on data",
        legend_fontoutline=1,
        legend_fontsize="xx-small",
        groups=["Fibroblast S2", "Fibroblast S3", "Endothelial lymphatic", "Endothelial arterial", "Endothelial venous", "Pericyte"],
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/metastasis-umap_stromal_seeds.png", bbox_inches="tight")

# %% [markdown]
# ### Stromal seeds dotplot

# %%
marker_genes = OrderedDict(
    [
        ("Pan-Endothelial", ["PECAM1", "CDH5", "VWF", "FLT1"]),
        ("Endothelial venous", ["ACKR1", "MADCAM1", "SELP"]), # "APLNR" ?
        ("Endothelial arterial", ["HEY1", "GJA4", "GJA5", "SEMA3G"]),
        ("Endothelial lymphatic", ["PROX1", "TFF3"]),
        ("Pan-Fibroblast", ["COL1A1", "COL3A1", "PDGFRA"]),
        ("Fibroblast S1", ["ADAMDEC1", "APOE"]),
        ("Fibroblast S2", ["F3", "AGT", "SOX6"]),
        ("Fibroblast S3", ["C7", "ASPN", "CCDC80", "GREM1"]),
        ("Pericyte", ["NDUFA4L2", "RGS5"]),
        ("Schwann cell", ["S100B", "PLP1", "NCAM1"]),
        ("Pan-Cycling", ["MKI67", "CDK1"]),
    ]
)

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    dp = sc.pl.dotplot(
        adata_stromal,
        groupby="cell_type_fine",
        var_names=marker_genes,
        layer="log1p_norm",
        standard_scale="var",  # standard scale: normalize each gene to range from 0 to 1
        return_fig=True,
    )
    axs = dp.add_totals(color="lightgrey").style(cmap="Reds").show(return_axes=True)
    ax = axs["mainplot_ax"]
    # Loop through ticklabels and make them italic
    for l in ax.get_xticklabels():
        l.set_style("italic")
        g = l.get_text()
        # Change settings (e.g. color) of certain ticklabels based on their text (here gene name)
        if g == "HLA-DRA":
            l.set_color("#d7191c")
            
    plt.savefig(f"{artifact_dir}/metastasis-dotplot_stromal_seeds.pdf", bbox_inches="tight")

# %%
adata_stromal.obs["cell_type_fine"].value_counts()

# %% [markdown]
# ## 2. Export metastasis seed cell types

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    fig = sc.pl.umap(
        adata,
        color="cell_type_fine",
        legend_fontoutline=1,
        legend_fontsize="xx-small",
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/metastasis-umap_cell_type_fine.png", bbox_inches="tight")

# %%
adata[adata.obs["cell_type_fine"] != "empty droplet"].shape

# %%
adata.obs["cell_type_fine"].value_counts(dropna=False)

# %%
# Save metastasis seed annotation
pd.DataFrame(adata.obs[["cell_type_fine"]]).reset_index(names=["obs_names"]).to_csv(
    f"{artifact_dir}/metastasis-cell_type_fine.csv", index=False
)
