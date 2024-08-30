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
# # Annotate blood seed

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
    "../../results/v1/artifacts/build_atlas/integrate_datasets/neighbors_leiden_umap_seed/leiden_paga_umap/blood-umap.h5ad",
)
bd_is_droplet = nxfvars.get(
    "bd_is_droplet",
    "../../results/v1/artifacts/build_atlas/integrate_datasets/seed/bd_rhapsody/artifacts/bd-is_droplet.csv"
)
marker_genes_path = nxfvars.get(
    "marker_genes_path",
    "../../tables/colon_atas_cell_type_marker_genes.csv",
)
artifact_dir = nxfvars.get(
    "artifact_dir",
    "/data/scratch/marteau/downloads/seed/",
)

# %%
adata = sc.read_h5ad(adata_path)
marker_genes = pd.read_csv(marker_genes_path)

# %% [markdown]
# ## 1. QC plots and general overview plots

# %%
# Append bd-adata is_droplet (bd-rhapsody)
is_droplet = pd.read_csv(bd_is_droplet)

adata.obs["is_droplet"] = (
    adata.obs_names.map(is_droplet.set_index("obs_names")["is_droplet"]).fillna("cell")
)

# %%
for col in ["cell_type_coarse", "cell_type_fine"]:    
    is_droplet_dict = is_droplet.set_index("obs_names")[col].to_dict()
    adata.obs[col] = adata.obs_names.map(is_droplet_dict)

# %%
# Basic QC (was set to "n_genes" < 100 for scAR denoising)
adata.obs["is_droplet2"] = np.where(
    (adata.obs["n_genes"] < 200)
    | (adata.obs["total_counts"] < 400),
    "empty droplet",
    "cell",
)

adata.obs.loc[
    (adata.obs["is_droplet2"] == "empty droplet"),
    "is_droplet"
] = "empty droplet"

# %%
# Keep Erythroid cells (bd datasets)
adata.obs.loc[
    (adata.obs["cell_type_fine"] == "Erythroid cell"),
    "is_droplet"
] = "cell"

# %%
# Remove bd-droplets
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
        #"dataset",
    ],
    save_plot=True,
    prefix="blood",
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
#    prefix="blood_",
#    file_ext="png",
#)

# %%
cluster_annot = {
    "Myeloid cell": [22, 30, 31, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 50, 52, 53, 54],
    "T cell": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 26, 27, 28, 29, 34, 44, 51],
    "B cell": [21, 23, 24, 25, 55],
    "Plasma cell": [49],
    "Epithelial cell": [32], # Circulating cancer cell
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
    fig.savefig(f"{artifact_dir}/blood-cell_type_coarse.png", bbox_inches="tight")

# %% [markdown]
# ## Reprocess major cell types

# %%
adata.obs["is_outlier"] = "cell"
adata.obs["cell_type_seed"] = "unknown"
adata.obs["cell_type_fine"] = "unknown"

# %% [markdown]
# ## Myeloid compartment

# %%
adata_myeloid = adata[adata.obs["cell_type_coarse"].isin(["Myeloid cell", "Epithelial cell"])].copy()
ah.pp.reprocess_adata_subset_scvi(adata_myeloid, leiden_res=2, n_neighbors=15, use_rep="X_scVI")

# %% [markdown]
# ### Myeloid marker genes

# %%
myeloid_marker = {
    "Cancer cell circulating": {
        "positive": [["EPCAM"]],
        "negative": [[]],
    },
    "T cell": {
        "positive": [["CD3E"], ["TRAC"]],
        "negative": [[]],
    },
    "Plasma cell": {
        "positive": [["JCHAIN"], ["IGHA1", "IGHA2"]],
        "negative": [[]],
    },
    "Granulocyte progenitor": {
        "positive": [["CD24"], ["CEACAM8"], ["ITGAM"], ["ARG1"]],
        "negative": [[]],
    },
    "Mast cell": {
        "positive": [
            ["TPSB2", "CPA3", "TPSAB1", "GATA2"],
        ],
        "negative": [[]],
    },
    "cDC1": {
        "positive": [
            ["CLEC9A", "XCR1", "CLNK", "CADM1", "WDFY4"],
        ],
        "negative": [
            ["VCAN", "CD14"],
            ["CXCR2", "FCGR3B"],
            ["EPCAM"],
        ],
    },
    "cDC2": {
        "positive": [
            ["CD1C", "FCER1A"],
            ["CLEC10A"]
            ],
        "negative": [["VCAN", "CD14"]],
    },
    "DC3": {
        "positive": [
            ["CD1C"],
            ["CD163", "CD14", "LYZ", "VCAN"],
        ],
        "negative": [["FCER1A"]],
    },
    "pDC": {
        "positive": [
            ["CLEC4C", "TCF4", "SMPD3", "ITM2C", "IL3RA"],
            ["GZMB"],
        ],
        "negative": [[]],
    },
    "Macrophage": {
        "positive": [
            ["C1QA", "C1QB"],
        ],
        "negative": [
            ["VCAN"],
        ],
    },
    "Monocyte": {
        "positive": [
            ["VCAN", "CD14"],
        ],
        "negative": [
            ["CD3E"],
            ["C1QA", "C1QB"],
            ["FCGR3B"],
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
    "Cancer cell circulating": [23, 27],
    "Granulocyte progenitor": [36, 41, 42, 47],
    "Mast cell": [35, 37],
    "Eosinophil": [33, 34, 39, 40],
    "Neutrophil": [24, 25, 26, 28, 29, 30, 31, 32, 43, 44, 45, 46],
    "Monocyte": [0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 15, 16, 18, 19, 21, 22, 38],
    "Macrophage": [10, 11, 12],
    "cDC2": [14],
    "pDC": [20],
    "Erythroid cell": [48],
    "Platelet": [13], # Thrombocyte
    "unknown": [17],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}

adata_myeloid.obs["cell_type_fine"] = (
    adata_myeloid.obs["leiden"].map(cluster_annot).fillna(adata_myeloid.obs["leiden"])
)

# %%
adata_myeloid.obs["cell_type_fine"] = adata_myeloid.obs["cell_type_fine"].astype(str)

adata_myeloid.obs.loc[
    (adata_myeloid.obs["cell_type_seed"] != "unknown")
& ~(adata_myeloid.obs["leiden"].isin(["24", "25", "28", "29", "30", "31", "33", "34", "39", "40", "43", "45", "46"])),
    "cell_type_fine",
] = adata_myeloid.obs["cell_type_seed"]

# %%
# Score Eosinophil/Neutrophil seperately
adata_myeloid.obs["cell_type_seed2"] = ah.pp.score_seeds(
    adata_myeloid,
    {
        "Eosinophil": {
            "positive": [
                ["CLC", "ADGRE1", "ALOX15"],
            ],
            "negative": [["CD24"], ["EPCAM"]],
        },
        "Neutrophil": {
            "positive": [
                ["CXCR2", "FCGR3B"],
            ],
            "negative": [["CD24"], ["VCAN"], ["C1QA", "C1QB"]],
        },
    },
    pos_cutoff=0.5,
    neg_cutoff=0,
)

# %%
adata_myeloid.obs["cell_type_fine"] = adata_myeloid.obs["cell_type_fine"].astype(str)
adata_myeloid.obs.loc[
    (adata_myeloid.obs["cell_type_seed2"] != "unknown")
& (adata_myeloid.obs["leiden"].isin(["23", "27", "35", "36", "37", "41", "42", "47", "48"])),
    "cell_type_fine",
] = adata_myeloid.obs["cell_type_seed2"]

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
    "Plasma cell": "Plasma cell",
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
adata_myeloid = adata_myeloid[~adata_myeloid.obs["cell_type_fine"].isin(["Plasma cell", "T cell", "Cancer cell circulating", "Erythroid cell", "Platelet", "empty droplet", "unknown"])].copy()
ah.pp.reprocess_adata_subset_scvi(adata_myeloid, leiden_res=2, n_neighbors=15, use_rep="X_scVI")

# %%
adata_myeloid.write(f"{artifact_dir}/blood-myeloid-adata.h5ad", compression="lzf")

# %%
adata_myeloid.obs["cell_type_fine"] = pd.Categorical(
    adata_myeloid.obs["cell_type_fine"],
    categories=["Monocyte", "Macrophage", "cDC1", "cDC2", "DC3", "pDC", "Granulocyte progenitor", "Neutrophil", "Eosinophil", "Mast cell"],
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
        groups=["Monocyte", "Macrophage", "cDC1", "cDC2", "DC3", "pDC", "DC mature", "Granulocyte progenitor", "Neutrophil", "Eosinophil", "Mast cell"],
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/blood-umap_myeloid_seeds.png", bbox_inches="tight")

# %% [markdown]
# ### Myeloid seeds dotplot

# %%
marker_genes = OrderedDict(
    [
        ("Monocyte", ["VCAN", "CD14", "CD163", "FCGR3A"]),
        ("Macrophage", ["C1QA", "C1QB"]),
        ("Pan-DC", ["FLT3"]),
        ("cDC1", ["CLEC9A", "XCR1", "CLNK", "CADM1", "WDFY4"]),
        ("cDC2", ["CD1C", "CD1E", "CLEC10A", "FCER1A"]),
        ("pDC", ["IL3RA", "ITM2C", "GZMB", "SMPD3", "TCF4", "BLNK"]),
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
            
    plt.savefig(f"{artifact_dir}/blood-dotplot_myeloid_seeds.pdf", bbox_inches="tight")

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
    "CD8": [9, 16, 28, 29, 30, 31, 32, 33, 36, 37, 38, 40, 41, 43, 44, 46, 47, 48, 50],
    "CD4": [0, 1, 2, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 35, 42, 51, 54, 55, 56, 58],
    "NK": [3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 22, 34, 52, 53, 57],
    "gamma-delta": [5, 39, 49],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}

adata_t.obs["cell_type_fine"] = (
    adata_t.obs["leiden"].map(cluster_annot).fillna(adata_t.obs["leiden"])
)

# %%
adata_sub = adata_t[adata_t.obs["leiden"] == "45"].copy()
sc.tl.leiden(adata_sub, 0.5, flavor="igraph")

adata_sub.obs["cell_type_fine"] = adata_sub.obs["cell_type_fine"].astype(str)
cluster_annot = {
    "CD8": [3],
    "CD4": [0, 1, 2, 4],
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
# Score naive T cells
adata_t.obs["cell_type_seed2"] = ah.pp.score_seeds(
    adata_t,
    {
    "naive": {
        "positive": [["CCR7", "LEF1", "SELL", "IL7R"]],
        "negative": [[]],
    },
    },
    pos_cutoff=1,
    neg_cutoff=0,
)

# %%
adata_t.obs["cell_type_fine"] = adata_t.obs["cell_type_fine"].astype(str)

adata_t.obs.loc[
    (adata_t.obs["cell_type_seed2"] != "unknown")
    & (adata_t.obs["cell_type_fine"].isin(["CD4", "CD8"])),
    "cell_type_fine",
] = adata_t.obs["cell_type_fine"] + " naive"

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
adata_t.write(f"{artifact_dir}/blood-t_cell-adata.h5ad", compression="lzf")

# %%
adata_t = adata_t[~adata_t.obs["cell_type_fine"].isin(["empty droplet", "unknown"])].copy()

adata_t.obs["cell_type_fine"] = pd.Categorical(
    adata_t.obs["cell_type_fine"],
    categories=["CD4", "Treg", "CD8", "NK", "ILC", "gamma-delta", "NKT", "CD4 naive", "CD8 naive", "CD4 cycling", "CD8 cycling", "NK cycling"],
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
        groups=["CD4", "Treg", "CD8", "NK", "ILC", "gamma-delta", "NKT", "CD4 naive", "CD8 naive", "CD4 cycling", "CD8 cycling", "NK cycling"],
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/blood-umap_t_cell_seeds.png", bbox_inches="tight")

# %% [markdown]
# ### T cell seeds dotplot

# %%
marker_genes = OrderedDict(
    [
        ("Pan-T cell", ["CD3E", "TRAC", "TRBC1", "TRBC2"]),
        ("CD4", ["CD4"]),
        ("Treg", ["FOXP3", "IL2RA"]),
        ("CD8", ["CD8A", "CD8B"]),
        ("T cell naive", ["CCR7", "LEF1", "SELL", "IL7R"]),
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
            
    plt.savefig(f"{artifact_dir}/blood-dotplot_t_cell_seeds.pdf", bbox_inches="tight")

# %%
adata_t.obs["cell_type_fine"].value_counts()

# %% [markdown]
# ## Plasma/B cell compartment

# %%
adata_plasma = adata[adata.obs["cell_type_coarse"].isin(["B cell", "Plasma cell"])].copy()
ah.pp.reprocess_adata_subset_scvi(adata_plasma, leiden_res=2, n_neighbors=15, use_rep="X_scVI")

# %%
cluster_annot = {
    "Monocyte": [35, 36, 39],
    "pDC": [38],
    "Plasma IgA": [37, 34],
    "Plasma IgG": [25, 26, 29, 30],
    "Plasmablast": [33],
    "B cell naive": [1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 22, 27, 32, 40],
    "B cell activated naive": [0, 13, 20, 28],
    "B cell memory": [8, 9, 10, 11, 12, 14, 21, 24, 31],
    "B cell activated": [],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}

adata_plasma.obs["cell_type_fine"] = (
    adata_plasma.obs["leiden"].map(cluster_annot).fillna(adata_plasma.obs["leiden"])
)

# %%
# Break up cycling cell cluster
adata_sub = adata_plasma[adata_plasma.obs["leiden"] == "23"].copy()
sc.tl.leiden(adata_sub, 0.5, flavor="igraph")

adata_sub.obs["cell_type_fine"] = adata_sub.obs["cell_type_fine"].astype(str)
cluster_annot = {
    "Plasma IgG": [2, 3, 5],
    "Plasmablast": [0, 1, 4, 6],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}
adata_sub.obs["cell_type_fine"] = adata_sub.obs["leiden"].map(cluster_annot).fillna(adata_sub.obs["leiden"])

ah.pp.integrate_back(adata_plasma, adata_sub, variable="cell_type_fine")

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

# %%
adata_plasma.obs.loc[
    (adata_plasma.obs["cell_type_seed"] != "unknown")
    & (adata_plasma.obs["leiden"].isin(["25", "26", "29", "30", "37"])),
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

adata_plasma.obs.loc[
    (adata_plasma.obs["cell_type_seed2"] != "unknown")
    & (adata_plasma.obs["leiden"].isin(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "14", "15", "16", "17", "18", "19", "20", "21", "22", "24", "27", "28", "31", "32", "40"])),
    "cell_type_fine",
] = adata_plasma.obs["cell_type_seed2"]

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
ah.pp.integrate_back(adata, adata_plasma, variable="cell_type_fine")


# %%
adata_plasma = adata_plasma[~adata_plasma.obs["cell_type_fine"].isin(["Monocyte", "pDC", "empty droplet"])].copy()
ah.pp.reprocess_adata_subset_scvi(adata_plasma, leiden_res=1, n_neighbors=15, use_rep="X_scVI")
adata_plasma.write(f"{artifact_dir}/blood-plasma_cell-adata.h5ad", compression="lzf")

# %%
adata_plasma.obs["cell_type_fine"] = pd.Categorical(
    adata_plasma.obs["cell_type_fine"],
    categories=["B cell naive", "B cell activated naive", "B cell activated", "B cell memory", "Plasma IgA", "Plasma IgG", "Plasma IgM", "Plasmablast"],
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
        groups=["B cell naive", "B cell activated naive", "B cell activated", "B cell memory", "Plasma IgA", "Plasma IgG", "Plasma IgM", "Plasmablast"],
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/blood-umap_plasma_seeds.png", bbox_inches="tight")

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
            
    plt.savefig(f"{artifact_dir}/blood-dotplot_plasma_cell_seeds.pdf", bbox_inches="tight")

# %%
adata_plasma.obs["cell_type_fine"].value_counts()

# %% [markdown]
# ## 2. Export blood seed cell types

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    fig = sc.pl.umap(
        adata,
        color="cell_type_fine",
        legend_fontoutline=1,
        legend_fontsize="xx-small",
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/blood-umap_cell_type_fine.png", bbox_inches="tight")

# %%
adata[adata.obs["cell_type_fine"] != "empty droplet"].shape

# %%
adata.obs["cell_type_fine"].value_counts(dropna=False)

# %%
# Save blood seed annotation
pd.DataFrame(adata.obs[["cell_type_fine"]]).reset_index(names=["obs_names"]).to_csv(
    f"{artifact_dir}/blood-cell_type_fine.csv", index=False
)
