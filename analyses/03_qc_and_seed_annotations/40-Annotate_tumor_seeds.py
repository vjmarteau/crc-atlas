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
# # Annotate tumor seed

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
    "../../results/v1/artifacts/build_atlas/integrate_datasets/neighbors_leiden_umap_seed/leiden_paga_umap/tumor-umap.h5ad",
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
# Plate-based protocols
adata_plate = adata[adata.obs["platform"].isin(["Microwell-seq", "SMARTer (C1)", "Smart-seq2", "scTrio-seq2_Dong_protocol", "scTrio-seq2_Tang_protocol"])].copy()
ah.pp.reprocess_adata_subset_scvi(adata_plate, leiden_res=2, n_neighbors=15, use_rep="X_scVI")

# %%
# Remove empty droplets
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
    "Epithelial cell": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 21, 59, 60, 62],
    "B cell": [19, 26, 27],
    "Plasma cell": [16, 17, 42, 43, 44, 47],
    "T cell": [7, 22, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 46, 48, 55],
    "Stromal cell": [18, 20, 23, 24, 25, 52, 53],
    "Myeloid cell": [28, 41, 45, 49, 50, 51, 54, 56, 57, 58, 61],
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
    fig.savefig(f"{artifact_dir}/tumor-umap_cell_type_coarse.png", bbox_inches="tight")

# %% [markdown]
# ## Reprocess major cell types

# %%
adata.obs["is_outlier"] = "cell"
adata.obs["cell_type_seed"] = "unknown"
adata.obs["cell_type_fine"] = "unknown"

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
            ["CD8A", "CD8B"],  # should express one of these 2 markers AND "CD3E" or "TRAC"
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

flattened_genes = list(
    chain.from_iterable(
        chain.from_iterable(
            gene_list
            for marker in t_cell_marker.values()
            for gene_list in marker.values()
        )
    )
)
# Remove duplicates
t_cell_all_markers = list(OrderedDict.fromkeys(flattened_genes))

# %%
adata_t.obs["cell_type_seed"] = ah.pp.score_seeds(
    adata_t,
    t_cell_marker,
    pos_cutoff=0,
    neg_cutoff=0.1,
)

# %%
# Set "cell_type_seed" to "unknown" for platform "DNBelab C4" -> very low T cell marker expression !!
adata_t.obs["cell_type_seed"] = adata_t.obs["cell_type_seed"].astype(str)
adata_t.obs.loc[
    (adata_t.obs["platform"] == "DNBelab C4"),
    "cell_type_seed"
] = "unknown"

# Dataset is enriched for gamma-delta -> Set "unknown" to gamma-delta
adata_t.obs.loc[
    (adata_t.obs["dataset"] == "deVries_2023_LUMC")
    & (adata_t.obs["cell_type_seed"] == "unknown"),
    "cell_type_seed"
] = "gamma-delta"

# %%
cluster_annot = {
    "CD8": [0, 5, 6, 8, 12, 13, 14, 15, 16, 18, 23, 27, 28, 29, 30, 31, 32, 34, 41, 42, 45, 47, 49, 50, 51, 53, 54, 57, 58, 59, 60, 61],
    "CD4": [9, 10, 11, 21, 22, 24, 35, 36, 38, 39, 40, 44, 45, 48, 56],
    "Treg": [1, 2, 3, 7, 26, 37, 52],
    "NK": [17, 43, 46, 55, 62],
    "ILC": [25, 33],
    "gamma-delta": [19, 20],
    "unknown": [4], # high mito + no marker gene expression! -> empty?!
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}

adata_t.obs["cell_type_fine"] = (
    adata_t.obs["leiden"].map(cluster_annot).fillna(adata_t.obs["leiden"])
)

# %%
# Break up smart-seq2 cell cluster
adata_sub = adata_t[adata_t.obs["leiden"] == "63"].copy()
sc.tl.leiden(adata_sub, 1, flavor="igraph")

adata_sub.obs["cell_type_fine"] = adata_sub.obs["cell_type_fine"].astype(str)
cluster_annot = {
    "CD8": [4, 2, 5, 3],
    "CD4": [9, 13, 1, 0, 6],
    "Treg": [7, 10, 8, 11],
    "CD8 cycling": [12],
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
# No expression of any of the t/nk cell markers?
adata_t.obs["tmp_t_cell"] = ah.pp.score_seeds(
    adata_t,
    {
        "T cell": {
            "positive": [t_cell_all_markers],
            "negative": [[]],
        },
     },
    pos_cutoff=0,
)

# %%
#adata_t.obs.loc[
#    (adata_t.obs["tmp_t_cell"] == "unknown"),
#    "cell_type_fine"
#] = "unknown"

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
adata_t.write(f"{artifact_dir}/tumor-t_cell-adata.h5ad", compression="lzf")

# %%
adata_t = adata_t[~adata_t.obs["cell_type_fine"].isin(["empty droplet", "unknown"])].copy()

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
    fig.savefig(f"{artifact_dir}/tumor-umap_t_cell_seeds.png", bbox_inches="tight")

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
            
    plt.savefig(f"{artifact_dir}/tumor-dotplot_t_cell_seeds.pdf", bbox_inches="tight")

# %%
adata_t.obs["cell_type_fine"].value_counts()

# %% [markdown]
# ## Plasma/B cell compartment

# %%
adata_plasma = adata[adata.obs["cell_type_coarse"].isin(["B cell", "Plasma cell"])].copy()
ah.pp.reprocess_adata_subset_scvi(adata_plasma, leiden_res=2, n_neighbors=15, use_rep="X_scVI")

# %% [markdown]
# ### B cell marker genes

# %%
b_cell_marker = {
    "B cell": {
        "positive": [
            ["CD79B", "CD19", "MS4A1"],
        ],
        "negative": [
            ["JCHAIN"], # Plasma cell marker
            ["IGHA1", "IGHA2"],
            ["IGHG1", "IGHG2", "IGHG3", "IGHG4"],
            ["IGHM"],
            ["ENO1"], # Plasmablast
            ["MKI67"],
        ],
    },
    "GC B cell": {
        "positive": [
            ["MARCKSL1", "GMDS", "S100A10"],
        ],
        "negative": [["JCHAIN"]],
    },
    "Plasma IgA": {
        "positive": [
            ["IGHA1", "IGHA2"],
            ["JCHAIN"],
        ],
        "negative": [
            ["CD79B", "CD19", "MS4A1"], # B cell marker
            ["IGHG1", "IGHG2", "IGHG3", "IGHG4"],
            ["IGHM"],
            ["ENO1"],
            ["MKI67"],
        ],
    },
    "Plasma IgG": {
        "positive": [
            ["IGHG1", "IGHG2", "IGHG3", "IGHG4"],
            ["JCHAIN"],
        ],
        "negative": [
            ["CD79B", "CD19", "MS4A1"],
            ["IGHA1", "IGHA2"],
            ["IGHM"],
            ["ENO1"],
            ["MKI67"],
        ],
    },
    "Plasma IgM": {
        "positive": [
            ["IGHM"],
            ["JCHAIN"],
        ],
        "negative": [
            ["CD79B", "CD19", "MS4A1"],
            ["IGHG1", "IGHG2", "IGHG3", "IGHG4"],
            ["IGHA1", "IGHA2"],
            ["ENO1"],
            ["MKI67"],
        ],
    },
    "Plasmablast": {
        "positive": [
            ["ENO1"],
            ["NME1"],
            ["HMGB2"],
            ["MYDGF"],
            ["JCHAIN"],
        ],
        "negative": [
            ["CD79B", "CD19", "MS4A1"],
            ["IGHG1", "IGHG2", "IGHG3", "IGHG4"],
            ["IGHA1", "IGHA2"],
        ],
    },
}

# %%
adata_plasma.obs["cell_type_seed"] = ah.pp.score_seeds(
    adata_plasma,
    b_cell_marker,
    pos_cutoff=0.1,
    neg_cutoff=0,
)

# %%
cluster_annot = {
    "Plasma IgA": [0, 1, 2, 4, 5, 6, 7, 8, 12, 13, 25, 31, 32, 33, 34, 35, 38, 39, 40],
    "Plasma IgG": [9, 10, 11, 26, 27, 28, 29],
    "Plasmablast": [3, 30, 37],
    "GC B cell": [24, 21],
    "B cell": [36, 22, 20, 15, 14, 23, 18, 17, 16, 19],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}

adata_plasma.obs["cell_type_fine"] = (
    adata_plasma.obs["leiden"].map(cluster_annot).fillna(adata_plasma.obs["leiden"])
)

# %%
adata_plasma.obs["cell_type_fine"] = adata_plasma.obs["cell_type_fine"].astype(str)

adata_plasma.obs.loc[
    (adata_plasma.obs["cell_type_seed"] != "unknown")
    & ~(adata_plasma.obs["leiden"].isin(["36", "22", "20", "15", "14", "23", "18", "17", "16", "19"])), # Don't have neg markers for B cells
    "cell_type_fine",
] = adata_plasma.obs["cell_type_seed"]

# Score IgG vs IgA with higher "pos_cutoff"
adata_plasma.obs["cell_type_seed2"] = ah.pp.score_seeds(
    adata_plasma,
    { "Plasma IgG": {
        "positive": [["IGHG1", "IGHG2", "IGHG3", "IGHG4"]],
        "negative": [[""]]
    },
     "Plasma IgA": {
        "positive": [["IGHA1", "IGHA2"]],
        "negative": [[""]]
    },
    },
    pos_cutoff=5,
    neg_cutoff=0,
)

adata_plasma.obs.loc[
    (adata_plasma.obs["cell_type_seed2"] != "unknown")
    # exclude plasmablast/b cell cluster
    & ~(adata_plasma.obs["leiden"].isin(["3", "30", "37", "24", "21", "36", "22", "20", "15", "14", "23", "18", "17", "16", "19"])),
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
# Closer look at B cells
adata_b = adata_plasma[adata_plasma.obs["cell_type_fine"].isin(["B cell"])].copy()
ah.pp.reprocess_adata_subset_scvi(adata_b, leiden_res=1, n_neighbors=15, use_rep="X_scVI")

# %%
# Score B cell subtypes
adata_b.obs["cell_type_seed2"] = ah.pp.score_seeds(
    adata_b,
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
cluster_annot = {
    "B cell naive": [10, 14],
    "B cell activated": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}

adata_b.obs["cell_type_fine"] = (
    adata_b.obs["leiden"].map(cluster_annot).fillna(adata_b.obs["leiden"])
)

# %%
adata_b.obs["cell_type_fine"] = adata_b.obs["cell_type_fine"].astype(str)

adata_b.obs.loc[
    (adata_b.obs["cell_type_seed2"] != "unknown"),
    "cell_type_fine",
] = adata_b.obs["cell_type_seed2"]

# %%
ah.pp.integrate_back(adata_plasma, adata_b, variable="cell_type_fine")
ah.pp.integrate_back(adata, adata_plasma, variable="cell_type_fine")

# %%
adata_plasma.write(f"{artifact_dir}/tumor-plasma_cell-adata.h5ad", compression="lzf")

# %%
adata_plasma = adata_plasma[~adata_plasma.obs["cell_type_fine"].isin(["empty droplet", "unknown"])].copy()

adata_plasma.obs["cell_type_fine"] = pd.Categorical(
    adata_plasma.obs["cell_type_fine"],
    categories=["GC B cell", "B cell naive", "B cell activated", "B cell memory", "Plasma IgA", "Plasma IgG", "Plasma IgM", "Plasmablast"],
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
        groups=["GC B cell", "B cell naive", "B cell activated", "B cell memory", "Plasma IgA", "Plasma IgG", "Plasma IgM", "Plasmablast"],
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/tumor-umap_plasma_seeds.png", bbox_inches="tight")

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
            
    plt.savefig(f"{artifact_dir}/tumor-dotplot_plasma_cell_seeds.pdf", bbox_inches="tight")

# %%
adata_plasma.obs["cell_type_fine"].value_counts()

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
        "negative": [["VCAN", "CD14"]]
        },
        "cDC2": {
        "positive": [
            ["CD1C", "FCER1A"],
            ["CLEC10A"],
        ],
        "negative": [["VCAN", "CD14"]]
        },
        "DC3": {
            "positive": [
                ["CD1C"],
                ["CD163", "CD14", "LYZ", "VCAN"],
            ],
            "negative": [["FCER1A"]]
        },
        "pDC": {
            "positive": [
                ["CLEC4C", "TCF4", "SMPD3", "ITM2C", "IL3RA"],
                ["GZMB"],
            ],
            "negative": [[]]
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
            "negative": [["VCAN"]],
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
    "Granulocyte progenitor": [47],
    "Eosinophil": [39, 12],
    "Neutrophil": [40, 38, 42, 37, 36, 35, 34],
    "Mast cell": [14, 15, 13, 48, 46],
    "cDC progenitor": [41, 44, 45],
    "cDC1": [6, 43],
    "cDC2": [1, 4, 11, 21],
    "DC mature": [5, 27],
    "pDC": [25],
    "Macrophage": [0, 7, 9, 10, 16, 17, 18, 19, 20, 22, 23, 24, 32],
    "Monocyte": [2, 3, 8, 28, 29, 30, 31, 33],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}

adata_myeloid.obs["cell_type_fine"] = (
    adata_myeloid.obs["leiden"].map(cluster_annot).fillna(adata_myeloid.obs["leiden"])
)

# %%
# Break up cell cluster leiden 26
adata_sub = adata_myeloid[adata_myeloid.obs["leiden"] == "26"].copy()
sc.tl.leiden(adata_sub, 0.5, flavor="igraph")

adata_sub.obs["cell_type_fine"] = adata_sub.obs["cell_type_fine"].astype(str)
cluster_annot = {
    "Mast cell": [3, 4, 5],
    "unknown": [1, 0, 2],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}
adata_sub.obs["cell_type_fine"] = adata_sub.obs["leiden"].map(cluster_annot).fillna(adata_sub.obs["leiden"])

ah.pp.integrate_back(adata_myeloid, adata_sub, variable="cell_type_fine")

# Break up cell cluster leiden 12
adata_sub = adata_myeloid[adata_myeloid.obs["leiden"] == "12"].copy()
sc.tl.leiden(adata_sub, 0.5, flavor="igraph")

adata_sub.obs["cell_type_fine"] = ah.pp.score_seeds(
    adata_sub,
    {
        "Eosinophil": {
            "positive": [["CLC", "CCR3", "ADGRE1", "ALOX15"]],
            "negative": [[]],
        },
        "T cell": {
            "positive": [["CD3E", "TRAC"]],
            "negative": [["CLC", "CCR3", "ADGRE1", "ALOX15"]],
        },
    },
    pos_cutoff=[0.1, 1],
    neg_cutoff=[0.1, 0],
    unidentified_label="Eosinophil",
)

ah.pp.integrate_back(adata_myeloid, adata_sub, variable="cell_type_fine")

# %%
adata_myeloid.obs["cell_type_fine"] = adata_myeloid.obs["cell_type_fine"].astype(str)
adata_myeloid.obs["cell_type_fine"] = adata_myeloid.obs["cell_type_fine"].str.replace("T cell", "unknown")

adata_myeloid.obs.loc[
    (adata_myeloid.obs["cell_type_seed"] != "unknown")
  & ~(adata_myeloid.obs["leiden"].isin(["12", "13", "14", "15", "34", "35", "36", "37", "38", "39", "40", "41", "42", "44", "45", "46", "47", "48"])),
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
ah.pp.integrate_back(adata, adata_myeloid, variable="cell_type_fine")

# %%
adata_myeloid.write(f"{artifact_dir}/tumor-myeloid-adata.h5ad", compression="lzf")

# %%
adata_myeloid = adata_myeloid[~adata_myeloid.obs["cell_type_fine"].isin(["empty droplet", "unknown"])].copy()

adata_myeloid.obs["cell_type_fine"] = pd.Categorical(
    adata_myeloid.obs["cell_type_fine"],
    categories=["Monocyte", "Macrophage", "Macrophage cycling", "cDC progenitor", "cDC1", "cDC2", "DC3", "pDC", "DC mature", "Granulocyte progenitor", "Neutrophil", "Eosinophil", "Mast cell"],
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
        groups=["Monocyte", "Macrophage", "Macrophage cycling", "cDC progenitor", "cDC1", "cDC2", "DC3", "pDC", "DC mature", "Granulocyte progenitor", "Neutrophil", "Eosinophil", "Mast cell"],
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/tumor-umap_myeloid_seeds.png", bbox_inches="tight")

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
        ("pDC", ["IL3RA", "ITM2C", "GZMB", "SMPD3", "TCF4", "BLNK"]),
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
            
    plt.savefig(f"{artifact_dir}/tumor-dotplot_myeloid_seeds.pdf", bbox_inches="tight")

# %%
adata_myeloid.obs["cell_type_fine"].value_counts()

# %% [markdown]
 ## Epithelial cell compartment

# %%
adata_epi = adata[adata.obs["cell_type_coarse"].isin(["Epithelial cell"])].copy()
ah.pp.reprocess_adata_subset_scvi(adata_epi, leiden_res=2, n_neighbors=15, use_rep="X_scVI")

# %% [markdown]
# ### Epithelial cell marker genes

# %%
epi_cell_marker = {
    "Tuft": {
        "positive": [["IRAG2", "SH2D6", "ALOX5AP", "AVIL"]],
        "negative": [],
    },
    "Enteroendocrine": {
        "positive": [["PCSK1N", "FEV", "SCGN", "NEUROD1", "SCG5", "CHGA", "CPE"]],
        "negative": [
            ["MUC2", "FCGBP", "ZG16", "ATOH1"],
            ["FABP1", "CEACAM7", "ANPEP", "SI"],
        ],
    },
    "Tumor Goblet-like": {
        "positive": [["MUC2", "FCGBP", "ZG16", "ATOH1"]],
        "negative": [["FABP1", "CEACAM7", "ANPEP", "SI"]],
    },
    "Tumor BEST4": {
        "positive": [["BEST4", "OTOP2", "SPIB"]],
        "negative": [["IRAG2", "SH2D6", "ALOX5AP", "AVIL"]],
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
    "Tuft": [14],
    "Enteroendocrine": [27],
    "Tumor Colonocyte-like": [3, 4, 7, 9, 12, 17, 18, 20, 21, 25, 28, 29, 33, 34, 35, 36, 37, 38, 40, 43, 44, 50],
    "Tumor Goblet-like": [19, 30, 31, 32, 39, 42, 46, 47],
    "Tumor BEST4": [45],
    "Tumor Crypt-like": [1, 5, 6, 8, 11, 15, 16, 41],
    "Tumor TA-like": [0, 2, 10, 13, 22, 23, 24, 26],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}

adata_epi.obs["cell_type_fine"] = (
    adata_epi.obs["leiden"].map(cluster_annot).fillna(adata_epi.obs["leiden"])
)

# %%
# Break up smart-seq2 cell cluster -> super high pct mito!
adata_sub = adata_epi[adata_epi.obs["leiden"].isin(["48", "49"])].copy()
sc.tl.leiden(adata_sub, 0.5, flavor="igraph")

adata_sub.obs["cell_type_fine"] = adata_sub.obs["cell_type_fine"].astype(str)
cluster_annot = {
    "Tumor Colonocyte-like": [5, 3, 6, 4, 7, 0],
    "unknown": [1, 2], # (low) CD3E and VCAN Pos; EPCAM neg ??
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
adata_epi.write(f"{artifact_dir}/tumor-epi-adata.h5ad", compression="lzf")

# %%
adata_epi = adata_epi[~adata_epi.obs["cell_type_fine"].isin(["empty droplet", "unknown"])].copy()

adata_epi.obs["cell_type_fine"] = pd.Categorical(
    adata_epi.obs["cell_type_fine"],
    categories=["Tumor Colonocyte-like", "Tumor BEST4", "Tumor Goblet-like", "Tumor Crypt-like", "Tumor TA-like", "Tuft", "Enteroendocrine"],
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
        groups=["Tumor Colonocyte-like", "Tumor BEST4", "Tumor Goblet-like", "Tumor Crypt-like", "Tumor TA-like", "Tuft", "Enteroendocrine"],
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/tumor-umap_epi_seeds.png", bbox_inches="tight")

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
        ("Tuft", ["IRAG2", "SH2D6", "ALOX5AP", "AVIL"]),
        ("Enteroendocrine", ["PCSK1N", "SCG5", "CHGA", "CPE"]),
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
            
    plt.savefig(f"{artifact_dir}/tumor-dotplot_epi_seeds.pdf", bbox_inches="tight")

# %%
adata_epi.obs["cell_type_fine"].value_counts()

# %% [markdown]
# ## Stromal cell compartment

# %%
adata_stromal = adata[adata.obs["cell_type_coarse"].isin(["Stromal cell"])].copy()
ah.pp.reprocess_adata_subset_scvi(adata_stromal, leiden_res=2, n_neighbors=15, use_rep="X_scVI")

# %% [markdown]
# ### Stromal cell marker genes

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
    "Schwann cell": {
        "positive": [["S100B", "PLP1"]],
        "negative": [["MKI67"]],
    },
    "Fibroblast S1": {
        "positive": [["ADAMDEC1"], ["FABP5"], ["APOE"]],
        "negative": [["MKI67"]],
    },
    "Fibroblast S2": {
        "positive": [["F3", "AGT", "NRG1", "SOX6"]],
        "negative": [
            ["ADAMDEC1", "FABP5", "APOE"], # Fibroblast S1
            ["C7", "GREM1"], # Fibroblast S3
            ["S100B", "PLP1"], # Schwann cell
            ["NDUFA4L2", "RGS5"], # Pericyte
            ["PECAM1", "CDH5", "VWF", "KDR", "FLT1", "PROX1", "TFF3"], # Endothelial
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
    "Schwann cell": [34, 45],
    "Pericyte": [11, 18, 25, 35, 36, 37, 38, 42, 43],
    "Endothelial lymphatic": [24, 28, 49],
    "Endothelial venous": [22, 26, 41],
    "Endothelial arterial": [1, 23, 27, 29, 30, 31, 40, 47, 50],
    "Fibroblast S1": [2, 20, 21, 33],
    "Fibroblast S2": [3, 6, 9, 12, 19, 32, 46],
    "Fibroblast S3": [0, 4, 5, 7, 8, 10, 13, 14, 15, 16, 17, 39, 44, 48],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}

adata_stromal.obs["cell_type_fine"] = (
    adata_stromal.obs["leiden"].map(cluster_annot).fillna(adata_stromal.obs["leiden"])
)

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
ah.pp.integrate_back(adata, adata_stromal, variable="cell_type_fine")

# %%
adata_stromal.write(f"{artifact_dir}/tumor-stromal-adata.h5ad", compression="lzf")

# %%
adata_stromal = adata_stromal[~adata_stromal.obs["cell_type_fine"].isin(["empty droplet"])].copy()

adata_stromal.obs["cell_type_fine"] = pd.Categorical(
    adata_stromal.obs["cell_type_fine"],
    categories=["Endothelial venous", "Endothelial arterial", "Endothelial lymphatic", "Fibroblast S1", "Fibroblast S2", "Fibroblast S3", "Pericyte", "Schwann cell"],
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
        groups=["Fibroblast S1", "Fibroblast S2", "Fibroblast S3", "Endothelial lymphatic", "Endothelial arterial", "Endothelial venous", "Pericyte", "Schwann cell"],
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/tumor-umap_stromal_seeds.png", bbox_inches="tight")

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
            
    plt.savefig(f"{artifact_dir}/tumor-dotplot_stromal_seeds.pdf", bbox_inches="tight")

# %%
adata_stromal.obs["cell_type_fine"].value_counts()

# %% [markdown]
# ### Seperate look at plate-based protocols

# %%
adata_plate.obs["cell_type_fine"] = (
    adata_plate.obs_names.map(adata.obs["cell_type_fine"]).fillna("unknown")
)

# Clean up plate-based seeds
adata_plate.obs["cell_type_fine"] = adata_plate.obs["cell_type_fine"].astype(str)

adata_plate.obs.loc[
    (adata_plate.obs["leiden"].isin(["1", "4", "6", "20"]))
    & (adata_plate.obs["cell_type_fine"] == "unknown"),
    "cell_type_fine"
] = "Tumor Colonocyte-like"

adata_plate.obs.loc[
    (adata_plate.obs["leiden"] == "5")
    & (adata_plate.obs["cell_type_fine"] == "Pericyte"),
    "cell_type_fine"
] = "Plasma IgA"

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    fig = sc.pl.umap(
        adata_plate,
        color="cell_type_fine",
        legend_fontoutline=1,
        legend_fontsize="xx-small",
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/tumor-umap_plate-based-cell_type_fine.png", bbox_inches="tight")


# %%
adata_plate.obs["cell_type_fine"].value_counts()

# %%
adata_plate = adata_plate[adata_plate.obs_names.isin(adata.obs_names)]
ah.pp.integrate_back(adata, adata_plate, variable="cell_type_fine")

# %% [markdown]
# ## 2. Export tumor seed cell types

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    fig = sc.pl.umap(
        adata,
        color="cell_type_fine",
        legend_fontoutline=1,
        legend_fontsize="xx-small",
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/tumor-umap_cell_type_fine.png", bbox_inches="tight")

# %%
adata[adata.obs["cell_type_fine"] != "empty droplet"].shape

# %%
adata.obs["cell_type_fine"].value_counts(dropna=False)

# %%
# Save tumor seed annotation
pd.DataFrame(adata.obs[["cell_type_fine"]]).reset_index(names=["obs_names"]).to_csv(
    f"{artifact_dir}/tumor-cell_type_fine.csv", index=False
)

# %% [markdown]
# ### Compute wilcoxon diff genes per cell_type_fine

# %%
#for adata, label in zip(
#    [adata_t, adata_plasma, adata_myeloid, adata_epi, adata_stromal],
#    ["t_cell", "plasma_cell", "myeloid", "epi", "stromal"]
#):
#    sc.tl.rank_genes_groups(
#        adata,
#        groupby="cell_type_fine",
#        method="wilcoxon",
#        layer="log1p_norm",
#        key_added="dea_cell_type_fine",
#        #    min_fold_change=3,
#    )
    
    # Save wilcox results by group: cell type
#    genes_groups_dict = {
#        ct: sc.get.rank_genes_groups_df(
#            adata, group=ct, key="dea_cell_type_fine"
#        )
#        for ct in adata.obs["cell_type_fine"].unique()
#    }
#    de_res_all = pd.concat(
#        [df.assign(cell_type=ct) for ct, df in genes_groups_dict.items()]
#    )
    
#    gtf = adata.var.reset_index(names="symbol")
#    de_res_all["Class"] = de_res_all["names"].map(gtf.set_index("symbol")["Class"])
#    de_res_all["Geneid"] = de_res_all["names"].map(gtf.set_index("symbol")["Geneid"])
#    de_res_all = de_res_all.dropna(subset=["names"]).rename(columns={"names": "symbol"})
#    de_res_all = de_res_all[["cell_type", "symbol", "scores", "logfoldchanges", "pvals", "pvals_adj", "Class", "Geneid"]]
#    de_res_all = de_res_all[de_res_all["pvals_adj"] <= 0.1]
#    de_res_all.to_csv(f"{artifact_dir}/tumor-{label}_wilcox_table.tsv", sep='\t', index=False)
