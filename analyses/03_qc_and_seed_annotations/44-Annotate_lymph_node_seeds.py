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
# # Annotate lymph_node seed

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
    "/data/projects/2022/CRCA/results/v1/artifacts/build_atlas/load_datasets/harmonize_datasets/artifacts/lymph_node-adata.h5ad",
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
adata_plate = adata[adata.obs["platform"].isin(["Microwell-seq", "SMARTer (C1)", "Smart-seq2", "scTrio-seq2_Dong_protocol", "scTrio-seq2_Tang_protocol"])].copy()
ah.pp.reprocess_adata_subset_scvi(adata_plate, leiden_res=2, n_neighbors=15, use_rep="X_scVI")

# %%
# Remove droplets
adata = adata[adata.obs["is_droplet"] == "cell"].copy()
ah.pp.reprocess_adata_subset_scvi(adata, leiden_res=2, n_neighbors=15, use_rep="X_scVI")

# %%
ah.pl.umap_qc_metrics(adata, vmax_total_counts=20000, vmax_n_genes=2000, save_plot=True, prefix="lymph_node", save_dir=artifact_dir)

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
    prefix="lymph_node",
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
    "Epithelial cell": [0],
    "B cell": [4, 5, 6, 28, 35, 36, 37, 38, 39, 40, 41, 43],
    "Plasma cell": [16, 48],
    "T cell": [2, 3, 7, 8, 9, 10, 11, 12, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 31, 44],
    "Stromal cell": [17, 29, 30, 32, 33],
    "Myeloid cell": [1, 34, 42, 45, 46, 47],
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
    fig.savefig(f"{artifact_dir}/lymph_node-cell_type_coarse.png", bbox_inches="tight")

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
    "Macrophage": [0, 1, 2, 11, 12, 13, 14, 18, 20, 21, 22, 23],
    "Monocyte": [3, 9, 10, 16],
    "cDC1": [15, 17],
    "cDC2": [6, 8],
    "pDC": [7],
    "DC mature": [4, 5],
    "Mast cell": [19],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}

adata_myeloid.obs["cell_type_fine"] = (
    adata_myeloid.obs["leiden"].map(cluster_annot).fillna(adata_myeloid.obs["leiden"])
)

# %%
adata_myeloid.obs["cell_type_fine"] = adata_myeloid.obs["cell_type_fine"].astype(str)

adata_myeloid.obs.loc[
    (adata_myeloid.obs["cell_type_seed"] != "unknown")
  & ~(adata_myeloid.obs["leiden"].isin(["5", "7", "17", "19"])),
    "cell_type_fine",
] = adata_myeloid.obs["cell_type_seed"]

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
adata_myeloid.write(f"{artifact_dir}/lymph_node-myeloid-adata.h5ad", compression="lzf")

# %%
adata_myeloid = adata_myeloid[~adata_myeloid.obs["cell_type_fine"].isin(["empty droplet"])].copy()

adata_myeloid.obs["cell_type_fine"] = pd.Categorical(
    adata_myeloid.obs["cell_type_fine"],
    categories=["Monocyte", "Macrophage", "cDC1", "cDC2", "pDC", "DC mature", "Mast cell"],
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
        groups=["Monocyte", "Macrophage", "cDC1", "cDC2", "pDC", "DC mature", "Mast cell"],
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/lymph_node-umap_myeloid_seeds.png", bbox_inches="tight")

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
        ("DC mature", ["CCL22", "CCR7", "LAMP3"]),
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
            
    plt.savefig(f"{artifact_dir}/lymph_node-dotplot_myeloid_seeds.pdf", bbox_inches="tight")

# %%
adata_myeloid.obs["cell_type_fine"].value_counts()

# %% [markdown]
# ## T cell compartment

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
            ["FCGR3A", "TYROBP"],
            ["TRDC", "TRGC2"],
            ["VCAN", "CD14"],
            ["MKI67"],
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
            ["FOXP3"],
            ["FCGR3A", "TYROBP"],
            ["VCAN", "CD14"],
            ["MKI67"],
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
}

# %%
# colorectal cancer samples
adata_t_crc = adata[
    (adata.obs["cell_type_coarse"].isin(["T cell"]))
    & (adata.obs["medical_condition"].isin(["colorectal cancer"]))
].copy()

ah.pp.reprocess_adata_subset_scvi(adata_t_crc, leiden_res=3, n_neighbors=15, use_rep="X_scVI")

# %%
adata_t_crc.obs["cell_type_seed"] = ah.pp.score_seeds(
    adata_t_crc,
    t_cell_marker,
    pos_cutoff=0,
    neg_cutoff=0.1,
)

# %%
cluster_annot = {
    "CD8": [6, 7, 10, 11, 21, 30, 31, 33],
    "CD4": [0, 1, 3, 5, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 27, 29, 30, 32, 34, 35, 36, 38],
    "CD4 cycling": [39],
    "Treg": [28],
    "CD4 stem-like": [2, 4, 41, 42],
    "CD8 stem-like": [37],
    "NK": [26],
    "ILC": [40],
}

cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}

adata_t_crc.obs["cell_type_fine"] = (
    adata_t_crc.obs["leiden"].map(cluster_annot).fillna(adata_t_crc.obs["leiden"])
)

# %%
adata_t_crc.obs["cell_type_fine"] = adata_t_crc.obs["cell_type_fine"].astype(str)

adata_t_crc.obs.loc[
    (adata_t_crc.obs["cell_type_seed"] != "unknown")
    & ~(adata_t_crc.obs["leiden"].isin(["2", "4", "37", "41", "42"])),
    "cell_type_fine",
] = adata_t_crc.obs["cell_type_seed"]

ah.pp.integrate_back(adata, adata_t_crc, variable="cell_type_fine")

# %%
# healthy samples
adata_t_healthy = adata[
    (adata.obs["cell_type_coarse"].isin(["T cell"]))
    & (adata.obs["medical_condition"].isin(["healthy"]))
].copy()

ah.pp.reprocess_adata_subset_scvi(adata_t_healthy, leiden_res=3, n_neighbors=15, use_rep="X_scVI")

# %%
adata_t_healthy.obs["cell_type_seed"] = ah.pp.score_seeds(
    adata_t_healthy,
    t_cell_marker,
    pos_cutoff=0,
    neg_cutoff=0.1,
)

# %%
adata_t_healthy.obs["cell_type_seed"] = adata_t_healthy.obs["cell_type_seed"].map(
    {
        "CD4": "CD4 stem-like",
        "CD4 cycling": "CD4 stem-like",
        "Treg": "Treg",
        "CD8": "CD8 stem-like",
        "CD8 cycling": "CD8 stem-like",
        "NK": "NK",
        "unknown": "unknown",
    }
)

# %%
cluster_annot = {
    "CD4 stem-like": [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20, 21, 24, 28, 30, 31, 33, 34, 35, 36, 37, 38, 40, 42, 43, 44, 45, 47, 48, 49, 51, 52],
    "Treg": [13, 14, 25, 26, 29],
    "CD8 stem-like": [0, 2, 3, 15, 16, 23, 27, 32, 41, 46, 50],
    "NK": [39],
    "ILC": [22],
}

cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}

adata_t_healthy.obs["cell_type_fine"] = (
    adata_t_healthy.obs["leiden"].map(cluster_annot).fillna(adata_t_healthy.obs["leiden"])
)

# %%
adata_t_healthy.obs["cell_type_fine"] = adata_t_healthy.obs["cell_type_fine"].astype(str)

adata_t_healthy.obs.loc[
    (adata_t_healthy.obs["cell_type_seed"] != "unknown"),
    "cell_type_fine",
] = adata_t_healthy.obs["cell_type_seed"]

ah.pp.integrate_back(adata, adata_t_healthy, variable="cell_type_fine")


# %%
adata_t = adata[adata.obs["cell_type_coarse"].isin(["T cell"])].copy()
ah.pp.reprocess_adata_subset_scvi(adata_t, leiden_res=3, n_neighbors=15, use_rep="X_scVI")

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
adata_t.obs["cell_type_fine"] = adata_t.obs["cell_type_fine"].astype(str)

adata_t.obs.loc[
    (adata_t.obs["is_droplet"] == "empty droplet")
    & ~(adata_t.obs["platform"].isin(["Microwell-seq", "SMARTer (C1)", "Smart-seq2", "scTrio-seq2_Dong_protocol", "scTrio-seq2_Tang_protocol", "TruDrop"])),
    "cell_type_fine",
] = "empty droplet"

# %%
adata_t.write(f"{artifact_dir}/lymph_node-t_cell-adata.h5ad", compression="lzf")

# %%
adata_t = adata_t[~adata_t.obs["cell_type_fine"].isin(["empty droplet"])].copy()

adata_t.obs["cell_type_fine"] = pd.Categorical(
    adata_t.obs["cell_type_fine"],
    categories=["CD4", "Treg", "CD8", "NK", "ILC", "CD4 stem-like", "CD8 stem-like", "CD4 cycling", "CD8 cycling"],
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
        groups=["CD4", "Treg", "CD8", "NK", "ILC", "CD4 stem-like", "CD8 stem-like", "CD4 cycling", "CD8 cycling"],
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/lymph_node-umap_t_cell_seeds.png", bbox_inches="tight")

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
        ("ILC", ["ICOS", "IL4I1", "KIT", "LST1", "RORC", "IL22", "IL1R1", "IL23R", "LMNA"]),
        ("T cell stem-like", ["CCR7", "LEF1", "SELL", "IL7R"]),
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
            
    plt.savefig(f"{artifact_dir}/lymph_node-dotplot_t_cell_seeds.pdf", bbox_inches="tight")

# %%
adata_t.obs["cell_type_fine"].value_counts()

# %% [markdown]
# ## Plasma/B cell compartment

# %% [markdown]
# ### B cell marker genes

# %%
b_cell_marker = {
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
}

# %%
# colorectal cancer samples
adata_plasma_crc = adata[
    (adata.obs["cell_type_coarse"].isin(["B cell", "Plasma cell"]))
    & (adata.obs["medical_condition"].isin(["colorectal cancer"]))
].copy()

ah.pp.reprocess_adata_subset_scvi(adata_plasma_crc, leiden_res=1, n_neighbors=15, use_rep="X_scVI")

# %%
# Score B cell subtypes
adata_plasma_crc.obs["cell_type_seed"] = ah.pp.score_seeds(
    adata_plasma_crc,
    b_cell_marker,
    pos_cutoff=0.3,
    neg_cutoff=0.3,
)

# %%
cluster_annot = {
    "B cell naive": [5, 6, 7, 9, 12, 13],
    "B cell activated": [2, 3, 4, 8, 10],
    "GC B cell": [1],
    "Plasma IgA": [11, 14],
    "Plasma IgG": [0],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}

adata_plasma_crc.obs["cell_type_fine"] = (
    adata_plasma_crc.obs["leiden"].map(cluster_annot).fillna(adata_plasma_crc.obs["leiden"])
)

adata_sub = adata_plasma_crc[adata_plasma_crc.obs["leiden"] == "9"].copy()
sc.tl.leiden(adata_sub, 0.5, flavor="igraph")

adata_sub.obs["cell_type_fine"] = adata_sub.obs["cell_type_fine"].astype(str)
cluster_annot = {
    "B cell naive": [4],
    "B cell activated": [1, 2, 3, 5],
    "GC B cell": [0],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}
adata_sub.obs["cell_type_fine"] = adata_sub.obs["leiden"].map(cluster_annot).fillna(adata_sub.obs["leiden"])

ah.pp.integrate_back(adata_plasma_crc, adata_sub, variable="cell_type_fine")

# %%
adata_plasma_crc.obs["cell_type_fine"] = adata_plasma_crc.obs["cell_type_fine"].astype(str)

adata_plasma_crc.obs.loc[
    (adata_plasma_crc.obs["cell_type_seed"] != "unknown")
    & (adata_plasma_crc.obs["leiden"].isin(["2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "13"])),
    "cell_type_fine",
] = adata_plasma_crc.obs["cell_type_seed"]

# %%
ah.pp.integrate_back(adata, adata_plasma_crc, variable="cell_type_fine")

# %%
# healthy samples
adata_plasma_healthy = adata[
    (adata.obs["cell_type_coarse"].isin(["B cell", "Plasma cell"]))
    & (adata.obs["medical_condition"].isin(["healthy"]))
].copy()

ah.pp.reprocess_adata_subset_scvi(adata_plasma_healthy, leiden_res=1, n_neighbors=15, use_rep="X_scVI")

# %%
# Score B cell subtypes
adata_plasma_healthy.obs["cell_type_seed"] = ah.pp.score_seeds(
    adata_plasma_healthy,
    b_cell_marker,
    pos_cutoff=0.3,
    neg_cutoff=0.3,
)

# %%
cluster_annot = {
    "B cell naive": [4, 5, 9, 10, 14],
    "B cell activated": [3, 6, 12, 13, 16],
    "B cell memory": [7, 11, 15],
    "GC B cell": [2, 8],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}

adata_plasma_healthy.obs["cell_type_fine"] = (
    adata_plasma_healthy.obs["leiden"].map(cluster_annot).fillna(adata_plasma_healthy.obs["leiden"])
)

# %%
adata_sub = adata_plasma_healthy[adata_plasma_healthy.obs["leiden"] == "1"].copy()
sc.tl.leiden(adata_sub, 1, flavor="igraph")

adata_sub.obs["cell_type_fine"] = adata_sub.obs["cell_type_fine"].astype(str)
cluster_annot = {
    "B cell memory": [0, 1, 5, 6, 7, 11],
    "B cell activated": [2, 3, 4, 9, 10],
    "GC B cell": [8],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}
adata_sub.obs["cell_type_fine"] = adata_sub.obs["leiden"].map(cluster_annot).fillna(adata_sub.obs["leiden"])

ah.pp.integrate_back(adata_plasma_healthy, adata_sub, variable="cell_type_fine")

# %%
adata_sub = adata_plasma_healthy[adata_plasma_healthy.obs["leiden"] == "0"].copy()
sc.tl.leiden(adata_sub, 0.5, flavor="igraph")

adata_sub.obs["cell_type_fine"] = adata_sub.obs["cell_type_fine"].astype(str)
cluster_annot = {
    "GC B cell": [1, 2, 3, 4],
    "Plasma IgA": [0],
    "Plasma IgG": [5],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}
adata_sub.obs["cell_type_fine"] = adata_sub.obs["leiden"].map(cluster_annot).fillna(adata_sub.obs["leiden"])

ah.pp.integrate_back(adata_plasma_healthy, adata_sub, variable="cell_type_fine")

# %%
adata_plasma_healthy.obs["cell_type_fine"] = adata_plasma_healthy.obs["cell_type_fine"].astype(str)

adata_plasma_healthy.obs.loc[
    (adata_plasma_healthy.obs["cell_type_seed"] != "unknown"),
    "cell_type_fine",
] = adata_plasma_healthy.obs["cell_type_seed"]

# %%
ah.pp.integrate_back(adata, adata_plasma_healthy, variable="cell_type_fine")

# %%
adata_plasma = adata[adata.obs["cell_type_coarse"].isin(["B cell", "Plasma cell"])].copy()
ah.pp.reprocess_adata_subset_scvi(adata_plasma, leiden_res=2, n_neighbors=15, use_rep="X_scVI")

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
adata_plasma.obs["cell_type_fine"] = adata_plasma.obs["cell_type_fine"].astype(str)

# %%
adata_plasma.obs.loc[
    (adata_plasma.obs["is_droplet"] == "empty droplet")
    & ~(adata_plasma.obs["platform"].isin(["Microwell-seq", "SMARTer (C1)", "Smart-seq2", "scTrio-seq2_Dong_protocol", "scTrio-seq2_Tang_protocol", "TruDrop"])),
    "cell_type_fine",
] = "empty droplet"

# %%
adata_plasma.write(f"{artifact_dir}/lymph_node-plasma_cell-adata.h5ad", compression="lzf")

# %%
adata_plasma = adata_plasma[~adata_plasma.obs["cell_type_fine"].isin(["empty droplet"])].copy()

adata_plasma.obs["cell_type_fine"] = pd.Categorical(
    adata_plasma.obs["cell_type_fine"],
    categories=["GC B cell", "B cell naive", "B cell activated", "B cell memory", "Plasma IgA", "Plasma IgG"],
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
        groups=["GC B cell", "B cell naive", "B cell activated", "B cell memory", "Plasma IgA", "Plasma IgG"],
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/lymph_node-umap_plasma_seeds.png", bbox_inches="tight")

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
            
    plt.savefig(f"{artifact_dir}/lymph_node-dotplot_plasma_cell_seeds.pdf", bbox_inches="tight")

# %%
adata_plasma.obs["cell_type_fine"].value_counts()

# %% [markdown]
# ## Stromal/Epithelial cell compartment

# %%
adata_stromal = adata[adata.obs["cell_type_coarse"].isin(["Stromal cell", "Epithelial cell"])].copy()
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
    "Fibroblast S1": {
        "positive": [["ADAMDEC1"], ["FABP5"], ["APOE"]],
        "negative": [["MKI67"]],
    },
    "Fibroblast S2": {
        "positive": [["F3", "AGT", "NRG1", "SOX6"]],
        "negative": [
            ["ADAMDEC1", "FABP5", "APOE"], # Fibroblast S1
            ["C7", "GREM1"], # Fibroblast S3
            ["NDUFA4L2", "RGS5"], # Pericyte
            ["PECAM1", "CDH5", "VWF", "KDR", "FLT1", "PROX1", "TFF3"], # Endothelial
            ["MKI67"],
        ],
    },
    "Fibroblast S3": {
        "positive": [["C7", "ASPN", "CCDC80", "GREM1"]],
        "negative": [
            ["F3", "AGT", "NRG1", "SOX6"], # Fibroblast S2
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
    "Pericyte": [14, 15],
    "Endothelial lymphatic": [23, 24, 25],
    "Endothelial venous": [17, 19, 21],
    "Endothelial arterial": [18, 26],
    "Fibroblastic reticular cell": [20],
    "Fibroblast S3": [11, 12, 13, 16, 22, 27, 29, 31, 32],
    "Plasma IgA": [1, 2],
    "Tumor Goblet-like": [8, 10],
    "Tumor Crypt-like": [3, 4, 5],
    "Tumor Colonocyte-like": [0, 6, 9],
    "unknown": [7, 28, 30],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}

adata_stromal.obs["cell_type_fine"] = (
    adata_stromal.obs["leiden"].map(cluster_annot).fillna(adata_stromal.obs["leiden"])
)

# %%
adata_stromal.obs["cell_type_fine"] = adata_stromal.obs["cell_type_fine"].astype(str)

adata_stromal.obs.loc[
    (adata_stromal.obs["cell_type_seed"] != "unknown")
    & ~(adata_stromal.obs["leiden"].isin(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "28", "30"])),
    "cell_type_fine",
] = adata_stromal.obs["cell_type_seed"]

# Score Epithelial cells
adata_stromal.obs["cell_type_seed2"] = ah.pp.score_seeds(
    adata_stromal,
    { 
    "Tumor Crypt-like": {
        "positive": [["LGR5"]],
        "negative": [],
    },
    "Tumor Goblet-like": {
        "positive": [["MUC2", "FCGBP", "ZG16", "ATOH1"]],
        "negative": [["FABP1", "CEACAM7", "SI"]],
    },
    "Tumor Colonocyte-like": {
        "positive": [["FABP1", "CEACAM7", "SI"]],
        "negative": [],
    },
    "Plasma IgG": {
        "positive": [["IGHG1", "IGHG2", "IGHG3", "IGHG4"]],
        "negative": [["EPCAM"]]
    },
     "Plasma IgA": {
        "positive": [["IGHA1", "IGHA2"]],
        "negative": [["EPCAM"]]
    },
    },
    pos_cutoff=0.1,
    neg_cutoff=0,
)

adata_stromal.obs.loc[
    (adata_stromal.obs["cell_type_seed2"] != "unknown")
    # exclude plasmablast/b cell cluster
    & (adata_stromal.obs["leiden"].isin(["0", "1", "2", "3", "4", "5", "6", "8", "9", "10"])),
    "cell_type_fine",
] = adata_stromal.obs["cell_type_seed2"]

adata_stromal.obs.loc[
    (adata_stromal.obs["cell_type_fine"] != "Plasma IgA")
    & (adata_stromal.obs["leiden"].isin(["0", "1", "2", "3", "4", "5", "6", "8", "9", "10"]))
    & (adata_stromal.obs["medical_condition"]=="healthy"),
    "cell_type_fine",
] = "Epithelial reticular cell"

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
adata_stromal = adata_stromal[~adata_stromal.obs["cell_type_coarse"].isin(["Plasma IgA", "Plasma IgG", "empty droplet"])].copy()
ah.pp.reprocess_adata_subset_scvi(adata_stromal, leiden_res=2, n_neighbors=15, use_rep="X_scVI")

# %%
adata_stromal.write(f"{artifact_dir}/lymph_node-stromal-adata.h5ad", compression="lzf")

# %%
adata_stromal.obs["cell_type_fine"] = pd.Categorical(
    adata_stromal.obs["cell_type_fine"],
    categories=["Endothelial venous", "Endothelial arterial", "Endothelial lymphatic", "Pericyte",
                "Fibroblast S1", "Fibroblast S2", "Fibroblast S3", "Fibroblastic reticular cell", "Epithelial reticular cell",
                "Tumor Crypt-like", "Tumor Colonocyte-like", "Tumor Goblet-like", "unknown"],
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
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/lymph_node-umap_stromal_seeds.png", bbox_inches="tight")

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
        ("Pan-Cancer cell", ["EPCAM", "ELF3", "KRT8", "KRT18", "KRT19"]),
        ("Tumor Colonocyte-like", ["FABP1", "CEACAM7", "ANPEP", "SI"]),
        ("Tumor Goblet-like", ["MUC2", "FCGBP", "ZG16", "ATOH1"]),
        ("Tumor Crypt-like", ["LGR5", "SMOC2"]),
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
            
    plt.savefig(f"{artifact_dir}/lymph_node-dotplot_stromal_seeds.pdf", bbox_inches="tight")

# %%
adata_stromal.obs["cell_type_fine"].value_counts()

# %% [markdown]
# ### Seperate look at plate-based protocols

# %%
adata_plate.obs["cell_type_fine"] = (
    adata_plate.obs_names.map(adata.obs["cell_type_fine"]).fillna("unknown")
)

adata_plate.write(f"{artifact_dir}/lymph_node-plate-adata.h5ad", compression="lzf")

# %% [markdown]
# ## 2. Export lymph_node seed cell types

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    fig = sc.pl.umap(
        adata,
        color="cell_type_fine",
        legend_fontoutline=1,
        legend_fontsize="xx-small",
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/lymph_node-umap_cell_type_fine.png", bbox_inches="tight")

# %%
adata[adata.obs["cell_type_fine"] != "empty droplet"].shape

# %%
adata.obs["cell_type_fine"].value_counts(dropna=False)

# %%
# Save lymph_node seed annotation
pd.DataFrame(adata.obs[["cell_type_fine"]]).reset_index(names=["obs_names"]).to_csv(
    f"{artifact_dir}/lymph_node-cell_type_fine.csv", index=False
)
