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
# # Compute lymph_node highly variable genes

# %%
# ensure reproducibility -> set numba multithreaded mode
import os

import numpy as np
import pandas as pd
import scanpy as sc
from nxfvars import nxfvars
from threadpoolctl import threadpool_limits

# %% tags=["parameters"]
artifact_dir = nxfvars.get("artifact_dir", "/data/scratch/marteau/downloads/crc-atlas/")
adata_path = nxfvars.get(
    "adata_path",
    "../../results/v1/artifacts/build_atlas/integrate_datasets/get_seeds/artifacts/lymph_node-adata.h5ad",
)
cpus = nxfvars.get("cpus", 6)

# %%
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

# %%
adata_bd = sc.read_h5ad(adata_path)

# %%
# For now using dataset, might switch to platform
adata_bd.obs["gene_dispersion_label"] = adata_bd.obs["dataset"].astype(str)

# %%
# Min value_count per label should be 3
adata_bd.obs["gene_dispersion_label"].value_counts()

# %%
adata_hvg = adata_bd[~adata_bd.obs["sample_type"].isin(["Undetermined", "Undetermined_blood"])].copy()

# %%
sc.pp.filter_genes(adata_hvg, min_cells=40)
sc.pp.filter_genes(adata_hvg, min_counts=40)

sc.pp.filter_cells(adata_hvg, min_counts=400)
sc.pp.filter_cells(adata_hvg, min_genes=200)

# %%
hvg = sc.pp.highly_variable_genes(
    adata_hvg[
        ~adata_hvg.obs["sample_id"].isin(
            [
             'UZH_Zurich_CD45Pos.Experiment3_HB',
             'UZH_Zurich_CD45Pos.Experiment4_B',
             'UZH_Zurich_CD45Pos.Experiment4_HB',
             'UZH_Zurich_CD45Pos.Experiment5_B',
            ]
        )
    ],
    n_top_genes=4000,
    subset=False,
    flavor="seurat_v3",
    layer="counts",
    batch_key="batch",
    span=0.3,  # does not work with default span=0.3 for 4 above samples
    inplace=False,
)
# %%
# Save tumor highly variable genes
hvg.to_csv(f"{artifact_dir}/bd-hvg.csv")

# %%
hvg_dict = hvg["highly_variable"].to_dict()
adata_bd.var["highly_variable"] = adata_bd.var_names.map(hvg_dict).fillna(
    value=False
)

# %%
# For scVi integration remove empty cells -> otherwise might fail inference.Data
sc.pp.filter_genes(adata_bd, min_cells=1)
sc.pp.filter_genes(adata_bd, min_counts=1)

sc.pp.filter_cells(adata_bd, min_counts=10)
sc.pp.filter_cells(adata_bd, min_genes=10)

# %%
adata_bd.write_h5ad(f"{artifact_dir}/bd-adata.h5ad", compression="lzf")
