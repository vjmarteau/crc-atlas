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
#     display_name: Python [conda env:.conda-2024-crc-atlas-scanpy]
#     language: python
#     name: conda-env-.conda-2024-crc-atlas-scanpy-py
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
adata_lymph_node = sc.read_h5ad(adata_path)

# %% [markdown]
# ### Set gene dispersion key (scVI)

# %%
# For now using dataset, might switch to platform
adata_lymph_node.obs["gene_dispersion_label"] = adata_lymph_node.obs["dataset"].astype(str)

# %%
# Min value_count per label should be 3
adata_lymph_node.obs["gene_dispersion_label"].value_counts()

# %% [markdown]
# ### Set batch key

# %%
adata_lymph_node.obs["batch"] = adata_lymph_node.obs["batch"].astype(str)

adata_lymph_node.obs["batch"] = adata_lymph_node.obs["batch"].replace(
    "Bian_2018_Dong_protocol.CRC03_LN", "Bian_2018_Dong_protocol.CRC06_LN"
)
adata_lymph_node.obs["batch"] = adata_lymph_node.obs["batch"].replace(
    "Elmentaite_2021_10x5p.ERS12382585", "Elmentaite_2021_10x5p.ERS12382616"
)

# %%
adata_lymph_node.obs["batch"].value_counts()

# %%
# Save tumor batch keys
pd.DataFrame(adata_lymph_node.obs["batch"]).reset_index(names=["obs_names"]).to_csv(
    f"{artifact_dir}/lymph_node-batch_key.csv", index=False
)

# %%
adata_hvg = adata_lymph_node[:, adata_lymph_node.var["Dataset_25pct_Overlap"]].copy()

# %%
adata_hvg.shape

# %%
sc.pp.filter_genes(adata_hvg, min_cells=40)
sc.pp.filter_genes(adata_hvg, min_counts=40)

sc.pp.filter_cells(adata_hvg, min_counts=400)
sc.pp.filter_cells(adata_hvg, min_genes=400)

# %%
hvg = sc.pp.highly_variable_genes(
    adata_hvg,
    n_top_genes=2000,
    subset=False,
    flavor="seurat_v3",
    layer="counts",
    batch_key="batch",
    span=0.3,
    inplace=False,
)

# %%
# Save lymph_node highly variable genes
hvg.to_csv(f"{artifact_dir}/lymph_node-hvg.csv")

# %%
hvg_dict = hvg["highly_variable"].to_dict()
adata_lymph_node.var["highly_variable"] = adata_lymph_node.var_names.map(hvg_dict).fillna(
    value=False
)

# %%
# For scVi integration remove empty cells -> otherwise might fail inference.Data
sc.pp.filter_genes(adata_lymph_node, min_cells=1)
sc.pp.filter_genes(adata_lymph_node, min_counts=1)

sc.pp.filter_cells(adata_lymph_node, min_counts=10)
sc.pp.filter_cells(adata_lymph_node, min_genes=10)

# %%
adata_lymph_node.write_h5ad(f"{artifact_dir}/lymph_node-adata.h5ad", compression="lzf")
