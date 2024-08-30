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
# # Subset by tissue type for seeds

# %%
# ensure reproducibility -> set numba multithreaded mode
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from nxfvars import nxfvars
from threadpoolctl import threadpool_limits

# %% tags=["parameters"]
artifact_dir = nxfvars.get("artifact_dir", "/data/scratch/marteau/downloads/crc-atlas/")
adata_path = nxfvars.get(
    "adata_path",
    "/data/projects/2022/CRCA/results/v1/artifacts/build_atlas/load_datasets/harmonize_datasets/artifacts/merged-adata.h5ad",
)
cpus = nxfvars.get("cpus", 12)

# %%
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

# %%
adata = sc.read_h5ad(adata_path)

# %%
scales_counts = sc.pp.normalize_total(adata, target_sum=None, inplace=False)
adata.layers["log1p_norm"] = sc.pp.log1p(scales_counts["X"], copy=True)

# %% [markdown]
# ### Save adatas by sample type for seed annotation and qc

# %%
# tumor
adata_tumor = adata[
    adata.obs["sample_type"].isin(["tumor", "polyp"])
] # Multiplet/Undetermined?
adata_tumor.obs = adata_tumor.obs.dropna(axis=1, how="all")
adata_tumor.write_h5ad(f"{artifact_dir}/tumor-adata.h5ad", compression="lzf")

# %%
adata_tumor

# %%
# normal
adata_normal = adata[
    adata.obs["sample_type"].isin(["normal"])
]  # Multiplet/Undetermined?
adata_normal.obs = adata_normal.obs.dropna(axis=1, how="all")
adata_normal.write_h5ad(f"{artifact_dir}/normal-adata.h5ad", compression="lzf")

# %%
adata_normal

# %%
# blood
adata_blood = adata[
    adata.obs["sample_type"].isin(["blood"])
]
adata_blood.obs = adata_blood.obs.dropna(axis=1, how="all")
adata_blood.write_h5ad(f"{artifact_dir}/blood-adata.h5ad", compression="lzf")

# %%
adata_blood

# %%
# metastasis
adata_metastasis = adata[adata.obs["sample_type"].isin(["metastasis"])]
adata_metastasis.obs = adata_metastasis.obs.dropna(axis=1, how="all")
adata_metastasis.write_h5ad(f"{artifact_dir}/metastasis-adata.h5ad", compression="lzf")

# %%
adata_metastasis

# %%
# lymph_node
adata_lymph_node = adata[adata.obs["sample_type"].isin(["lymph node"])]
adata_lymph_node.obs = adata_lymph_node.obs.dropna(axis=1, how="all")
adata_lymph_node.write_h5ad(f"{artifact_dir}/lymph_node-adata.h5ad", compression="lzf")

# %%
adata_lymph_node
