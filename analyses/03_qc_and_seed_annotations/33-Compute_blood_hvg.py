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
# # Compute blood highly variable genes

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
    "../../results/v1/artifacts/build_atlas/integrate_datasets/get_seeds/artifacts/blood-adata.h5ad",
)
cpus = nxfvars.get("cpus", 6)

# %%
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

# %%
adata_blood = sc.read_h5ad(adata_path)

# %% [markdown]
# ### Set gene dispersion key (scVI)

# %%
# For now using dataset, might switch to platform
adata_blood.obs["gene_dispersion_label"] = adata_blood.obs["dataset"].astype(str)

# %%
# Min value_count per label should be 3
adata_blood.obs["gene_dispersion_label"].value_counts()

# %% [markdown]
# ### Set batch key

# %%
adata_blood.obs["batch"].value_counts()

# %%
# Save tumor batch keys
pd.DataFrame(adata_blood.obs["batch"]).reset_index(names=["obs_names"]).to_csv(
    f"{artifact_dir}/blood-batch_key.csv", index=False
)

# %%
adata_hvg = adata_blood[
    adata_blood.obs["sample_type"] != "Undetermined_blood",
    adata_blood.var["Dataset_25pct_Overlap"],
].copy()

# %%
adata_hvg.shape

# %%
sc.pp.filter_genes(adata_hvg, min_cells=40)
sc.pp.filter_genes(adata_hvg, min_counts=40)

sc.pp.filter_cells(adata_hvg, min_counts=400)
sc.pp.filter_cells(adata_hvg, min_genes=400)

# %%
hvg = sc.pp.highly_variable_genes(
    adata_hvg[
        ~adata_hvg.obs["sample_id"].isin(
            [
                "UZH_Zurich_CD45Pos.Experiment4_HB",
                "UZH_Zurich_CD45Pos.Experiment5_B",
                "Wu_2024_CD66bPos.COAD16_B",
                "Zhang_2020_10X_CD45Pos.P_P0323",
            ]
        )
    ],
    n_top_genes=2000,
    subset=False,
    flavor="seurat_v3",
    layer="counts",
    batch_key="batch",
    span=0.3,  # does not work with default span=0.3 for 4 samples (above)
    inplace=False,
)

# %%
# Save blood highly variable genes
hvg.to_csv(f"{artifact_dir}/blood-hvg.csv")

# %%
hvg_dict = hvg["highly_variable"].to_dict()
adata_blood.var["highly_variable"] = adata_blood.var_names.map(hvg_dict).fillna(
    value=False
)

# %%
# For scVi integration remove empty cells -> otherwise might fail inference.Data
sc.pp.filter_genes(adata_blood, min_cells=1)
sc.pp.filter_genes(adata_blood, min_counts=1)

sc.pp.filter_cells(adata_blood, min_counts=10)
sc.pp.filter_cells(adata_blood, min_genes=10)

# %%
adata_blood.write_h5ad(f"{artifact_dir}/blood-adata.h5ad", compression="lzf")
