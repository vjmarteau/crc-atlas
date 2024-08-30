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
# # Compute metastasis highly variable genes

# %%
# ensure reproducibility -> set numba multithreaded mode
import os

import numpy as np
import pandas as pd
import scanpy as sc
from nxfvars import nxfvars
from threadpoolctl import threadpool_limits

# %%
artifact_dir = nxfvars.get("artifact_dir", "/data/scratch/marteau/downloads/crc-atlas/")
adata_path = nxfvars.get(
    "adata_path",
    "../../results/v1/artifacts/build_atlas/integrate_datasets/get_seeds/artifacts/normal-adata.h5ad",
)
cpus = nxfvars.get("cpus", 6)

# %%
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

# %%
adata_metastasis = sc.read_h5ad(adata_path)

# %% [markdown]
# ### Set gene dispersion key (scVI)

# %%
# For now using dataset, might switch to platform
adata_metastasis.obs["gene_dispersion_label"] = adata_metastasis.obs["dataset"].astype(str)

# %%
# Min value_count per label should be 3
adata_metastasis.obs["gene_dispersion_label"].value_counts()

# %% [markdown]
# ### Set batch key

# %%
# For samples with less than 10 cells, group them with similar samples in batch key.
adata_metastasis.obs["batch"] = adata_metastasis.obs["batch"].astype(str)

adata_metastasis.obs["batch"] = adata_metastasis.obs["batch"].replace(
    "Wu_2022_CD45Pos.P5_Liver_T", "Wu_2022_CD45Pos.P4_Liver_T"
)
replace_dict = {
    "Tian_2023.P11D15a": "Tian_2023",
    "Tian_2023.P11D15b": "Tian_2023",
    "Tian_2023.P23D15a": "Tian_2023",
    "Tian_2023.P14D15": "Tian_2023",
    "Tian_2023.P36D15": "Tian_2023",
    "Tian_2023.P26D15a": "Tian_2023",
    "Tian_2023.P25D15a": "Tian_2023",
    "Tian_2023.P26D15b": "Tian_2023",
    "Tian_2023.P25D15b": "Tian_2023",
    "Tian_2023.P23D15b": "Tian_2023",
    "Tian_2023.P17D01a": "Tian_2023",
}

adata_metastasis.obs["batch"] = adata_metastasis.obs["batch"].replace(replace_dict)

# %%
adata_metastasis.obs["batch"].value_counts()

# %%
# Save tumor batch keys
pd.DataFrame(adata_metastasis.obs["batch"]).reset_index(names=["obs_names"]).to_csv(
    f"{artifact_dir}/metastasis-batch_key.csv", index=False
)

# %%
adata_hvg = adata_metastasis[:, adata_metastasis.var["Dataset_25pct_Overlap"]].copy()

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
                "Wu_2022_CD45Pos.P4_Liver_T",
                "Wu_2022_CD45Pos.P5_Liver_T",
                'Sathe_2023.6593_mCRC',
                'Sathe_2023.6648_mCRC',
                'Sathe_2023.8640_mCRC',
                "Giguelay_2022_CAF.19G00081",
                "Giguelay_2022_CAF.19G00619",
                'Tian_2023.P11D15a',
                'Tian_2023.P11D15b',
                'Tian_2023.P14D15',
                'Tian_2023.P17D01a',
                'Tian_2023.P23D15a',
                'Tian_2023.P23D15b',
                'Tian_2023.P25D15a',
                'Tian_2023.P25D15b',
                'Tian_2023.P26D15a',
                'Tian_2023.P26D15b',
                'Tian_2023.P29D15a',
                'Tian_2023.P29D15b',
                'Tian_2023.P36D15',
            ]
        )
    ],
    n_top_genes=2000,
    subset=False,
    flavor="seurat_v3",
    layer="counts",
    batch_key="batch",
    span=0.3, # does not work with default span=0.3 for above samples (probably not enough cells)
    inplace=False,
)

# %%
# Save metastasis highly variable genes
hvg.to_csv(f"{artifact_dir}/metastasis-hvg.csv")

# %%
hvg_dict = hvg["highly_variable"].to_dict()
adata_metastasis.var["highly_variable"] = adata_metastasis.var_names.map(hvg_dict).fillna(
    value=False
)

# %%
# For scVi integration remove empty cells -> otherwise might fail inference.Data
sc.pp.filter_genes(adata_metastasis, min_cells=1)
sc.pp.filter_genes(adata_metastasis, min_counts=1)

sc.pp.filter_cells(adata_metastasis, min_counts=10)
sc.pp.filter_cells(adata_metastasis, min_genes=10)

# %%
adata_metastasis.write_h5ad(f"{artifact_dir}/metastasis-adata.h5ad", compression="lzf")
