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
# # Compute tumor highly variable genes

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
    "../../results/v1/artifacts/build_atlas/integrate_datasets/get_seeds/artifacts/tumor-adata.h5ad",
)
cpus = nxfvars.get("cpus", 6)

# %%
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

# %%
adata_tumor = sc.read_h5ad(adata_path)

# %% [markdown]
# ### Set gene dispersion key (scVI)

# %%
# For now using dataset, might switch to platform
adata_tumor.obs["gene_dispersion_label"] = adata_tumor.obs["dataset"].astype(str)

# %%
# Min value_count per label should be 3
adata_tumor.obs["gene_dispersion_label"].value_counts()

# %% [markdown]
# ### Set batch key
# For samples with less than 10 cells, group them with similar samples in batch key.

# %%
adata_tumor.obs["batch"] = adata_tumor.obs["batch"].astype(str)

adata_tumor.obs["batch"] = adata_tumor.obs["batch"].replace(
    "Zhang_2020_10X_CD45Pos.T_P0305", "Zhang_2020_10X_CD45Pos.T_P0202"
)
replace_dict = {
    "Li_2017.CRC03_tumor": "Li_2017",
    "Li_2017.CRC06_tumor": "Li_2017",
    "Li_2017.CRC04_tumor": "Li_2017",
    "Li_2017.CRC08_tumor": "Li_2017",
    "Li_2017.CRC11_tumor": "Li_2017",
    "Li_2017.CRC02_tumor": "Li_2017",
    "Li_2017.CRC01_tumor": "Li_2017",
    "Li_2017.CRC05_tumor": "Li_2017",
    "Li_2017.CRC10_tumor": "Li_2017",
    "Li_2017.CRC07_tumor": "Li_2017",
    "Li_2017.CRC09_tumor": "Li_2017",
}

adata_tumor.obs["batch"] = adata_tumor.obs["batch"].replace(replace_dict)

# %%
adata_tumor.obs["batch"].value_counts()

# %%
# Save tumor batch keys
pd.DataFrame(adata_tumor.obs["batch"]).reset_index(names=["obs_names"]).to_csv(
    f"{artifact_dir}/tumor-batch_key.csv", index=False
)

# %%
adata_hvg = adata_tumor[
    adata_tumor.obs["sample_type"] != "Undetermined",
    adata_tumor.var["Dataset_25pct_Overlap"],
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
                "Joanito_2022_SG1.MUX9380",
                "Khaliq_2022.T_CAC5",
                "Pelka_2021_10Xv2.C138_T_0_0_0_c2_v2",
                "Pelka_2021_10Xv2_CD45Pos.C138_T_0_0_1_c1_v2",
                "Pelka_2021_10Xv2_CD45Pos.C138_T_0_0_1_c2_v2",
                "VUMC_HTAN_cohort3.HTA11_3997_2000001011",
                "VUMC_HTAN_cohort3.HTA11_3997_2000001021",
                "VUMC_HTAN_cohort3.HTA11_10888_2000001011",
                "VUMC_HTAN_cohort3.HTA11_9827_2000001011",
                "VUMC_HTAN_cohort3.HTA11_12480_2000001031",
                "VUMC_HTAN_cohort3.HTA11_6862_2000001011",
                "VUMC_HTAN_cohort3.HTA11_10573_2000001011",
                "VUMC_HTAN_cohort3.HTA11_3372_3003781011",
                "VUMC_HTAN_cohort3.HTA11_11275_2000001011",
                "VUMC_HTAN_cohort3.HTA11_9531_2000001021",
                "VUMC_HTAN_cohort3.HTA11_13794_2000001011",
                "VUMC_HTAN_discovery.HTA11_104_2000001011",
                "VUMC_HTAN_discovery.HTA11_2992_2000001011",
                "VUMC_HTAN_discovery.HTA11_5212_2000001011",
                "VUMC_HTAN_discovery.HTA11_5216_2000001011",
                "VUMC_HTAN_discovery.HTA11_5363_2000001011",
                "VUMC_HTAN_discovery.HTA11_6147_2000001011",
                "VUMC_HTAN_discovery.HTA11_6025_2000001011",
                "VUMC_HTAN_validation.HTA11_9408_2000001011",
                "VUMC_HTAN_validation.HTA11_11156_2000001021",
                "VUMC_HTAN_validation.HTA11_7128_2000001011",
                "Wu_2022_CD45Pos.P1_Colon_T",
                "Wu_2022_CD45Pos.P15_Colon_T",
                "Wu_2022_CD45Pos.P20_Colon_T",
                "Wu_2024_CD66bPos.COAD2_T",
                "Wu_2024_CD66bPos.COAD3_T",
                "Wu_2024_CD66bPos.COAD14_T",
                "Wu_2024_CD66bPos.COAD15_T",
                "Wu_2024_CD66bPos.COAD16_T",
                "Wu_2024_CD66bPos.COAD19_T",
                "Wu_2024_CD66bPos.COAD20_T",
                "Zhang_2020_10X_CD45Pos.T_P0123",
                "deVries_2023_LUMC.HTO1",
                "deVries_2023_LUMC.HTO6",
            ]
        )
    ],
    n_top_genes=4000,
    subset=False,
    flavor="seurat_v3",
    layer="counts",
    batch_key="batch",
    span=0.3,  # does not work with default span=0.3 for above samples (probably not enough cells)
    inplace=False,
)

# %%
# Save tumor highly variable genes
hvg.to_csv(f"{artifact_dir}/tumor-hvg.csv")

# %%
hvg_dict = hvg["highly_variable"].to_dict()
adata_tumor.var["highly_variable"] = adata_tumor.var_names.map(hvg_dict).fillna(
    value=False
)

# %%
# For scVi integration remove empty cells -> otherwise might fail inference.Data
sc.pp.filter_genes(adata_tumor, min_cells=1)
sc.pp.filter_genes(adata_tumor, min_counts=1)

sc.pp.filter_cells(adata_tumor, min_counts=10)
sc.pp.filter_cells(adata_tumor, min_genes=10)

# %%
adata_tumor.write_h5ad(f"{artifact_dir}/tumor-adata.h5ad", compression="lzf")
