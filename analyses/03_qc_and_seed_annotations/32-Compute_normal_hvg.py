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
# # Compute normal highly variable genes

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
    "../../results/v1/artifacts/build_atlas/integrate_datasets/get_seeds/artifacts/normal-adata.h5ad",
)
cpus = nxfvars.get("cpus", 6)

# %%
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

# %%
adata_normal = sc.read_h5ad(adata_path)

# %% [markdown]
# ### Set gene dispersion key (scVI)

# %%
# For now using dataset, might switch to platform
adata_normal.obs["gene_dispersion_label"] = adata_normal.obs["dataset"].astype(str)

# %%
# Min value_count per label should be 3
adata_normal.obs["gene_dispersion_label"] = adata_normal.obs["gene_dispersion_label"].replace(
    "Bian_2018_Tang_protocol", "Bian_2018_Dong_protocol"
)

# %%
adata_normal.obs["gene_dispersion_label"].value_counts()

# %% [markdown]
# ### Set batch key

# %%
# For samples with less than 10 cells, group them with similar samples in batch key.
adata_normal.obs["batch"] = adata_normal.obs["batch"].astype(str)

adata_normal.obs["batch"] = adata_normal.obs["batch"].replace(
    "VUMC_HTAN_discovery.HTA11_6310_2000002011",
    "VUMC_HTAN_discovery.HTA11_5363_2000002021",
)
adata_normal.obs["batch"] = adata_normal.obs["batch"].replace(
    "VUMC_HTAN_discovery.HTA11_5363_2000002021",
    "VUMC_HTAN_discovery.HTA11_6025_2000002011",
)
adata_normal.obs["batch"] = adata_normal.obs["batch"].replace(
    "Li_2017.CRC03_normal", "Li_2017.CRC06_normal"
)
replace_dict = {
    "Bian_2018_Dong_protocol.CRC03_NC": "Bian_2018",
    "Bian_2018_Dong_protocol.CRC09_NC": "Bian_2018",
    "Bian_2018_Dong_protocol.CRC04_NC": "Bian_2018",
    "Bian_2018_Tang_protocol.CRC01_NC": "Bian_2018",
    "Bian_2018_Dong_protocol.CRC06_NC": "Bian_2018",
}
adata_normal.obs["batch"] = adata_normal.obs["batch"].replace(replace_dict)


# %%
adata_normal.obs["batch"].value_counts()

# %%
# Save normal batch keys
pd.DataFrame(adata_normal.obs["batch"]).reset_index(names=["obs_names"]).to_csv(
    f"{artifact_dir}/normal-batch_key.csv", index=False
)

# %%
adata_hvg = adata_normal[:, adata_normal.var["Dataset_25pct_Overlap"]].copy()

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
                "Bian_2018_Dong_protocol.CRC03_NC",
                "Bian_2018_Dong_protocol.CRC04_NC",
                "Bian_2018_Dong_protocol.CRC06_NC",
                "Bian_2018_Dong_protocol.CRC09_NC",
                "Bian_2018_Tang_protocol.CRC01_NC",
                "Conde_2022.Pan_T7980366",
                "Elmentaite_2021_10x3p_CD45Neg.WTDAtest7844027",
                "Elmentaite_2021_10x3p_CD45Pos.WTDAtest7844026",
                "Elmentaite_2021_10x5p.ERS12382583",
                "Elmentaite_2021_10x5p.ERS12382584",
                "Elmentaite_2021_10x5p.ERS12382606",
                "Han_2020.adult_ascending_colon1",
                "Han_2020.adult_rectum1",
                "Han_2020.adult_sigmoid_colon1",
                "James_2020_10x3p_CD45Pos_CD3Pos_CD4Pos.Human_colon_16S7255675",
                "James_2020_10x3p_CD45Pos_CD3Pos_CD4Pos.Human_colon_16S7255679",
                "Joanito_2022_SG1.XHC086",
                "Kong_2023.H106265_N",
                "Li_2017.CRC03_normal",
                "Li_2017.CRC08_normal",
                "Pelka_2021_10Xv2.C119_N_0_2_0_c1_v2",
                "Pelka_2021_10Xv2_CD45Pos.C106_N_1_1_3_c1_v2",
                "Smillie_2019_10Xv1.N11_LP_A",
                "Smillie_2019_10Xv1.N13_Epi_A",
                "Smillie_2019_10Xv1.N13_Epi_B",
                "Smillie_2019_10Xv1.N20_LP_A",
                "VUMC_HTAN_cohort3.HTA11_9829_2000002011",
                "VUMC_HTAN_cohort3.HTA11_10466_2000002011",
                "VUMC_HTAN_cohort3.HTA11_3997_2000001031",
                "VUMC_HTAN_discovery.HTA11_5363_2000002021",
                "VUMC_HTAN_discovery.HTA11_5212_2000002011",
                "VUMC_HTAN_discovery.HTA11_5216_2000002011",
                "VUMC_HTAN_discovery.HTA11_1938_2000002011",
                "VUMC_HTAN_discovery.HTA11_6134_2000002021",
                "VUMC_HTAN_discovery.HTA11_6310_2000002011",
                "VUMC_HTAN_discovery.HTA11_6025_2000002011",
                "VUMC_HTAN_validation.HTA11_6182_2000002021",
                "Wu_2022_CD45Pos.P15_Colon_P",
                "Wu_2022_CD45Pos.P20_Colon_P",
                "Wu_2024_CD66bPos.COAD15_P",
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
# Save normal highly variable genes
hvg.to_csv(f"{artifact_dir}/normal-hvg.csv")

# %%
hvg_dict = hvg["highly_variable"].to_dict()
adata_normal.var["highly_variable"] = adata_normal.var_names.map(hvg_dict).fillna(
    value=False
)

# %%
# For scVi integration remove empty cells -> otherwise might fail inference.Data
sc.pp.filter_genes(adata_normal, min_cells=1)
sc.pp.filter_genes(adata_normal, min_counts=1)

sc.pp.filter_cells(adata_normal, min_counts=10)
sc.pp.filter_cells(adata_normal, min_genes=10)

# %%
adata_normal.write_h5ad(f"{artifact_dir}/normal-adata.h5ad", compression="lzf")
