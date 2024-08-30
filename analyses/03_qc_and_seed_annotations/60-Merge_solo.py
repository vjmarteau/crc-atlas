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
# # Merge SOLO doublet probabilities

# %%
# ensure reproducibility -> set numba multithreaded mode
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sc_atlas_helpers as ah
import scanpy as sc
from nxfvars import nxfvars
from threadpoolctl import threadpool_limits

# %% tags=["parameters"]
artifact_dir = nxfvars.get("artifact_dir", "/data/scratch/marteau/downloads/crc-atlas/")
adata_path = nxfvars.get(
    "adata_path",
    "../../results/v1/artifacts/build_atlas/integrate_datasets/get_seeds/artifacts/merged-adata.h5ad",
)
solo_dir = nxfvars.get(
    "solo_dir",
    "../../results/v1/artifacts/build_atlas/integrate_datasets/solo_doublets/solo"
)
cpus = nxfvars.get("cpus", 2)

os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

sc.settings.set_figure_params(
    dpi=200,
    facecolor="white",
    frameon=False,
)

# %% [markdown]
# ## add SOLO doublet information to anndata

# %%
adata = sc.read_h5ad(adata_path)

# %%
# Append SOLO output
for col in ["SOLO_doublet_prob", "SOLO_singlet_prob", "SOLO_is_doublet"]:

    solo_res = {}
    for f in Path(solo_dir).glob("*-solo_doublet_status.csv"):
        dataset = f.name.replace("-solo_doublet_status.csv", "")
        solo_res[dataset] = pd.read_csv(f).rename(columns={"Unnamed: 0": "obs_names"})
    
    solo_all = pd.concat([df.assign(dataset=dataset) for dataset, df in solo_res.items()])
    
    solo_dict = solo_all.set_index("obs_names")[col].to_dict()
    adata.obs[col] = adata.obs_names.map(solo_dict)

adata.obs.rename({'SOLO_is_doublet': 'SOLO_doublet_status'}, axis=1, inplace=True)

# %% [markdown]
# ## doublet statistics
NA = datasets that have been excluded from doublet detection (i.e plate-based such as Smartseq2 datasets)

# %%
adata.obs["SOLO_doublet_status"].value_counts(dropna=False)

# %%
# Set plate-based datasets to "singlet"
adata.obs["SOLO_doublet_status"] = adata.obs["SOLO_doublet_status"].fillna("singlet")

# %%
# Set display options to show all rows and columns
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# %%
adata[
    adata.obs["platform"].isin(
        [
            '10x',
            "10x 3' v2",
            "10x 3' v3",
            "10X 5'",
            "10x 5' v1",
            "10x 5' v2",
            "10x 5' v3",
            "GEXSCOPE Singleron",
            'TruDrop',
        ]
    )
].obs.groupby("sample_id", observed=True)[["SOLO_doublet_status"]].apply(
    lambda df: pd.DataFrame().assign(
        doublet_frac=[np.sum(df["SOLO_doublet_status"] == "doublet") / df.shape[0]],
        n=[df.shape[0]],
    )
).sort_values("doublet_frac")

# %%
pd.reset_option("display.max_rows")
pd.reset_option("display.max_columns")

# %%
# I will only remove doublets from the datasets below that were denoised using scAR, no need for BD datasets -> have a column "Multiplet"
# Set all values to singelt except for datasets below!
adata.obs.loc[
    ~adata.obs["dataset"].isin(
        [
            "Che_2021",
            "Conde_2022",
            "Elmentaite_2021_10x5p",
            "Elmentaite_2021_10x5p_CD45Neg",
            "Elmentaite_2021_10x5p_CD45Pos",
            "GarridoTrigo_2023",
            "Guo_2022",
            "He_2020",
            "James_2020_10x3p_CD45Pos_CD3Neg_CD4Neg",
            "James_2020_10x3p_CD45Pos_CD3Pos_CD4Pos",
            "James_2020_10x5p",
            "James_2020_10x5p_CD45Pos",
            "Lee_2020_KUL3",
            "Wang_2020",
            "Yang_2023",
            "HTAPP_HTAN",
            "Ji_2024_scopeV2",
            "Ji_2024_scopeV1",
            #'MUI_Innsbruck',
            #'UZH_Zurich_CD45Pos',
            #'UZH_Zurich_healthy_blood',
            #'MUI_Innsbruck_AbSeq',
            "Sathe_2023",
            "WUSTL_HTAN",
            "Thomas_2024_Nat_Med_CD45Pos",
        ]
    ),
    "SOLO_doublet_status",
] = "singlet"

# %%
scales_counts = sc.pp.normalize_total(adata, target_sum=None, inplace=False)
adata.layers["log1p_norm"] = sc.pp.log1p(scales_counts["X"], copy=True)

# %%
# Clean up CD8 vs NK cell annotation for seeds
adata.obs["tmp_CD8"] = ah.pp.score_seeds(
    adata,
    {
        "CD8": {
            "positive": [["CD3E", "TRAC", "TRBC1", "TRBC2"]],
            "negative": [[]],
        },
     },
    pos_cutoff=0.1,
)

# %%
adata.obs["cell_type_fine"] = adata.obs["cell_type_fine"].astype(str)
adata.obs.loc[
    (adata.obs["tmp_CD8"] != "unknown")
& (adata.obs["cell_type_fine"].isin(["NK"])),
    "cell_type_fine"
] = "CD8"

# %%
# Clean up CD8 vs NK cell annotation for seeds
adata.obs["tmp_gd"] = ah.pp.score_seeds(
    adata,
    {
        "gamma-delta": {
            "positive": [["TRDC", "TRDV1", "TRGC1", "TRGC2"]],
            "negative": [[]],
        },
     },
    pos_cutoff=0.1,
)

# %%
adata.obs.loc[
    (adata.obs["tmp_gd"] != "unknown")
& (adata.obs["cell_type_fine"].isin(["ILC"])),
    "cell_type_fine"
] = "gamma-delta"

adata.obs.loc[
    (adata.obs["tmp_gd"] != "unknown")
& (adata.obs["cell_type_fine"].isin(["NK"])),
    "cell_type_fine"
] = "gamma-delta"

adata.obs.loc[
    (adata.obs["tmp_CD8"] != "unknown")
& (adata.obs["cell_type_fine"].isin(["ILC"])),
    "cell_type_fine"
] = "CD4"

# %%
adata.obs["tmp_ab"] = ah.pp.score_seeds(
    adata,
    {
        "CD8": {
            "positive": [["TRBC1", "TRBC2", "CD8A", "CD8B"]],
            "negative": [[]],
        },
     },
    pos_cutoff=0.1,
)

adata.obs.loc[
    (adata.obs["tmp_ab"] != "unknown")
& (adata.obs["cell_type_fine"].isin(["gamma-delta"])),
    "cell_type_fine"
] = "CD8"

# %%
cluster_annot = {
    "Neutrophil": [47, 48],
    "Eosinophil": [46],
    "Mast cell": [57],
    "Granulocyte progenitor": [38],
    "CD8": [62],
    "B cell activated": [23],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}

adata.obs["tmp_cell_type_fine"] = (
    adata.obs["leiden"].map(cluster_annot).fillna("unknown")
)

# %%
adata.obs["cell_type_fine"] = adata.obs["cell_type_fine"].astype(str)
adata.obs.loc[
    (adata.obs["tmp_cell_type_fine"] != "unknown")
& (adata.obs["cell_type_fine"] == "unknown"),
    "cell_type_fine"
] = adata.obs["tmp_cell_type_fine"]

# %%
# Clean up Granulocytes -> Had a look at marker gene expresssion for cells below set to "unknown"
adata.obs["cell_type_fine"] = adata.obs["cell_type_fine"].astype(str)
adata.obs.loc[
    (adata.obs["cell_type_fine"] == "Eosinophil")
& (adata.obs["platform"] != "BD Rhapsody"),
    "cell_type_fine"
] = "unknown"

adata.obs.loc[
    (adata.obs["cell_type_fine"] == "Neutrophil")
& (adata.obs["dataset"].isin(["Tian_2023", "Lee_2020_SMC_test_setup_fresh", "Zhang_2020_CD45Pos_CD45Neg", "Khaliq_2022", "Zhang_2020_10X_CD45Pos", "Sathe_2023", "Li_2023_10Xv2", "Elmentaite_2021_10x5p", "Lee_2020_SMC_test_setup_frozen", "Elmentaite_2021_10x5p_CD45Pos", "Zhang_2018_CD3Pos", "WUSTL_HTAN", "Giguelay_2022_CAF", "VUMC_HTAN_discovery", "Terekhanova_2023_HTAN", "VUMC_HTAN_cohort3"])),
    "cell_type_fine"
] = "unknown"

# %%
adata.write_h5ad(f"{artifact_dir}/merged-adata.h5ad", compression="lzf")
