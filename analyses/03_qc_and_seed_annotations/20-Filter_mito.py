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
# # Apply hard percent mito cut-off & remove not informative genes

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
phase_dir = nxfvars.get(
    "phase_dir", "../../results/v1/artifacts/build_atlas/cell_cycle_phase/score_cell_cycle"
)
cpus = nxfvars.get("cpus", 12)

# %%
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

# %%
adata = sc.read_h5ad(adata_path)

# %% [markdown]
# ## Get qc metrics

# %%
#sc.pp.filter_genes(adata, min_cells=10)
#sc.pp.filter_genes(adata, min_counts=10)

#sc.pp.filter_cells(adata, min_counts=10)
#sc.pp.filter_cells(adata, min_genes=10)

# %%
# mitochondrial genes
adata.var["mito"] = adata.var_names.str.startswith("MT-")
# ribosomal genes
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
# hemoglobin genes.
adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

sc.pp.calculate_qc_metrics(
    adata,
    qc_vars=["mito", "ribo", "hb"],
    inplace=True,
    percent_top=[20],
    log1p=True,
)

# %% [markdown]
# ## Append cell cycle scores

# %%
for col in ["S_score", "G2M_score", "phase"]:

    phase_res = {}
    for f in Path(phase_dir).glob("*-cell_cycle_score.csv"):
        dataset = f.name.replace("-cell_cycle_score.csv", "")
        phase_res[dataset] = pd.read_csv(f)
    
    phase_all = pd.concat([df.assign(dataset=dataset) for dataset, df in phase_res.items()])
    
    phase_dict = phase_all.set_index("obs_names")[col].to_dict()
    adata.obs[col] = adata.obs_names.map(phase_dict)

# %%
adata.obs["batch"] = adata.obs["sample_id"].astype(str)

# %%
def hard_mito_cut_off(row):
    # BD Rhapsody hard percent mito cut off: 60%
    if row["platform"] == "BD Rhapsody" and row["pct_counts_mito"] < 60:
        return "cell"
    # Droplet-based protocols hard percent mito cut off: 40%
    elif (
        row["platform"]
        in [
            '10x',
            "10x 3' v1",
            "10x 3' v2",
            "10x 3' v3",
            "10x 5' v1",
            "10x 5' v2",
            "10x 5' v3",
            'TruDrop',
            'GEXSCOPE Singleron',
            "DNBelab C4",
        ]
        and row["pct_counts_mito"] < 40
    ):
        return "cell"
    # Well-based protocols hard percent mito cut off: 40%
    elif (
        row["platform"]
        in [
            'Microwell-seq',
            'SMARTer (C1)',
            'Smart-seq2',
            'scTrio-seq2_Dong_protocol',
            'scTrio-seq2_Tang_protocol',
        ]
        and row["pct_counts_mito"] < 100
    ):
        return "cell"
    else:
        return "droplet"

# %%
# Apply the function to create is_droplet_mito column
adata.obs["is_droplet_mito"] = adata.obs.apply(
    lambda row: hard_mito_cut_off(row), axis=1
).fillna('cell')

# %% [markdown]
# ## Save own BD Rhapsody datasets seperately

# %%
# Get BD Rhapsody datasets
adata_bd = adata[adata.obs["dataset"].isin(["MUI_Innsbruck", "MUI_Innsbruck_AbSeq", "UZH_Zurich_CD45Pos", "UZH_Zurich_healthy_blood"])]
adata_bd = adata_bd[adata_bd.obs["sample_type"] != "Multiplet"]
adata_bd = adata_bd[adata_bd.obs["is_droplet_mito"] == "cell"]
adata_bd.obs = adata_bd.obs.dropna(axis=1, how="all")

#sc.pp.filter_genes(adata_bd, min_cells=10)
#sc.pp.filter_genes(adata_bd, min_counts=10)

adata_bd.write_h5ad(f"{artifact_dir}/bd-adata.h5ad", compression="lzf")

# %% [markdown]
# ## Remove uninformative gene classes for downstream steps

# %%
var_names = adata.var

# %%
# For annotation will remove some of the non annotation useful genes
rm_classes = [
    "lncRNA",
    "processed_pseudogene",
    "unprocessed_pseudogene",
    "misc_RNA",
    "snRNA",
    "TEC",
    "transcribed_unprocessed_pseudogene",
    "snoRNA",
    "lincRNA",
    "transcribed_processed_pseudogene",
    "rRNA_pseudogene",
    "pseudogene",
    "antisense",
    "vault_RNA",
    "scRNA",
    "IG_pseudogene",
    "sRNA",
    "scaRNA",
    "artifact",
    "translated_processed_pseudogene",
    "unitary_pseudogene",
    "nan",
]

condition_class = ~var_names["Class"].isin(rm_classes)
var_names = var_names.loc[condition_class | var_names["Dataset_25pct_Overlap"]]
var_names = var_names.loc[var_names["Version"] == "gencode.v44"]

# %%
# Apply hard percent mito cut-off & subset genes
adata = adata[
    adata.obs["is_droplet_mito"] == "cell", adata.var_names.isin(var_names.index)
].copy()

# %%
del adata.obsm["X_scAR"]
del adata.layers["denoised"]

# %%
adata

# %%
adata.write_h5ad(f"{artifact_dir}/mito_filtered-adata.h5ad", compression="lzf")
