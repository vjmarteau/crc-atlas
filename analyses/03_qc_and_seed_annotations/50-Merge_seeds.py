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
# # Merge seed annotations

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
    "../../results/v1/artifacts/build_atlas/integrate_datasets/get_seeds/artifacts/merged-adata.h5ad",
)
bd_is_droplet = nxfvars.get(
    "bd_is_droplet", "../../results/v1/artifacts/build_atlas/integrate_datasets/seed/bd_rhapsody/artifacts/bd-is_droplet.csv"
)
hvg_dir = nxfvars.get(
    "hvg_dir", "../../results/v1/artifacts/build_atlas/integrate_datasets/hvg_test"
)
cpus = nxfvars.get("cpus", 12)

# %%
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

# %% [markdown]
# ## Merge seed annotations

# %%
adata = sc.read_h5ad(adata_path)

# %%
adata.shape

# %%
# Append columns from seed annotations
for col in ["batch_key", "cell_type_fine"]:

    seed_res = {}
    for f in Path(hvg_dir).glob(f"*-{col}.csv"):
        seed = f.name.replace(f"-{col}.csv", "")
        seed_res[seed] = pd.read_csv(f).rename(columns={"batch": "batch_key"})
    
    seed_all = pd.concat([df.assign(seed=seed) for seed, df in seed_res.items()])
    
    seed_dict = seed_all.set_index("obs_names")[col].to_dict()
    adata.obs[col] = adata.obs_names.map(seed_dict).fillna("empty droplet")

# %%
# Append bd-adata is_droplet (bd-rhapsody); all non BD datasets will be nan
bd_is_droplet = pd.read_csv(bd_is_droplet)

adata.obs["bd_is_droplet"] = (
    adata.obs_names.map(bd_is_droplet.set_index("obs_names")["is_droplet"])
)

# %%
# Set BD Undetermined seed labels to "unknown"
adata.obs["cell_type_fine"] = adata.obs["cell_type_fine"].astype(str)
adata.obs.loc[
    (adata.obs["cell_type_fine"] == "empty droplet")
    & (adata.obs["platform"] == "BD Rhapsody")
    & (adata.obs["bd_is_droplet"] == "cell"),
    "cell_type_fine",
] = "unknown"

# %%
mapping = {
    "Tumor Colonocyte-like": "Colonocyte",
    "Tumor BEST4": "Colonocyte BEST4",
    "Tumor Goblet-like": "Goblet",
    "Tumor Crypt-like": "Crypt cell",
    "Tumor TA-like": "TA progenitor",
}

adata.obs.loc[
    adata.obs["sample_type"] == "polyp",
    "cell_type_fine"
] = adata.obs.loc[
    adata.obs["sample_type"] == "polyp",
    "cell_type_fine"
].map(lambda x: mapping.get(x, x))

adata.obs["cell_type_fine"] = adata.obs["cell_type_fine"].replace(
    "Epithelial cell", "unknown"
)

# %%
adata = adata[
    (adata.obs["sample_type"] != "Multiplet")
    # This platform ("Microwell-seq") is weird and does not really integrate -> remove!
    & (adata.obs["dataset"] != "Han_2020")
    # Looks like small intestine samples
    & (adata.obs["sample_id"] != "Burclaff_2022.nan")
    & (adata.obs["cell_type_fine"] != "empty droplet")
    ].copy()

# %%
adata.obs.loc[
    adata.obs["anatomic_region"].isin(["mesenteric mesenteric lymph nodess"]), "anatomic_region"
] = "mesenteric lymph nodes"

adata.obs["anatomic_region"] = adata.obs["anatomic_region"].astype(str)

# %%
cluster_annot = {
    "Monocyte": "Monocyte",
    "Macrophage": "Macrophage",
    "Macrophage cycling": "Macrophage",
    "cDC progenitor": "Dendritic cell",
    "cDC1": "Dendritic cell",
    "cDC2": "Dendritic cell",
    "DC3": "Dendritic cell",
    "pDC": "Dendritic cell",
    "DC mature": "Dendritic cell",
    "Granulocyte progenitor": "Neutrophil",
    "Neutrophil": "Neutrophil",
    "Eosinophil": "Eosinophil",
    "Mast cell": "Mast cell",
    "Erythroid cell": "Erythroid cell",
    "Platelet": "Platelet",
    "CD4": "CD4",
    "Treg": "Treg",
    "CD8": "CD8",
    "NK": "NK",
    "ILC": "ILC",
    "gamma-delta": "gamma-delta",
    "NKT": "NKT",
    "CD4 naive": "CD4",
    "CD8 naive": "CD8",
    "CD4 stem-like": "CD4",
    "CD8 stem-like": "CD8",
    "CD4 cycling": "CD4",
    "CD8 cycling": "CD8",
    "NK cycling": "NK",
    "GC B cell": "B cell",
    "B cell naive": "B cell",
    "B cell activated naive": "B cell",
    "B cell activated": "B cell",
    "B cell memory": "B cell",
    "Plasma IgA": "Plasma cell",
    "Plasma IgG": "Plasma cell",
    "Plasma IgM": "Plasma cell",
    "Plasmablast": "Plasma cell",
    "Crypt cell": "Epithelial progenitor",
    "TA progenitor": "Epithelial progenitor",
    "Colonocyte": "Epithelial cell",
    "Colonocyte BEST4": "Epithelial cell",
    "Goblet": "Goblet",
    "Tuft": "Tuft",
    "Enteroendocrine": "Enteroendocrine",
    "Tumor Colonocyte-like": "Cancer cell",
    "Tumor BEST4": "Cancer cell",
    "Tumor Goblet-like": "Cancer cell",
    "Tumor Crypt-like": "Cancer cell",
    "Tumor TA-like": "Cancer cell",
    "Cancer cell circulating": "Cancer cell circulating",
    "Endothelial venous": "Endothelial cell",
    "Endothelial arterial": "Endothelial cell",
    "Endothelial lymphatic": "Endothelial cell",
    "Fibroblast S1": "Fibroblast",
    "Fibroblast S2": "Fibroblast",
    "Fibroblast S3": "Fibroblast",
    "Pericyte": "Pericyte",
    "Schwann cell": "Schwann cell",
    "Hepatocyte": "Hepatocyte",
    "Fibroblastic reticular cell": "Fibroblast",
    "Epithelial reticular cell": "Epithelial cell",
    "unknown": "unknown",
}
adata.obs["cell_type_middle"] = (
    adata.obs["cell_type_fine"].map(cluster_annot)
)

# %%
cluster_annot = {
    "Monocyte": "Myeloid cell",
    "Macrophage": "Myeloid cell",
    "Dendritic cell": "Myeloid cell",
    "Neutrophil": "Granulocyte",
    "Eosinophil": "Granulocyte",
    "Mast cell": "Granulocyte",
    "Erythroid cell": "Myeloid cell",
    "Platelet": "Myeloid cell",
    "CD4": "T cell CD4",
    "Treg": "T cell CD4",
    "CD8": "T cell CD8",
    "NK": "NK",
    "ILC": "ILC",
    "gamma-delta": "gamma-delta",
    "NKT": "NKT",
    "B cell": "B cell",
    "Plasma cell": "Plasma cell",
    "Epithelial progenitor": "Epithelial cell",
    "Epithelial cell": "Epithelial cell",
    "Goblet": "Epithelial cell",
    "Tuft": "Epithelial cell",
    "Enteroendocrine": "Epithelial cell",
    "Cancer cell": "Cancer cell",
    "Cancer cell circulating": "Cancer cell",
    "Endothelial cell": "Stromal cell",
    "Fibroblast": "Stromal cell",
    "Pericyte": "Stromal cell",
    "Schwann cell": "Stromal cell",
    "Hepatocyte": "Hepatocyte",
    "unknown": "unknown",
}
adata.obs["cell_type_coarse"] = (
    adata.obs["cell_type_middle"].map(cluster_annot)
)

# %% [markdown]
# ### Set batch key

# %%
adata.obs["batch"] = adata.obs["batch_key"].astype(str)
del adata.obs["batch_key"]

# %%
# Min value_count per label should be 3
adata.obs["batch"].value_counts()

# %% [markdown]
# ## Merge highly variable genes from seeds

# %%
hvg_res = {}
for f in Path(hvg_dir).glob("*-hvg.csv"):
    seed = f.name.replace("-hvg.csv", "")
    hvg_res[seed] = pd.read_csv(f).rename(columns={"Unnamed: 0": "var_names"})

# %%
hvg_all = pd.concat([df.assign(seed=seed) for seed, df in hvg_res.items()])

# %%
hvg_all = hvg_all.loc[hvg_all["highly_variable"]].drop_duplicates(subset=["var_names"])

# %%
hvg_dict = hvg_all.set_index("var_names")["highly_variable"].to_dict()
adata.var["highly_variable"] = adata.var_names.map(hvg_dict).fillna(
    value=False
)

# %%
adata.var["highly_variable"].value_counts()

# %%
# Save merged seed highly variable genes
hvg_all.to_csv(f"{artifact_dir}/merged_seed-hvg.csv")

# %%
adata.var["highly_variable"] = adata.var["highly_variable"].astype(str)
adata.var['Dataset_25pct_Overlap'] = adata.var['Dataset_25pct_Overlap'].map({'True': True, 'False': False}).astype(bool)
adata.var['highly_variable'] = adata.var['highly_variable'].map({'True': True, 'False': False}).astype(bool)

adata.obs = adata.obs.dropna(axis=1, how="all")
adata.write_h5ad(f"{artifact_dir}/merged-adata.h5ad", compression="lzf")

# %%
adata
