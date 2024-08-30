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
#     display_name: Python [conda env:crc-atlas]
#     language: python
#     name: conda-env-crc-atlas-py
# ---

# %% [markdown]
# # Dataloader: Thomas_2024_Nat_Med

# %%
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import sc_atlas_helpers as ah
import scanpy as sc
import yaml
from nxfvars import nxfvars
from scipy.sparse import csr_matrix
from tqdm.contrib.concurrent import process_map

# %%
artifact_dir = nxfvars.get("artifact_dir", "/data/scratch/marteau/downloads/crc-atlas/")
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")
cpus = nxfvars.get("cpus", 8)

# %%
d10_1038_s41591_024_02895_x = f"{databases_path}/d10_1038_s41591_024_02895_x"
datasets_path = f"{d10_1038_s41591_024_02895_x}/GSE206301/GSE206301_RAW/channels"

# %%
# Read the YAML file and load it into a dictionary
with open(reference_meta, "r") as f:
    ref_meta_dict = yaml.load(f, Loader=yaml.Loader)

# %% [markdown]
# ## 1. Load raw adata counts

# %%
files = [x.stem for x in Path(datasets_path).glob("*")]
files.sort()

# %%
def load_counts(file):
    adata = sc.read_10x_h5(
        f"{d10_1038_s41591_024_02895_x}/GSE206301/GSE206301_RAW/channels/{file}/raw_feature_bc_matrix.h5"
    )
    adata.X = csr_matrix(adata.X)
    adata.obs["file"] = file
    
    adata.var["symbol"] = adata.var_names
    adata.var_names = adata.var["gene_ids"]
    adata.var_names.name = None
    return adata

# %%
adatas = process_map(load_counts, files, max_workers=cpus)
adata = anndata.concat(adatas, index_unique="-", join="outer", fill_value=0)

adata.var["symbol"] = adata.var_names.map(adatas[0].var["symbol"].to_dict())

# %%
adata.obs["GEO_sample_accession"] = adata.obs["file"].str.split("_").str[0]

adata.obs["sample_id"] = (
    adata.obs["file"].str.rsplit("_", n=3).str[0]
    .str.replace("_CD45sort", "")
    .str.split("_", n=1).str[1]
)

# %%
geofetch = pd.read_csv(
    f"{d10_1038_s41591_024_02895_x}/GSE206301/GSE206301_PEP/GSE206301_PEP_raw.csv"
).dropna(axis=1, how="all")

adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    geofetch,
    how="left",
    left_on=["GEO_sample_accession"],
    right_on=["sample_geo_accession"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
sample_meta = pd.read_csv(f"{d10_1038_s41591_024_02895_x}/metadata/TableS2.csv")

adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    sample_meta,
    how="left",
    left_on=["simplified_patient_id"],
    right_on=["Simple Pt ID"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
patient_meta = pd.read_csv(f"{d10_1038_s41591_024_02895_x}/metadata/TableS1.csv")

adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    patient_meta,
    how="left",
    left_on=["simplified_patient_id"],
    right_on=["Simple Pt ID"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
adata.obs = adata.obs.assign(
    study_id="Thomas_2024_Nat_Med",
    study_doi="10.1038/s41591-024-02895-x",
    study_pmid="38724705",
    tissue_processing_lab="Alexandra-Chloé Villani lab",
    dataset="Thomas_2024_Nat_Med_CD45Pos",
    medical_condition="healthy",
    treatment_status_before_resection="naive",
    sample_tissue="colon",
    sample_type="normal",
    #anatomic_region"pooled",
    enrichment_cell_types="CD45+",
    platform="10x 5' v1",
    cellranger_version="3.1.0",
    #reference_genome="",
    matrix_type="raw counts",
    hospital_location="Massachusetts General Hospital | Dana-Farber Cancer Institute, Brigham and Women’s Hospital",
    country="US",
    NCBI_BioProject_accession="PRJNA849977",
)

# %%
adata.obs["age"] = adata.obs["Patient Age"].fillna(np.nan).astype("Int64")
mapping = {
    "F": "female",
    "M": "male",
}
adata.obs["sex"] = adata.obs["Sex"].map(mapping)

adata.obs["ethnicity"] = adata.obs["Race"].map({"White": "white"})

# %%
mapping = {
    "Fresh": "fresh",
    "Frozen": "frozen",
}
adata.obs["tissue_cell_state"] = adata.obs["processed_from_fresh_or_frozen_blood"].map(mapping)

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["GEO_sample_accession"]
    + "-"
    + adata.obs_names.str.split("-").str[0]
)

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
# Save unfiltered raw counts
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
