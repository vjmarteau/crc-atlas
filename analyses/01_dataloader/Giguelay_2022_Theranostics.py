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
# # Dataloader: Giguelay_2022_Theranostics

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
from threadpoolctl import threadpool_limits
from tqdm.contrib.concurrent import process_map

# %%
artifact_dir = nxfvars.get("artifact_dir", "/data/scratch/marteau/downloads/crc-atlas/")
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")
cpus = nxfvars.get("cpus", 2)

# %%
d10_7150_thno_72853 = f"{databases_path}/d10_7150_thno_72853"
datasets_path = f"{d10_7150_thno_72853}/GSE158692_RAW"

# %%
# Read the YAML file and load it into a dictionary
with open(reference_meta, "r") as f:
    ref_meta_dict = yaml.load(f, Loader=yaml.Loader)

# %% [markdown]
# ## 1. Load raw adata counts

# %%
files = [x.stem for x in Path(datasets_path).glob("*.txt.gz")]
files.sort()

# %%
def load_counts(file):
    counts = pd.read_csv(
        f"{d10_7150_thno_72853}/GSE158692_RAW/{file}.gz", delimiter="\t"
    ).T
    adata = sc.AnnData(counts)
    adata.X = csr_matrix(adata.X)
    adata.obs["file"] = file
    return adata

# %%
adatas = process_map(load_counts, files, max_workers=cpus)
adata = anndata.concat(adatas, index_unique="-", join="outer", fill_value=0)

# %%
adata.obs["type"] = adata.obs["file"].str.split("_").str[3].str.split(".").str[0]
adata.obs["File"] = adata.obs["file"].str.split("_").str[2]
adata.obs["GEO_sample_accession"] = adata.obs["file"].str.split("_").str[0]

# %%
patient_meta = pd.read_csv(f"{d10_7150_thno_72853}/metadata/TableS1.csv")

adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    patient_meta,
    how="left",
    on=["File"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
adata.obs = adata.obs.assign(
    study_id="Giguelay_2022_Theranostics",
    study_doi="10.7150/thno.72853",
    study_pmid="36438498",
    tissue_processing_lab="Jacques Colinge lab",
    dataset="Giguelay_2022",
    medical_condition="colorectal cancer",
    # Paper methods: All patients were treated with neo-adjuvant chemotherapy before surgery and sample collection
    treatment_status_before_resection="treated",
    sample_tissue="liver",
    sample_type="metastasis",
    tissue_cell_state="fresh",
    # TableS1: MSI No -> MSS
    microsatellite_status="MSS",
    platform="10x 3' v3",
    cellranger_version="3.1.0",
    # reference_genome="",
    matrix_type="raw counts",
    hospital_location="Regional Cancer Hospital ICM, Montpellier",
    country="France",
    NCBI_BioProject_accession="PRJNA666199",
    SRA_sample_accession="SRP285677",
)

# %%
adata.obs["sample_id"] = adata.obs["File"]
adata.obs["patient_id"] = adata.obs["Patient"]
adata.obs["treatment_drug"] = adata.obs["Last treatment"]

# %%
adata.obs["age"] = adata.obs["Age"].fillna(np.nan).astype("Int64")
mapping = {
    "F": "female",
    "M": "male",
}
adata.obs["sex"] = adata.obs["Gender"].map(mapping)

# %%
# define the dictionary mapping
mapping = {
    "Sigmoid": "sigmoid colon",
    "High rectum": "rectum",
    "Inferior rectum": "rectum",
    "Recto-sigmoid hinge": "rectosigmoid junction",
    "Right colon": "nan",
}
adata.obs["anatomic_location"] = adata.obs["Primary tumor location"].map(mapping)

# define the dictionary mapping
mapping = {
    "Sigmoid": "distal colon",
    "High rectum": "distal colon",
    "Inferior rectum": "distal colon",
    "Recto-sigmoid hinge": "distal colon",
    "Right colon": "proximal colon",
}
adata.obs["anatomic_region"] = adata.obs["Primary tumor location"].map(mapping)

# %%
adata.obs["KRAS_status_driver_mut"] = "wt"
adata.obs["KRAS_status_driver_mut"] = np.where(
    adata.obs["patient_id"] == "P5", "mut", adata.obs["KRAS_status_driver_mut"]
)

# %%
# Paper: We purified CAFs using a triple negative (TN) selection strategy and flow cytometry (EPCAM-/CD45-/CD31-
mapping = {
    "EPCAM": "EPCAM+",
    "TN": "EPCAM-/CD45-/CD31-",
    "TN1": "EPCAM-/CD45-/CD31-",
    "TN2": "EPCAM-/CD45-/CD31-",
    "EPCAM-CD45": "EPCAM+",
}
adata.obs["enrichment_cell_types"] = adata.obs["type"].map(mapping)

# %%
adata.obs["dataset"] = adata.obs["dataset"] + "_" + adata.obs["enrichment_cell_types"]

mapping = {
    "Giguelay_2022_EPCAM+": "Giguelay_2022_EPCAM_Pos",
    "Giguelay_2022_EPCAM-/CD45-/CD31-": "Giguelay_2022_CAF",
}
adata.obs["dataset"] = adata.obs["dataset"].map(mapping)

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]]

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["GEO_sample_accession"]
    + "-"
    + adata.obs_names.str.split("-", n=1).str[0]
)

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
adata.obs["sample_id"].value_counts()

# %% [markdown]
# ## 4. Save adata by dataset

# %%
datasets = [
    adata[adata.obs["dataset"] == dataset, :].copy()
    for dataset in adata.obs["dataset"].unique()
    if adata[adata.obs["dataset"] == dataset, :].shape[0] != 0
]

# %%
datasets

# %%
for _adata in datasets:
    _adata.obs = _adata.obs.dropna(axis=1, how="all")  # Remove all nan columns, else write_h5ad will fail
    dataset = _adata.obs["dataset"].values[0]  # Get the dataset name from the subset
    _adata.write_h5ad(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
