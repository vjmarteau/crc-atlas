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
# # Dataloader: Kong_2023_Immunity

# %%
import os

import anndata
import fast_matrix_market as fmm
import numpy as np
import pandas as pd
import sc_atlas_helpers as ah
import scanpy as sc
import yaml
from nxfvars import nxfvars
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits
from tqdm.contrib.concurrent import process_map

# %% tags=["parameters"]
artifact_dir = nxfvars.get("artifact_dir", "/data/scratch/marteau/downloads/crc-atlas/")
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")
cpus = nxfvars.get("cpus", 6)

# %%
d10_1016_j_immuni_2023_01_002 = f"{databases_path}/d10_1016_j_immuni_2023_01_002"
gtf = f"{annotation}/ensembl.v84_gene_annotation_table.csv"

# %%
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

# %%
# Read the YAML file and load it into a dictionary
with open(reference_meta, "r") as f:
    ref_meta_dict = yaml.load(f, Loader=yaml.Loader)

# %% [markdown]
# ## 1. Load raw adata counts

# %%
def load_counts(file):
    adata = sc.AnnData(
        X=fmm.mmread(f"{d10_1016_j_immuni_2023_01_002}/SCP1884/{file}.scp.raw.mtx").T,
        obs=pd.read_csv(
            f"{d10_1016_j_immuni_2023_01_002}/SCP1884/{file}.scp.barcodes.tsv",
            delimiter="\t",
            header=None,
            index_col=0,
        ),
        var=pd.read_csv(
            f"{d10_1016_j_immuni_2023_01_002}/SCP1884/{file}.scp.features.tsv",
            delimiter="\t",
            header=None,
            names=["ensembl", "symbol"],
            index_col=0,
        ),
    )
    adata.X = csr_matrix(adata.X)
    adata.var_names.name = None
    adata.obs_names.name = None
    adata.obs["type"] = file
    return adata

# %%
adatas = process_map(
    load_counts,
    ["CO_EPI", "CO_IMM", "CO_STR", "TI_EPI", "TI_IMM", "TI_STR"],
    max_workers=cpus,
)
adata = anndata.concat(adatas, join="outer")

# %%
cell_meta = pd.read_csv(
    f"{d10_1016_j_immuni_2023_01_002}/SCP1884/scp_metadata_combined.v2.txt",
    delimiter="\t",
).rename({"NAME": "barcodes"}, axis=1)

adata.obs["barcodes"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    cell_meta,
    how="left",
    on=["barcodes"],
    validate="m:1",
).set_index("barcodes")
adata.obs_names.name = None

adata.obs["patient_id"] = adata.obs["biosample_id"].str.split("_", n=1).str[0]

# %%
# subset for healthy colon samples -> could probably just not load "TI_" (terminal ileum) samples above
adata = adata[
    (adata.obs["disease__ontology_label"] == "normal")
    & (adata.obs["organ__ontology_label"] == "colon")
].copy()

# %%
patient_meta = pd.read_csv(f"{d10_1016_j_immuni_2023_01_002}/metadata/TableS1.csv")

# Subset for healthy controls and keep only relevant columns
patient_meta = patient_meta.loc[
    patient_meta["Type (Heal:Non-IBD CD:Crohn's Disease)"] == "Heal"
][["Donor.ID", "Age group", "Sex", "Smoking (Never Current Former)"]].reset_index(
    drop=True
)
patient_meta["Donor.ID"] = "H" + patient_meta["Donor.ID"].astype(str)

# %%
# Paper: These include 24 samples from 12 non-IBD donors published previously
# Most healthy colon samples seem to be from Smillie_2019 (PMID:31348891)
patient_meta_Smillie_2019 = pd.read_csv(
    f"{d10_1016_j_immuni_2023_01_002}/metadata/Smillie_2019_TableS1.csv"
)

patient_meta_Smillie_2019 = patient_meta_Smillie_2019.loc[
    patient_meta_Smillie_2019["Disease"] == "HC"
].reset_index(drop=True)

# %%
# Will do the 2 dataset separately
adata_Smillie_2019 = adata[
    adata.obs["patient_id"].isin(patient_meta_Smillie_2019["Subject ID"])
].copy()

adata_Smillie_2019.obs["barcodes"] = adata_Smillie_2019.obs_names
adata_Smillie_2019.obs = pd.merge(
    adata_Smillie_2019.obs,
    patient_meta_Smillie_2019,
    how="left",
    left_on=["patient_id"],
    right_on=["Subject ID"],
    validate="m:1",
).set_index("barcodes")
adata_Smillie_2019.obs_names.name = None


adata = adata[
    ~adata.obs["patient_id"].isin(patient_meta_Smillie_2019["Subject ID"])
].copy()

adata.obs["barcodes"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    patient_meta,
    how="left",
    left_on=["patient_id"],
    right_on=["Donor.ID"],
    validate="m:1",
).set_index("barcodes")
adata.obs_names.name = None

# %% [markdown]
# ## 2. Harmonize Kong_2023_Immunity to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="Kong_2023_Immunity",
    study_doi="10.1016/j.immuni.2023.01.002",
    study_pmid="36720220",
    tissue_processing_lab="Ramnik Xavier lab",
    dataset="Kong_2023",
    # Paper methods: Healthy controls were recruited at the time of routine colonoscopy
    medical_condition="healthy",
    sample_tissue="colon",
    sample_type="normal",
    treatment_status_before_resection="naive",
    matrix_type="raw counts",
    cellranger_version="3.0.2",
    # reference_genome="",
    hospital_location="Massachusetts General Hospital",
    country="US",
    tissue_cell_state="fresh",
    enrichment_cell_types="naive",
)

# %%
adata.obs["sample_id"] = adata.obs["biosample_id"]
adata.obs["platform"] = adata.obs["library_preparation_protocol__ontology_label"]
adata.obs["cell_type_study"] = adata.obs["Celltype"]
adata.obs["age"] = adata.obs["Age group"]
adata.obs["original_obs_names"] = adata.obs_names

# define the dictionary mapping
mapping = {
    "Female": "female",
    "Male": "male",
}
adata.obs["sex"] = adata.obs["Sex"].map(mapping)

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["sample_id"]
    + "-"
    + adata.obs_names.str.split("-").str[1]
)

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
adata

# %%
adata.obs["patient_id"].value_counts()

# %%
adata.obs["sample_id"].value_counts()

# %%
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")

# %% [markdown]
# ## 3. Harmonize Smillie_2019 to ref_meta_dict

# %%
adata_Smillie_2019.obs = adata_Smillie_2019.obs.assign(
    study_id="Smillie_2019_Cell",
    study_doi="10.1016/j.cell.2019.06.029",
    study_pmid="31348891",
    tissue_processing_lab="Aviv Regev lab",
    dataset="Smillie_2019",
    # Paper methods: Healthy controls were recruited at the time of routine colonoscopy
    medical_condition="healthy",
    sample_tissue="colon",
    sample_type="normal",
    treatment_status_before_resection="naive",
    matrix_type="raw counts",
    cellranger_version="3.0.2",  # "2.0.0" in original study but was reprocessed by Kong_2023
    # reference_genome="",
    hospital_location="Massachusetts General Hospital",
    country="US",
    tissue_cell_state="fresh",
    enrichment_cell_types="naive",
)

# %%
adata_Smillie_2019.obs["sample_id"] = adata_Smillie_2019.obs["biosample_id"]
adata_Smillie_2019.obs["platform"] = adata_Smillie_2019.obs[
    "library_preparation_protocol__ontology_label"
]
adata_Smillie_2019.obs["cell_type_study"] = adata_Smillie_2019.obs["Celltype"]
adata_Smillie_2019.obs["original_obs_names"] = adata_Smillie_2019.obs_names

# %%
# define the dictionary mapping
mapping = {
    "Female": "female",
    "Male": "male",
}
adata_Smillie_2019.obs["sex"] = adata_Smillie_2019.obs["Gender"].map(mapping)

# define the dictionary mapping
mapping = {
    "Right_Colon": "nan",
    "Left_Colon": "nan",
    "Transverse_Colon": "transverse colon",
    "Cecum": "cecum",
    "nan": "nan",
}
adata_Smillie_2019.obs["anatomic_location"] = (
    adata_Smillie_2019.obs["Location"].astype(str).map(mapping)
)

# define the dictionary mapping
mapping = {
    "Right_Colon": "proximal colon",
    "Left_Colon": "distal colon",
    "Transverse_Colon": "proximal colon",
    "Cecum": "proximal colon",
    "nan": "nan",
}
adata_Smillie_2019.obs["anatomic_region"] = (
    adata_Smillie_2019.obs["Location"].astype(str).map(mapping)
)

# separated and independently processed the epithelial (E) and lamina propria (L) fractions
mapping = {
    "E": "epithelial fraction",
    "L": "lamina propria fraction",
}
adata_Smillie_2019.obs["Layer"] = adata_Smillie_2019.obs["Layer"].map(mapping)

adata_Smillie_2019.obs["dataset"] = adata_Smillie_2019.obs["dataset"] + "_" + adata_Smillie_2019.obs["platform"].astype(str)

mapping = {
    "Smillie_2019_10x 3' v1": "Smillie_2019_10Xv1",
    "Smillie_2019_10x 3' v2": "Smillie_2019_10Xv2",
}
adata_Smillie_2019.obs["dataset"] = adata_Smillie_2019.obs["dataset"].map(mapping)

adata_Smillie_2019.obs_names = (
    adata_Smillie_2019.obs["dataset"] + "-" + adata_Smillie_2019.obs_names.str.split("-", n=1).str[1]
)

# %%
ref_meta_cols += ["Layer"]

# Subset adata for columns in the reference meta
adata_Smillie_2019.obs = adata_Smillie_2019.obs[
    [col for col in ref_meta_cols if col in adata_Smillie_2019.obs.columns]
].copy()

# %%
adata_Smillie_2019.obs_names = (
    adata_Smillie_2019.obs["dataset"]
    + "-"
    + adata_Smillie_2019.obs["sample_id"]
    + "-"
    + adata_Smillie_2019.obs_names.str.split("-").str[1]
)

# %%
assert np.all(np.modf(adata_Smillie_2019.X.data)[0] == 0), "X does not contain all integers"
assert adata_Smillie_2019.var_names.is_unique
assert adata_Smillie_2019.obs_names.is_unique

# %%
adata_Smillie_2019.obs["patient_id"].value_counts()

# %%
adata_Smillie_2019.obs["sample_id"].value_counts()

# %% [markdown]
# ## 4. Save adata by dataset

# %%
datasets = [
    adata_Smillie_2019[adata_Smillie_2019.obs["dataset"] == dataset, :].copy()
    for dataset in adata_Smillie_2019.obs["dataset"].unique()
    if adata_Smillie_2019[adata_Smillie_2019.obs["dataset"] == dataset, :].shape[0] != 0
]

# %%
datasets

# %%
for _adata in datasets:
    _adata.obs = _adata.obs.dropna(axis=1, how="all")  # Remove all nan columns, else write_h5ad will fail
    dataset = _adata.obs["dataset"].values[0]  # Get the dataset name from the subset
    _adata.write_h5ad(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
