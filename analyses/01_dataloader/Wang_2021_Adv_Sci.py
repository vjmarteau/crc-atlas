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
# # Dataloader: Wang_2021_Adv_Sci

# %%
import os
import re

import numpy as np
import pandas as pd
import sc_atlas_helpers as ah
import scanpy as sc
import yaml
from nxfvars import nxfvars
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

# %% tags=["parameters"]
artifact_dir = nxfvars.get("artifact_dir", "/data/scratch/marteau/downloads/crc-atlas/")
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")
d10_1002_advs_202004320 = f"{databases_path}/d10_1002_advs_202004320"

# %%
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

# %%
# Read the YAML file and load it into a dictionary
with open(reference_meta, "r") as f:
    ref_meta_dict = yaml.load(f, Loader=yaml.Loader)

# %%
counts = pd.read_csv(
    f"{d10_1002_advs_202004320}/10_nfcore_rnaseq/star_salmon/salmon.merged.gene_counts_length_scaled.tsv",
    sep="\t",
)

# %%
counts["gene_id"] = counts["gene_id"].apply(ah.pp.remove_gene_version)
mat = counts.drop(columns=["gene_name"]).T
mat.columns = mat.iloc[0]
mat = mat[1:]
mat = mat.rename_axis(None, axis=1)

# %%
# Round length corrected plate-based study
mat = np.ceil(mat).astype(int)

adata = sc.AnnData(mat)
adata.X = csr_matrix(adata.X)

adata.obs["sample"] = adata.obs_names

# %%
# map back gene symbols to .var
gene_symbols = counts.set_index("gene_id")["gene_name"].to_dict()

adata.var["ensembl"] = adata.var_names
adata.var["symbol"] = (
    adata.var["ensembl"].map(gene_symbols).fillna(value=adata.var["ensembl"])
)

# %%
fetchngs = pd.read_csv(
    f"{d10_1002_advs_202004320}/fetchngs/PRJNA394593/samplesheet/samplesheet.csv"
)

# Merge accession numbers
adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    fetchngs,
    how="left",
    on=["sample"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

adata.obs["original_obs_names"] = adata.obs["sample_alias"]
adata.obs["patient_id"] = adata.obs["sample_alias"].str.rsplit("l", n=1).str[0]

# %%
# Not sure what "l" and "l[digit]" stands for,
# according to the paper metods only 10x data has matched normal ->  For 10 × Genomics analysis, primary colorectal carcinomas and their corresponding control tissues as well as clinical information were provided by the Tissue Bank of Yale-New Haven Hospital
adata.obs["sample_alias"].str.rsplit("_", n=1).str[0].unique()

# %%
# Append "_t" to samples to make them different from patient ids
adata.obs["sample_id"] = adata.obs["patient_id"] + "_t"

# %%
metatable = pd.read_csv(f"{d10_1002_advs_202004320}/metadata/TableS1.csv")

# %%
# Merge accession numbers
adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    metatable,
    how="left",
    left_on=["patient_id"],
    right_on=["CRC patient number"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
# Get original sample/patient meta and remove cell level meta
sample_meta = (
    adata.obs[
        adata.obs.columns[adata.obs.nunique() <= adata.obs["sample_id"].nunique()]
    ]
    .groupby("sample_id")
    .first()
    .reset_index()
)

# %%
sample_meta.to_csv(
    f"{artifact_dir}/Wang_2021-original_sample_meta.csv",
    index=False,
)

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="Wang_2021_Adv_Sci",
    study_doi="10.1002/advs.202004320",
    study_pmid="33898197",
    tissue_processing_lab="Lin Liu lab",
    dataset="Wang_2021",
    medical_condition="colorectal cancer",
    hospital_location="Tianjin Medical University General Hospital",
    country="China",
    platform="Smart-seq2",
    reference_genome="gencode.v44",
    matrix_type="raw counts",
    # Freshly resected primary tumors from eight CRC patients who were treatment-naïve at the time of surgery
    tissue_cell_state="fresh",
    treatment_status_before_resection="naive",
    enrichment_cell_types="naive",
    sample_tissue="colon",
    sample_type="tumor",
)

# %%
adata.obs["NCBI_BioProject_accession"] = adata.obs["study_accession"]
adata.obs["SRA_sample_accession"] = adata.obs["secondary_sample_accession"]

adata.obs["age"] = adata.obs["Age"].fillna(np.nan).astype("Int64")
# define the dictionary mapping
mapping = {
    "Female": "female",
    "Male": "male",
}
adata.obs["sex"] = adata.obs["Gender"].map(mapping)

mapping = {
    "Adenocarcinoma": "adenocarcinoma",
}
adata.obs["histological_type"] = adata.obs["Type"].map(mapping)

mapping = {
    "Rectum": "rectum",
    "Sigmoid colon": "sigmoid colon",
    "Right colon": "nan",
    "Subtotal colectomy": "nan",  # -> what part have they been using for dissociation ??
}
adata.obs["anatomic_location"] = adata.obs["Tumor site"].map(mapping)

mapping = {
    "Rectum": "distal colon",
    "Sigmoid colon": "distal colon",
    "Right colon": "proximal colon",
    "Subtotal colectomy": "proximal colon",  # -> Subtotal colectomy: surgeon leaves part of the left side ?
}
adata.obs["anatomic_region"] = adata.obs["Tumor site"].map(mapping)


mapping = {
    "Well-Moderate": "G1-G2",
    "Moderate": "G2",
    "Moderate-Poor": "G2-G3",
}
adata.obs["tumor_grade"] = adata.obs["Histology"].map(mapping)

adata.obs["tumor_size"] = adata.obs["SIZE(CM)"]

adata.obs["tumor_stage_TNM_T"] = adata.obs["Depth of invasion(T)"]
adata.obs["tumor_stage_TNM_N"] = adata.obs["Lymph node involvement(N)"]
adata.obs["tumor_stage_TNM_M"] = adata.obs["Metastasis(M)"]

# Matches "Stage" column from original meta
adata.obs["tumor_stage_TNM"] = ah.pp.tnm_tumor_stage(adata.obs, "patient_id")

# %%
# Stage matches
adata.obs[["patient_id", "tumor_stage_TNM", "TNM staging"]].groupby(
    "patient_id"
).first()

# %%
adata.obs_names = (
    adata.obs["dataset"] + "-" + adata.obs_names + "-" + adata.obs["sample_alias"]
)

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
# Add back potentially interesting metadata that does not fit the reference meta
ref_meta_cols += [
    "Dukes classification",
]

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
adata

# %%
adata.obs["sample_id"].value_counts()

# %%
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
