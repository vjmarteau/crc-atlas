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
# # Dataloader: Bian_2018_Science

# %%
import os

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
cpus = nxfvars.get("cpus", 2)
d10_1126_science_aao3791 = f"{databases_path}/d10_1126_science_aao3791"

# %%
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

# %%
# Read the YAML file and load it into a dictionary
with open(reference_meta, "r") as f:
    ref_meta_dict = yaml.load(f, Loader=yaml.Loader)

# %% [markdown]
# ## 1. Load raw counts

# %%
counts = pd.read_csv(
    f"{d10_1126_science_aao3791}/10_nfcore_rnaseq/star_salmon/salmon.merged.gene_counts_length_scaled.tsv",
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

# %%
# map back gene symbols to .var
gene_symbols = counts.set_index("gene_id")["gene_name"].to_dict()

adata.var["ensembl"] = adata.var_names
adata.var["symbol"] = (
    adata.var["ensembl"].map(gene_symbols).fillna(value=adata.var["ensembl"])
)

# %%
adata.obs["sample_id"] = adata.obs_names
adata.obs["Patient ID"] = adata.obs_names.str.split("_").str[0]

# %%
geofetch = pd.read_csv(
    f"{d10_1126_science_aao3791}/GSE97693/GSE97693_PEP/GSE97693_PEP_raw.csv"
).dropna(axis=1, how="all")

fetchngs = pd.read_csv(
    f"{d10_1126_science_aao3791}/fetchngs_meta/PRJNA382695/samplesheet/samplesheet.csv"
)

accessions = pd.merge(
    fetchngs,
    geofetch,
    how="left",
    on=["sample_title"],
    validate="m:1",
)

accessions["sample_id"] = (
    accessions["sample_name"].str.split("_").str[1:4].str.join("_").str.upper()
)

# %%
# Merge accession numbers
adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    accessions,
    how="left",
    on=["sample_id"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
metatable = pd.read_excel(
    f"{d10_1126_science_aao3791}/metadata/aao3791_table_s1(copy).xlsx"
)

# Merge accession numbers
adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    metatable[["Patient ID", "Gender", "Age", "AJCC Stage", "Primary Site"]],
    how="left",
    on=["Patient ID"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
adata.obs["Samples"] = (
    adata.obs_names.str.split("_").str[1].str.replace(r"\d+", "", regex=True)
)

# define the dictionary mapping
mapping = {
    "LN": "Lymph Node Metastasis",
    "ML": "Treatment-naive Liver Metastasis",
    "MP": "Post-treatment Liver Metastasis",
    "NC": "Adjacent Normal Colon",
    "PT": "Primary Tumor",
}
adata.obs["Samples"] = adata.obs["Samples"].map(mapping)

# %%
adata.obs["Samples"].value_counts()

# %% [markdown]
# -> Almost no matched normal cells!

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
    f"{artifact_dir}/Bian_2018-original_sample_meta.csv",
    index=False,
)

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="Bian_2018_Science",
    study_doi="10.1126/science.aao3791",
    study_pmid="30498128",
    tissue_processing_lab="Fuchou Tang lab",
    dataset="Bian_2018",
    medical_condition="colorectal cancer",
    matrix_type="raw counts",
    reference_genome="gencode.v44",
    platform="scTrio-seq2",
    hospital_location="Department of General Surgery, Peking University Third Hospital, Beijing",
    country="China",
    # Paper methods: Patient derived tumors and adjacent normal tissue were collected and processed immediately after surgical resection.
    tissue_cell_state="fresh",
    # Paper methods: The single viable cells were individually picked into 200-μL tubes containing the lysis buffer of scTrio-seq2. -> not sorted!
    enrichment_cell_types="naive",
)

adata.obs["NCBI_BioProject_accession"] = adata.obs["study_accession"]
adata.obs["SRA_sample_accession"] = adata.obs["secondary_sample_accession"]
adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]

adata.obs["patient_id"] = adata.obs["Patient ID"]

adata.obs["age"] = adata.obs["Age"].fillna(np.nan).astype("Int64")
# define the dictionary mapping
mapping = {
    "Female": "female",
    "Male": "male",
}
adata.obs["sex"] = adata.obs["Gender"].map(mapping)

# %%
# Paper methods: For each tumor, 3-6 regions, including both surface and center areas, were sampled (Figs. 1A, S2).

adata.obs["tumor_source"] = adata.obs_names.str.split("_").str[1]
adata.obs["tumor_source"] = np.where(
    adata.obs["tumor_source"].str.startswith("PT"), adata.obs["tumor_source"], np.nan
)

# From Figure S2 it looks like colon sample PT1 is tumor core, and the rest border
mapping = {
    "PT1": "core",
    "PT2": "border",
    "PT3": "border",
    "PT4": "border",
    "PT5": "border",
    "PT6": "border",
}

adata.obs["tumor_source"] = adata.obs["tumor_source"].map(mapping)

# %%
# define the dictionary mapping
mapping = {
    "Left Colon": "distal colon",
    "Right Colon": "proximal colon",
    "Rectum": "distal colon",
}
adata.obs["anatomic_region"] = adata.obs["Primary Site"].map(mapping)

# define the dictionary mapping
mapping = {
    "Left Colon": "nan",
    "Right Colon": "nan",
    "Rectum": "rectum",
}
adata.obs["anatomic_location"] = adata.obs["Primary Site"].map(mapping)

# define the dictionary mapping
mapping = {
    "Primary Tumor": "tumor",
    "Adjacent Normal Colon": "normal",
    "Lymph Node Metastasis": "lymph node",
    "Treatment-naive Liver Metastasis": "metastasis",
    "Post-treatment Liver Metastasis": "metastasis",
}

adata.obs["sample_type"] = adata.obs["Samples"].map(mapping)

# define the dictionary mapping
mapping = {
    "Primary Tumor": "colon",
    "Adjacent Normal Colon": "colon",
    "Lymph Node Metastasis": "lymph node",
    "Treatment-naive Liver Metastasis": "liver",
    "Post-treatment Liver Metastasis": "liver",
}

adata.obs["sample_tissue"] = adata.obs["Samples"].map(mapping)


adata.obs["tumor_stage_TNM_T"] = "nan"
adata.obs["tumor_stage_TNM_N"] = "nan"
adata.obs["tumor_stage_TNM"] = adata.obs["AJCC Stage"]

# Paper: for 12 CRC patients (stage III or IV)
mapping = {
    "III": "nan",
    "IV": "M1",
}

adata.obs["tumor_stage_TNM_M"] = adata.obs["tumor_stage_TNM"].map(mapping)


# define the dictionary mapping
mapping = {
    "Primary Tumor": "naive",
    "Adjacent Normal Colon": "naive",
    "Lymph Node Metastasis": "naive",
    "Treatment-naive Liver Metastasis": "naive",
    "Post-treatment Liver Metastasis": "treated",
}

adata.obs["treatment_status_before_resection"] = adata.obs["Samples"].map(mapping)


# Paper Figure S2 caption: MP was sampled after 6 cycles of chemotherapy from the same patient.
# define the dictionary mapping
mapping = {
    "naive": "nan",
    "treated": "6x chemotherapy",
}

adata.obs["treatment_drug"] = adata.obs["treatment_status_before_resection"].map(mapping)

# %%
mapping = {
    "CRC01": "Tang_protocol", # Tang 2009; PMID: 19349980
    "CRC02": "Tang_protocol",
    "CRC03": "Dong_protocol", # Dong 2018; PMID: 29540203
    "CRC04": "Dong_protocol",
    "CRC06": "Dong_protocol",
    "CRC09": "Dong_protocol",
    "CRC10": "Dong_protocol",
    "CRC11": "Dong_protocol",
}

adata.obs["platform"] = adata.obs["platform"] + "_" + adata.obs["patient_id"].map(mapping)
adata.obs["dataset"] = adata.obs["dataset"] + "_" + adata.obs["patient_id"].map(mapping)

# %%
# Paper: Most cancer cells from six of our study patients (CRC01, CRC03, CRC04, CRC06, CRC09, and CRC11) were assigned to the CMS2 group (fig. S3, A to C),
# a canonical CRC group with abnormal activation of the WNT/β-catenin and MYC signaling pathways, frequent SCNAs, and nonhypermutation.
# Whole-genome sequencing (WGS) verified a low frequency of somatic single-nucleotide variations (SNVs) in tumors from CRC01 (fig. S3D)
# and identified an oncogenic mutation of NRAS and inactivating mutations of APC and SMAD4, consistent with the features of non-hypermutated CRC.

adata.obs["NRAS_status_driver_mut"] = "wt"
adata.obs["NRAS_status_driver_mut"] = np.where(
    adata.obs["patient_id"] == "CRC01", "mut", adata.obs["NRAS_status_driver_mut"]
)

adata.obs["APC_status_driver_mut"] = "wt"
adata.obs["APC_status_driver_mut"] = np.where(
    adata.obs["patient_id"] == "CRC01", "mut", adata.obs["APC_status_driver_mut"]
)

adata.obs["SMAD4_status_driver_mut"] = "wt"
adata.obs["SMAD4_status_driver_mut"] = np.where(
    adata.obs["patient_id"] == "CRC01", "mut", adata.obs["SMAD4_status_driver_mut"]
)

# %%
# For sample_id remove specific sample region -> too few cells!
adata.obs["sample_id"] = adata.obs_names.str.rsplit("_", n=1).str[0].str.replace(r'\d$', '', regex=True)

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)
    
# Add newly created driver_mutation columns
ref_meta_cols += [col for col in adata.obs.columns if "_status_driver_mut" in col]

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["SRA_sample_accession"]
    + "-"
    + adata.obs_names
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
adata.obs["sample_tissue"].value_counts()

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
