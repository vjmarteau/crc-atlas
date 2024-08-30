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
# # Dataloader: Tian_2023_Nat_Med

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

# %% tags=["parameters"]
artifact_dir = nxfvars.get("artifact_dir", "/data/scratch/marteau/downloads/crc-atlas/")
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../tables/annotation")
cpus = nxfvars.get("cpus", 2)
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")

# %%
d10_1038_s41591_022_02181_8 = f"{databases_path}/d10_1038_s41591_022_02181_8"
gtf = f"{annotation}/gencode.v30_gene_annotation_table.csv"
SCP2079 = f"{d10_1038_s41591_022_02181_8}/single_cell_portal/SCP2079"

# %%
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

# %%
# Read the YAML file and load it into a dictionary
with open(reference_meta, "r") as f:
    ref_meta_dict = yaml.load(f, Loader=yaml.Loader)

# %%
# Load norm counts from SCP2079 (raw not available even though SCP data pretends to be ...)
adata = anndata.AnnData(
    X=fmm.mmread(f"{SCP2079}/epi_matrix.mtx.gz").T,
    obs=pd.read_csv(
        f"{SCP2079}/epi_barcodes.tsv", delimiter="\t", header=None, index_col=0
    ),
    var=pd.read_csv(
        f"{SCP2079}/features.tsv", delimiter="\t", header=None, index_col=0
    ),
)
adata.obs_names.name = None
adata.var_names.name = None

# Strip potential leading/trailing white space from gene_ids
adata.var_names = [gene_id.strip() for gene_id in adata.var_names]

# Need to give column names str names else saving to h5ad fails
adata.var.rename(columns={1: 'symbol'}, inplace=True)
adata.var.drop(2, axis=1, inplace=True)

# Add embeddings
umap = pd.read_csv(f"{SCP2079}/epi_umap.txt", delimiter="\t", index_col=0).tail(-1)
# adata.obsm["X_umap"] = np.array(umap) #-> no need downstream!

# %%
ah.pp.undo_log_norm(adata)

# %%
adata.X = csr_matrix(adata.X)
adata.layers["log_norm"] = csr_matrix(adata.layers["log_norm"])

# %%
supplemental_info = pd.read_csv(f"{SCP2079}/file_supplemental_info.tsv", delimiter="\t")
meta = pd.read_csv(
    f"{SCP2079}/epi_metadata.txt", delimiter="\t", index_col=False, low_memory=False
).tail(-1)

# %%
supplemental_info

# %%
# Merge available metadata
adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    meta,
    how="left",
    left_on=["obs_names"],
    right_on=["NAME"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
adata.obs = adata.obs.assign(NCBI_BioProject_accession="PRJNA923099")

# %%
sample_meta = (
    adata.obs[adata.obs.columns[adata.obs.nunique() <= adata.obs["sampleID"].nunique()]]
    .groupby("sampleID")
    .first()
    .reset_index()
)
sample_meta = sample_meta.rename(columns={"sampleID": "sample_id"})

# %%
sample_meta.to_csv(
    f"{artifact_dir}/Tian_2023-original_sample_meta.csv", index=False
)

# %% [markdown]
# ## Harmonize metadata to ref_meta_dict

# %% [markdown]
# # Patient demographics
#     - age: median=63 (35-87)
#     - Left Sidedness (n=14)
#         - Left Sidedness: Descending Colon (n=2), Sigmoid (n=8), Rectum (n=3), Rectosigmoid (n=1)
#     - Right Sidedness (n=23)
#         - Right Sidedness: Ascending Colon (n=10), Cecum (n=3), Distal Transverse Colon (n=1), Hepatic Flexure (n=2), Transverse (n=7)
#     
# !!! -> Dataset contains only Epithelial cells, immune and stromal cells were filtered out
#     -> Tumor mutational burden data available
#     -> Key exclusion criteria included chemotherapy or radiotherapy within 4 weeks prior to entering the study -> Not sure if they were all treatment naive before study treatment!
#
# Explanations for [RECIST](https://ctep.cancer.gov/protocoldevelopment/docs/recist_guideline.pdf) classification

# %%
adata.obs = adata.obs.assign(
    study_id="Tian_2023_Nat_Med",
    study_doi="10.1038/s41591-022-02181-8",
    study_pmid="36702949",
    tissue_processing_lab="Ryan Corcoran lab",
    dataset="Tian_2023",
    medical_condition="colorectal cancer",
    # !!! From the methods and paper its not clear what exactly was sampled. According to the paper, all patients had metastasis, and pre/on treatment biopsies were taken from the same location
    # It makes no sense to sample "Muscle" in this context + I checked and patients have only samples within the same adata.obs["tissueSite"]. "organ__ontology_label" says its all liver metastasis
    # Will go with that info! Also what is "Other" supposed to be - primary tumor ( or maybe brain?!)?
    sample_type="metastasis",
    sample_tissue="liver",
    # Paper: Core needle biopsies
    tumor_source="core",
    age="median: 63 (35-87)",
    enrichment_cell_types="naive",
    matrix_type="log norm",
    cellranger_version="6.0.2",
    reference_genome="gencode.v30",
    tissue_cell_state="fresh",
    tissue_dissociation="enzymatic",
    cell_type_study="epithelial cell",
    treatment_drug="spartalizumab|dabrafenib|trametinib",
    hospital_location="Massachusetts General Hospital | Dana-Farber Cancer Institute, Brigham and Women’s Hospital",
    country="US",
    tumor_stage_TNM_T="nan",
    tumor_stage_TNM_N="nan",
    tumor_stage_TNM_M="M1",
    # paper methods: patients with metastatic CRC characterized by the BRAF V600E mutation. Eligible patients must have histologically or cytologically confirmed metastatic CRC,
    # have a documented BRAF V600E mutation by a CLIA-certified laboratory test and be wild type for KRAS and NRAS.
    BRAFV600E_status_driver_mut="mut",
    KRAS_status_driver_mut="wt",
    NRAS_status_driver_mut="wt",
)

adata.obs["sample_id"] = adata.obs["sampleID"]

adata.obs["patient_id"] = adata.obs["donor_id"]

# Paper methods: Per patient and time point, the first two to three cores from the operative procedure were allocated for scRNAseq
adata.obs["replicate"] = adata.obs.apply(
    lambda row: row["sampleID"].replace(str(row["specimenID"]), ""), axis=1
)

# define the dictionary mapping
mapping = {
    "": "no_rep",
    "a": "rep1",
    "b": "rep2",
    "c": "rep3",
}
adata.obs["replicate"] = adata.obs["replicate"].map(mapping)

adata.obs["platform"] = adata.obs["library_preparation_protocol__ontology_label"]

adata.obs["histological_type"] = adata.obs["disease__ontology_label"].replace(
    {"colon adenocarcinoma": "adenocarcinoma"}
)
adata.obs["microsatellite_status"] = adata.obs["MMRstatus"]

adata.obs["tumor_stage_TNM"] = ah.pp.tnm_tumor_stage(adata.obs, "patient_id")

# define the dictionary mapping
mapping = {
    "Pre": "treated",
    "On": "treated",
}
adata.obs["treatment_status_before_resection"] = adata.obs["treatment"].map(mapping)

# define the dictionary mapping
mapping = {
    "R": "responder",
    "NR": "non-responder",
}
adata.obs["treatment_response"] = adata.obs["response"].map(mapping)

adata.obs["progression-free survival [month]"] = (
    adata.obs["PFSmo"].astype("float").round(1).astype(str)
)

# Response Evaluation Criteria in Solid Tumors (RECIST)
# The current de facto standard for measuring response by imaging in oncology clinical trials,
# RECIST primarily uses anatomic change in lesion size for determining response to treatment.
mapping = {
    "CR": "CR: complete response",
    "PR": "PR: partial response",
    "PD": "PD: progressive disease",
    "SD": "SD: stable disease",
    "NE": "NE: inevaluable",
}
adata.obs["RECIST"] = adata.obs["RECIST"].map(mapping)

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)
# Add newly created driver_mutation columns
ref_meta_cols += [col for col in adata.obs.columns if "_status_driver_mut" in col]

# %%
# Add back potentially interesting metadata that does not fit the reference meta
ref_meta_cols += [
    "progression-free survival [month]",
    "pctChange",
    "pctChangeNoPre",
    "resp_tx",
]

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
adata.obs_names = adata.obs["dataset"] + "-" + adata.obs["sample_id"] + "-" + adata.obs_names.str.split("-").str[0]

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
adata

# %%
adata.obs["patient_id"].value_counts()

# %%
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
