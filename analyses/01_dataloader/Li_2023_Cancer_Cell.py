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
# # Dataloader: Li_2023_Cancer_Cell

# %%
import os
import re
from pathlib import Path

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
cpus = nxfvars.get("cpus", 6)
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")

# %%
gtf = f"{annotation}/ensembl.v93_gene_annotation_table.csv"
d10_1016_j_ccell_2023_04_011 = f"{databases_path}/d10_1016_j_ccell_2023_04_011"
GSE205506 = f"{d10_1016_j_ccell_2023_04_011}/GSE205506/GSE205506_RAW/"

# %%
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

# %%
# Read the YAML file and load it into a dictionary
with open(reference_meta, "r") as f:
    ref_meta_dict = yaml.load(f, Loader=yaml.Loader)

# %% [markdown]
# ## 1. Load raw adata counts from GEO

# %%
# Get file names
files = [str(x) for x in Path(GSE205506).glob("*.mtx.gz")]
files = [x for x in files if not isinstance(x, float)]
files.sort()

samples = [Path(file).stem.split("_matrix.mtx")[0] for file in files]

meta = pd.DataFrame({"FileName": samples})
meta["sample_geo_accession"] = meta["FileName"].str.split("_").str[0]

# %%
geofetch = pd.read_csv(
    f"{d10_1016_j_ccell_2023_04_011}/GSE205506/GSE205506_PEP/GSE205506_PEP_raw.csv"
).dropna(axis=1, how="all")

meta = pd.merge(
    meta,
    geofetch,
    how="left",
    on=["sample_geo_accession"],
    validate="m:1",
)

# %%
patient_meta = pd.read_csv(f"{d10_1016_j_ccell_2023_04_011}/metadata/TableS1.csv")

meta = pd.merge(
    meta,
    patient_meta,
    how="left",
    left_on=["subject"],
    right_on=["PatientID"],
    validate="m:1",
)

meta = meta.copy()

# %%
meta.to_csv(f"{artifact_dir}/Li_2023-original_sample_meta.csv", index=False)


# %%
def _load_mtx(file_meta):
    FileName = f"{GSE205506}/{file_meta['FileName']}"
    adata = anndata.AnnData(
        X=fmm.mmread(f"{FileName}_matrix.mtx.gz").T,
        obs=pd.read_csv(
            f"{FileName}_barcodes.tsv.gz",
            delimiter="\t",
            header=None,
            names=["obs_names"],
            index_col=0,
        ),
        var=pd.read_csv(
            f"{FileName}_features.tsv.gz",
            delimiter="\t",
            header=None,
            names=["ensembl", "symbol"],
            index_col=0,
        ),
    )
    adata.obs = adata.obs.assign(**file_meta)
    adata.obs_names.name = None
    adata.obs["original_obs_names"] = adata.obs_names
    adata.X = csr_matrix(adata.X)
    return adata


# %%
adatas = process_map(_load_mtx, [r for _, r in meta.iterrows()], max_workers=cpus)
adata = anndata.concat(adatas, index_unique="-", join="outer", fill_value=0)

# %% [markdown]
# ### 1. Gene annotations

# %%
# get id mappings from one of the samples
# map back gene symbols to .var
gene_symbols = adatas[0].var["ensembl"].to_dict()

adata.var["ensembl"] = adata.var_names
adata.var["symbol"] = (
    adata.var["ensembl"].map(gene_symbols).fillna(value=adata.var["ensembl"])
)
adata.var_names = adata.var["ensembl"]
adata.var_names.name = None

# %%
unmapped_genes = ah.pp.find_unmapped_genes(adata)
len(unmapped_genes)

# %% [markdown]
# ## 2. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="Li_2023_Cancer_Cell",
    study_doi="10.1016/j.ccell.2023.04.011",
    study_pmid="37172580",
    tissue_processing_lab="Yanhong Deng lab",
    dataset="Li_2023",
    medical_condition="colorectal cancer",
    matrix_type="raw counts",
    # Paper Methods: v4.1.0, GEO: data processing v5.0.1
    cellranger_version="4.1.0",
    reference_genome="ensembl.v93",
    NCBI_BioProject_accession="PRJNA846169",
    sample_tissue="colon",
    # Tumor and adjacent normal samples were first stored in liquid nitrogen, or proceed directly to tissue disassociation.
    # Does this mean untreated tumor samples are fresh, treated tumor + matched normal are frozen??
    tissue_cell_state="fresh/frozen",
    enrichment_cell_types="naive",
    #histological_type="nan",# ?
    # 19 patients with d-MMR/MSI-H CRC -> Not sure why some patients are nan for microsatelite status in Table S1
    mismatch_repair_deficiency_status="dMMR",
    hospital_location="Department of Medical Oncology, Department of General Surgery, The Sixth Affiliated Hospital, Sun Yat-sen University, Guangzhou",
    country="China",
)

# %%
adata.obs["patient_id"] = adata.obs["PatientID"]
adata.obs["sample_id"] = (
    adata.obs["FileName"]
    .str.split("_", n=2)
    .str[2]
    .apply(lambda x: "MGI_" + x if not x.startswith("MGI_") else x)
)
adata.obs["sample_id"] = adata.obs["sample_id"].str.rsplit("_", n=1).str[0]
adata.obs["sample_type"] = adata.obs["sample_title"].str.split(", ").str[2]
adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]

adata.obs["age"] = adata.obs["Age"].fillna(np.nan).astype("Int64")

mapping = {
    "M": "male",
    "F": "female",
}
adata.obs["sex"] = adata.obs["Sex"].map(mapping)

mapping = {
    "Hepatic flexure of colon": "hepatic flexure",
    "Transverse colon": "transverse colon",
    "Ascending colon": "ascending colon",
    "Descending colon": "descending colon",
    "Sigmoid colon": "sigmoid colon",
    "Rectum": "rectum",
}
adata.obs["anatomic_location"] = adata.obs["Tumor anatomical location"].map(mapping)

mapping = {
    "hepatic flexure": "proximal colon",
    "transverse colon": "proximal colon",
    "ascending colon": "proximal colon",
    "descending colon": "distal colon",
    "sigmoid colon": "distal colon",
    "rectum": "distal colon",
}
adata.obs["anatomic_region"] = adata.obs["anatomic_location"].map(mapping)

adata.obs["microsatellite_status"] = adata.obs["Microsatellite status"]

mapping = {
    "Anti-PD-1+celecoxib": "treated",
    "anti-PD-1": "treated",
    "untreated": "naive",
}
adata.obs["treatment_status_before_resection"] = (
    adata.obs["sample_title"].str.split(", ").str[3].map(mapping)
)

mapping = {
    "Anti-PD-1+celecoxib": "Anti-PD-1|celecoxib",
    "anti-PD-1": "Anti-PD-1",
    "untreated": "naive",
}
adata.obs["treatment_drug"] = (
    adata.obs["sample_title"].str.split(", ").str[3].map(mapping)
)

mapping = {
    "pCR": "responder",
    "non-pCR": "non-responder",
}
adata.obs["treatment_response"] = adata.obs["Treatment response"].map(mapping)

# GEO Single-cell RNA-seq libraries were prepared using the Chromium Next GEM Single Cell 3’ Kit v2 or v3 from 10x Genomics
mapping = {
    "v2": "10x 3' v2",
    "v3": "10x 3' v3",
}
adata.obs["platform"] = adata.obs["FileName"].str.rsplit("_", n=1).str[1].map(mapping)

mapping = {
    "Li_2023_10x 3' v2": "Li_2023_10Xv2",
    "Li_2023_10x 3' v3": "Li_2023_10Xv3",
}
adata.obs["dataset"] = (adata.obs["dataset"] + "_" + adata.obs["platform"]).map(mapping)

# %%
adata.obs["tumor_stage_TNM_T"] = adata.obs["Stage(cTMN)"].apply(
    lambda x: re.search(r"(T[0-4][a-c]?)", x).group(1)
    if re.search(r"(T[0-4][a-c]?)", x)
    else None
)
adata.obs["tumor_stage_TNM_N"] = adata.obs["Stage(cTMN)"].apply(
    lambda x: re.search(r"(N[0-4][a-c]?)", x).group(1)
    if re.search(r"(N[0-4][a-c]?)", x)
    else None
)
adata.obs["tumor_stage_TNM_M"] = adata.obs["Stage(cTMN)"].apply(
    lambda x: re.search(r"((M[0-1]|Mx)[a-c]?)", x).group(1)
    if re.search(r"((M[0-1]|Mx)[a-c]?)", x)
    else None
)

adata.obs["tumor_stage_TNM"] = ah.pp.tnm_tumor_stage(adata.obs, "patient_id")

# %%
adata.obs[
    [
        "patient_id",
        "Stage(cTMN)",
        "tumor_stage_TNM",
        "tumor_stage_TNM_T",
        "tumor_stage_TNM_N",
        "tumor_stage_TNM_M",
    ]
].groupby("patient_id").first()

# %%
# Split and explode the "Key Driver Mutations" column to get unique mutations per patient
mutation_list = (
    adata.obs["Mismatch repair defective protein"]
    .str.split("/")
    .explode()
    .dropna()
    .str.strip()
    .unique()
)

# %%
# List comprehension to create new columns for each unique mutation
for mutation in mutation_list:
    adata.obs[f"{mutation}_status_driver_mut"] = adata.obs[
        "Mismatch repair defective protein"
    ].apply(lambda x: "mut" if mutation in str(x) else "wt" if x != "None" else "nan")

# %%
# Get the column names that contain "_status_driver_mut"
driver_mutation_columns = [
    col for col in adata.obs.columns if "_status_driver_mut" in col
]
# Set the values to NaN for "normal" sample_type in the driver mutation columns
adata.obs.loc[adata.obs["sample_type"] == "normal", driver_mutation_columns] = "nan"

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)
# Add newly created driver_mutation columns
ref_meta_cols += [f"{mutation}_status_driver_mut" for mutation in mutation_list]

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["GEO_sample_accession"]
    + "-"
    + adata.obs_names.str.split("-", n=1).str[0]
)

# %%
adata

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
adata.obs["sample_id"].value_counts()

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
