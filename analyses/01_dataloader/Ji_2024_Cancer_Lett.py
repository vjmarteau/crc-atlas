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
#     display_name: Python [conda env:.conda-2024-scanpy]
#     language: python
#     name: conda-env-.conda-2024-scanpy-py
# ---

# %% [markdown]
# # Dataloader: Ji_2024_Cancer_Lett

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
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")
d10_1016_j_canlet_2024_216664 = f"{databases_path}/d10_1016_j_canlet_2024_216664"
gtf = f"{annotation}/ensembl.v111_gene_annotation_table.csv"
cpus = nxfvars.get("cpus", 4)

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
# fastq data
def _load_mtx(sample):
    mat = fmm.mmread(
        f"{d10_1016_j_canlet_2024_216664}/10_celescope_scrnaseq/{sample}/outs/raw/matrix.mtx.gz"
    )
    features = pd.read_csv(
        f"{d10_1016_j_canlet_2024_216664}/10_celescope_scrnaseq/{sample}/outs/raw/features.tsv.gz",
        delimiter="\t",
        header=None,
    ).rename(columns={0: "gene_id", 1: "symbol"})
    barcodes = pd.read_csv(
        f"{d10_1016_j_canlet_2024_216664}/10_celescope_scrnaseq/{sample}/outs/raw/barcodes.tsv.gz",
        delimiter="\t",
        header=None,
    ).rename(columns={0: "barcodes"})

    adata = sc.AnnData(X=mat.T)
    adata.X = csr_matrix(adata.X)

    adata.var = features
    adata.var_names = adata.var["gene_id"]
    adata.var_names.name = None
    adata.var_names_make_unique()

    adata.obs = pd.DataFrame(index=barcodes["barcodes"])
    adata.obs_names.name = None
    adata.obs["sample_id"] = sample

    return adata


# %%
samples = [
    "metastasis",
    "rectum_03N",
    "rectum_03T",
    "sigmoid_03N",
    "sigmoid_03T",
]

# %%
adatas = process_map(_load_mtx, samples, max_workers=cpus)
adata = anndata.concat(adatas, index_unique="-", join="outer", fill_value=0)
adata.obs_names = adata.obs_names.str.rsplit("-", n=1).str[0]

# %%
adata.obs["original_obs_names"] = adata.obs_names
adata.obs_names_make_unique()

# %%
# define the dictionary mapping
mapping = {
    "metastasis": "Liver metastasis",
    "rectum_03N": "Rectum-03N",
    "rectum_03T": "Rectum-03T",
    "sigmoid_03N": "Sigmoid-03N",
    "sigmoid_03T": "Sigmoid-03T",
}
adata.obs["sample_title"] = adata.obs["sample_id"].map(mapping)

# %% [markdown]
# ### 1a. Gene annotations

# %%
# load gtf for gene mapping
gtf = pd.read_csv(gtf)
gtf["ensembl"] = gtf["gene_id"].apply(ah.pp.remove_gene_version)
gene_ids = gtf.set_index("ensembl")["gene_name"].to_dict()

# %%
# Reorder by gtf (i.e. chromosome position)
gene_index = gtf[gtf["ensembl"].isin(adata.var_names)]["ensembl"].values
adata = adata[:, gene_index]

# %% [markdown]
# ### 1b. Compile sample/patient metadata

# %%
adata.obs = adata.obs.assign(
    study_id="Ji_2024_Cancer_Lett",
    study_doi="10.1016/j.canlet.2024.216664",
    study_pmid="38253219",
    tissue_processing_lab="Zengjie Lei lab",
    dataset="Ji_2024",
    medical_condition="colorectal cancer",
    matrix_type="raw counts",
    reference_genome="ensembl.v111",
    cellranger_version="CeleScope v2.0.7",
    platform="GEXSCOPE Singleron",
    hospital_location="Nanjing Jinling Hospital, Jiangsu",
    country="China",
    tissue_cell_state="fresh",
    enrichment_cell_types="naive",
    NCBI_BioProject_accession="PRJNA915211",
)

# %%
geofetch = pd.read_csv(
    f"{d10_1016_j_canlet_2024_216664}/GSE221575/GSE221575_PEP/GSE221575_PEP_raw.csv"
).dropna(axis=1, how="all")

# %%
fetchngs = pd.read_csv(
    f"{d10_1016_j_canlet_2024_216664}/fetchngs/PRJNA915211/samplesheet/samplesheet.csv"
).dropna(axis=1, how="all")

# %%
meta = pd.merge(
    geofetch,
    fetchngs,
    how="left",
    on=["sample_title"],
    validate="m:1",
)

# %%
adata.obs = adata.obs.reset_index(names="obs_names")
adata.obs = pd.merge(
    adata.obs,
    meta,
    how="left",
    on=["sample_title"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
adata.obs["sample_title"] = adata.obs["sample_title"].str.lower().str.replace(" ", "-")

# define the dictionary mapping
mapping = {
    "liver-metastasis": "rectum_03",
    "rectum-03n": "rectum_03",
    "rectum-03t": "rectum_03",
    "sigmoid-03n": "sigmoid_03",
    "sigmoid-03t": "sigmoid_03",
}
adata.obs["patient_id"] = adata.obs["sample_title"].map(mapping)

# %%
patient_meta = pd.read_csv(f"{d10_1016_j_canlet_2024_216664}/metadata/TableS1.csv")
patient_meta["patient_id"] = patient_meta["Patients"].str.lower().str.replace("-", "_")

# %%
adata.obs = adata.obs.reset_index(names="obs_names")
adata.obs = pd.merge(
    adata.obs,
    patient_meta,
    how="left",
    on=["patient_id"],
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
    f"{artifact_dir}/Ji_2024-original_sample_meta.csv",
    index=False,
)

# %%
adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]
adata.obs["SRA_sample_accession"] = adata.obs["run_accession"]
adata.obs["microsatellite_status"] = adata.obs["MSI status"]

mapping = {
    "liver-metastasis": "metastasis",
    "rectum-03n": "normal",
    "rectum-03t": "tumor",
    "sigmoid-03n": "normal",
    "sigmoid-03t": "tumor",
}
adata.obs["sample_type"] = adata.obs["sample_title"].map(mapping)

mapping = {
    "liver-metastasis": "liver",
    "rectum-03n": "colon",
    "rectum-03t": "colon",
    "sigmoid-03n": "colon",
    "sigmoid-03t": "colon",
}
adata.obs["sample_tissue"] = adata.obs["sample_title"].map(mapping)

mapping = {
    "liver-metastasis": "rectum",
    "rectum-03n": "rectum",
    "rectum-03t": "rectum",
    "sigmoid-03n": "sigmoid colon",
    "sigmoid-03t": "sigmoid colon",
}
adata.obs["anatomic_location"] = adata.obs["sample_title"].map(mapping)

mapping = {
    "rectum": "distal colon",
    "sigmoid colon": "distal colon",
}
adata.obs["anatomic_region"] = adata.obs["anatomic_location"].map(mapping)

adata.obs["tumor_stage_TNM"] = adata.obs["tumor_stage"]

mapping = {
    "rectum_03": "M1",
    "sigmoid_03": "M0",
}
adata.obs["tumor_stage_TNM_M"] = adata.obs["patient_id"].map(mapping)

adata.obs["KRAS_status_driver_mut"] = adata.obs["KRAS"].map({"Mutant": "mut"})
adata.obs["BRAF_status_driver_mut"] = adata.obs["BRAF"].map({"Mutant": "mut"})

mapping = {
    "rectum_03": "Ji_2024_scopeV2",
    "sigmoid_03": "Ji_2024_scopeV1",
}
adata.obs["dataset"] = adata.obs["patient_id"].map(mapping)

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
    + adata.obs["SRA_sample_accession"]
    + "-"
    + adata.obs["original_obs_names"]
)
adata.obs_names_make_unique()

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
adata

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
    _adata.obs = _adata.obs.dropna(
        axis=1, how="all"
    )  # Remove all nan columns, else write_h5ad will fail
    dataset = _adata.obs["dataset"].values[0]  # Get the dataset name from the subset
    _adata.write_h5ad(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")


# %% [markdown]
# ### Load GEO counts

# %%
def _load_mtx(sample):
    mat = fmm.mmread(
        f"{d10_1016_j_canlet_2024_216664}/GSE221575/GSE221575_RAW/{sample}_matrix.mtx.gz"
    )
    features = pd.read_csv(
        f"{d10_1016_j_canlet_2024_216664}/GSE221575/GSE221575_RAW/{sample}_genes.tsv.gz",
        delimiter="\t",
        header=None,
    ).rename(columns={0: "gene_id", 1: "symbol"})
    barcodes = pd.read_csv(
        f"{d10_1016_j_canlet_2024_216664}/GSE221575/GSE221575_RAW/{sample}_barcodes.tsv.gz",
        delimiter="\t",
        header=None,
    ).rename(columns={0: "barcodes"})

    adata = sc.AnnData(X=mat.T)
    adata.X = csr_matrix(adata.X)

    adata.var = features
    adata.var_names = adata.var["symbol"]
    adata.var_names.name = None
    adata.var_names_make_unique()

    adata.obs = pd.DataFrame(index=barcodes["barcodes"])
    adata.obs_names.name = None
    adata.obs["sample_id"] = sample

    return adata


# %%
samples = [
    "GSM6886536_Sigmoid-03N",
    "GSM6886537_Sigmoid-03T",
    "GSM6886538_Rectum-03N",
    "GSM6886539_Rectum-03T",
    "GSM6886540_Liver_metastasis",
]
