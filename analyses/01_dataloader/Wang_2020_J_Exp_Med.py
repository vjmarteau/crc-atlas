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
# # Dataloader: Wang_2020_J_Exp_Med

# %%
import itertools
import os
import re

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
d10_1084_jem_20191130 = f"{databases_path}/d10_1084_jem_20191130"

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
# Get available metadata
SraRunTable = pd.read_csv(
    f"{d10_1084_jem_20191130}/fetchngs/PRJNA518129/samplesheet/samplesheet.csv"
)

# %%
geofetch = pd.read_csv(
    f"{d10_1084_jem_20191130}/GSE125970/GSE125970_PEP/GSE125970_PEP_raw.csv"
).dropna(axis=1, how="all")

# %%
meta = (
    pd.merge(
        SraRunTable,
        geofetch,
        how="left",
        left_on=["sample"],
        right_on=["srx"],
        validate="m:1",
    )
    .rename(columns={"sample_title_x": "sample_title"})
    .drop("sample_title_y", axis=1)
)

# %%
patient_meta = pd.read_csv(
    f"{d10_1084_jem_20191130}/metadata/jem_20191130_sm-FigS1A.csv"
)
patient_meta["sample_title"] = (
    patient_meta["Human Intestine"] + "-" + patient_meta["Sample"].astype(str)
)

meta = pd.merge(
    meta,
    patient_meta,
    how="left",
    on=["sample_title"],
    validate="m:1",
).rename(columns={"run_accession": "SRA_sample_accession"})


# %%
def load_counts(sample_meta, databases_path, mat_type="filtered"):

    file_path = f"{databases_path}/10_nfcore_scrnaseq/cellranger/count/{sample_meta['SRA_sample_accession']}/outs/{mat_type}_feature_bc_matrix.h5"
    adata = sc.read_10x_h5(file_path)
    adata.obs = adata.obs.assign(**sample_meta)
    adata.var_names_make_unique()
    adata.X = csr_matrix(adata.X)
    adata.obs_names = (
        adata.obs["SRA_sample_accession"] + "-" + adata.obs_names.str.replace("-1", "")
    )
    assert adata.obs_names.is_unique
    return adata


# %%
adatas = process_map(
    load_counts,
    [r for _, r in meta.iterrows()],
    itertools.repeat(d10_1084_jem_20191130),
    itertools.repeat("raw"),
    max_workers=cpus,
)

adata = anndata.concat(adatas, join="outer")

# %% [markdown]
# ## 2. Gene annotations

# %%
# Append gtf gene info
gtf = f"{annotation}/gencode.v44_gene_annotation_table.csv"
gtf = pd.read_csv(gtf)

adata.var.reset_index(names="symbol", inplace=True)
gtf["symbol"] = adata.var["symbol"]
gtf["ensembl"] = gtf["Geneid"].apply(ah.pp.remove_gene_version)
gtf.set_index("ensembl", inplace=True)
adata.var = gtf.copy()
adata.var_names.name = None

# %%
# Sum up duplicate gene ids
duplicated_ids = adata.var_names[adata.var_names.duplicated()].unique()
adata = ah.pp.aggregate_duplicate_gene_ids(adata, duplicated_ids)

# %% [markdown]
# ## 3. Compile original meta from study supplements

# %%
adata.obs = adata.obs.assign(
    study_id="Wang_2020_J_Exp_Med",
    study_doi="10.1084/jem.20191130",
    study_pmid="31753849",
    tissue_processing_lab="Ye-Guang Chen lab",
    dataset="Wang_2020",
    medical_condition="colorectal cancer",
    hospital_location="Peking University Third Hospital, Beijing",
    country="China",
    enrichment_cell_types="naive",
    tissue_cell_state="fresh",
    sample_type="normal",
    cellranger_version="7.1.0",
    reference_genome="gencode.v44",
    platform="10x 3' v2",
    matrix_type="raw counts",
)

# %%
adata.obs["sample_id"] = adata.obs["sample_title"].str.replace("-", "")
adata.obs["patient_id"] = adata.obs["srr"]

adata.obs["NCBI_BioProject_accession"] = adata.obs["study_accession"]
adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]

adata.obs["age"] = adata.obs["Age"].fillna(np.nan).astype("Int64")

# define the dictionary mapping
mapping = {
    "F": "female",
    "M": "male",
}
adata.obs["sex"] = adata.obs["Sex"].map(mapping)

# define the dictionary mapping
mapping = {
    "Ileum": "ileum",
    "Ascending colon": "ascending colon",
    "Rectum": "rectum",
}
adata.obs["anatomic_location"] = adata.obs["Epithelial Tissue Source"].map(mapping)

# define the dictionary mapping
mapping = {
    "Ileum": "nan",
    "Ascending colon": "proximal colon",
    "Rectum": "distal colon",
}
adata.obs["anatomic_region"] = adata.obs["Epithelial Tissue Source"].map(mapping)


# define the dictionary mapping
mapping = {
    "Ileum": "small intestine",
    "Ascending colon": "colon",
    "Rectum": "colon",
}
adata.obs["sample_tissue"] = adata.obs["Epithelial Tissue Source"].map(mapping)

# define the dictionary mapping
mapping = {
    "Adenocarcinoma": "adenocarcinoma",
    "Neuroendocrine Carcinoma": "neuroendocrine carcinoma",
}
adata.obs["histological_type"] = adata.obs["Diagnosis"].map(mapping)

# %%
adata.obs["tumor_stage_TNM_T"] = adata.obs["Stage TNM"].apply(
    lambda x: re.search(r"(T[0-4][a-c]?)", x).group(1)
    if re.search(r"(T[0-4][a-c]?)", x)
    else None
)
adata.obs["tumor_stage_TNM_N"] = adata.obs["Stage TNM"].apply(
    lambda x: re.search(r"(N[0-4][a-c]?)", x).group(1)
    if re.search(r"(N[0-4][a-c]?)", x)
    else None
)
adata.obs["tumor_stage_TNM_M"] = adata.obs["Stage TNM"].apply(
    lambda x: re.search(r"((M[0-1]|Mx)[a-c]?)", x).group(1)
    if re.search(r"((M[0-1]|Mx)[a-c]?)", x)
    else None
)

adata.obs["tumor_stage_TNM"] = ah.pp.tnm_tumor_stage(adata.obs, "patient_id")

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
adata.obs_names = adata.obs["dataset"] + "-" + adata.obs_names

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
adata

# %%
# Save unfiltered raw counts
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
