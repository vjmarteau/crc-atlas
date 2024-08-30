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
#     display_name: Python [conda env:.conda-2024-crc-atas-scanpy]
#     language: python
#     name: conda-env-.conda-2024-crc-atas-scanpy-py
# ---

# %% [markdown]
# # Dataloader: Yang_2023_Front_Oncol

# %%
import itertools
import os
import re

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

# %% tags=["parameters"]
artifact_dir = nxfvars.get("artifact_dir", "/data/scratch/marteau/downloads/crc-atlas/")
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")
cpus = nxfvars.get("cpus", 2)
d10_3389_fonc_2023_1219642 = f"{databases_path}/d10_3389_fonc_2023_1219642"

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
geofetch = pd.read_csv(
    f"{d10_3389_fonc_2023_1219642}/GSE232525/GSE232525_PEP/GSE232525_PEP_raw.csv"
).dropna(axis=1, how="all")

# %%
fetchngs = (
    pd.read_csv(
        f"{d10_3389_fonc_2023_1219642}/fetchngs/PRJNA972665/samplesheet/samplesheet.csv"
    )
    .dropna(axis=1, how="all")
    .drop_duplicates("sample")
)

# %%
meta = pd.merge(
    geofetch,
    fetchngs,
    how="left",
    left_on=["big_key"],
    right_on=["sample"],
    validate="m:1",
)
meta["file"] = meta["big_key"] + "_" + meta["sample_title"].str.split("_").str[0]

# %%
meta.to_csv(
    f"{artifact_dir}/Yang_2023-original_sample_meta.csv",
    index=False,
)


# %%
def load_counts(sample_meta, databases_path, mat_type="filtered"):

    file_path = f"{databases_path}/10_nfcore_scrnaseq/cellranger/count/{sample_meta['file']}/outs/{mat_type}_feature_bc_matrix.h5"
    adata = sc.read_10x_h5(file_path)
    adata.obs = adata.obs.assign(**sample_meta)
    adata.var_names_make_unique()
    adata.X = csr_matrix(adata.X)
    assert adata.obs_names.is_unique
    return adata


# %%
adatas = process_map(
    load_counts,
    [r for _, r in meta.iterrows()],
    itertools.repeat(d10_3389_fonc_2023_1219642),
    itertools.repeat("raw"),
    max_workers=cpus,
)

adata = anndata.concat(adatas, join="outer")

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

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="Yang_2023_Front_Oncol",
    study_doi="10.3389/fonc.2023.1219642",
    study_pmid="37576892",
    tissue_processing_lab="Dongyang Yang lab",
    dataset="Yang_2023",
    medical_condition="colorectal cancer",
    matrix_type="raw counts",
    cellranger_version="7.1.0",
    reference_genome="gencode.v44",
    platform="10x 3' v3",
    hospital_location="Guangdong Provincial People’s Hospital, Guangzhou",
    country="China",
    # Fresh tumor biopsy tissue was dissociated,
    tissue_cell_state="fresh",
    enrichment_cell_types="naive",
    NCBI_BioProject_accession="PRJNA972665",
    sample_type="tumor",
    sample_tissue="colon",
    # Paper Results: we collected biopsy samples from a patient with advanced colon cancer and multiple metastases ... multiple metastases to perienteric lymph nodes
    tumor_stage_TNM_N="N2",
    tumor_stage_TNM_M="M1",
    tumor_stage_TNM="IV",
    # Paper Results: ... confirmed the diagnosis of sigmoid colon cancer
    anatomic_location="sigmoid colon",
    anatomic_region="distal colon",
    patient_id="JCML",
    # Paper Results: The patient received four cycles of FOLFOX chemotherapy and four cycles of bevacizumab, resulting in a partial response
    treatment_drug="4x FOLFOX / 4x bevacizumab",
    RECIST="PR",
    treatment_response="responder",
)

# %%
adata.obs["sample_id"] = adata.obs["file"].str.split("_").str[1]
adata.obs["SRA_sample_accession"] = adata.obs["big_key"]
adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]

# define the dictionary mapping
mapping = {
    "naive": "naive",
    "FOLFOX-bevacizumab": "treated",
}
adata.obs["treatment_status_before_resection"] = adata.obs["treatment"].map(mapping)

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["SRA_sample_accession"]
    + "-"
    + adata.obs_names.str.split("-").str[0]
)

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]]

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
adata

# %%
adata.obs["sample_id"].value_counts()

# %%
# Save unfiltered raw counts
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
