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
# # Dataloader: He_2020_Genome_Biol

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
cpus = nxfvars.get("cpus", 6)
d10_1186_s13059_020_02210_0 = f"{databases_path}/d10_1186_s13059_020_02210_0"

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
fetchngs = pd.read_csv(
    f"{d10_1186_s13059_020_02210_0}/fetchngs/PRJNA670909/samplesheet/samplesheet.csv"
).dropna(axis=1, how="all")

geofetch = pd.read_csv(
    f"{d10_1186_s13059_020_02210_0}/GSE159929/GSE159929_PEP/GSE159929_PEP_raw.csv"
).dropna(axis=1, how="all")

samplesheet = pd.read_csv(f"{d10_1186_s13059_020_02210_0}/samplesheet.csv").drop(
    columns=["fastq_1", "fastq_2"]
)

# %%
meta = pd.merge(
    fetchngs,
    geofetch.drop(columns=["sample_title"]),
    how="left",
    left_on=["sample"],
    right_on=["srx"],
    validate="m:1",
).rename(columns={"run_accession": "SRA_sample_accession"})

# %%
meta.to_csv(
    f"{artifact_dir}/He_2020-original_sample_meta.csv",
    index=False,
)

# %%
meta = meta.loc[meta["SRA_sample_accession"].isin(samplesheet["sample"])].reset_index(
    drop=True
)


# %%
def load_counts(sample_meta, databases_path, mat_type="filtered"):

    file_path = f"{databases_path}/10_nfcore_scrnaseq/cellranger/count/{sample_meta['SRA_sample_accession']}/outs/{mat_type}_feature_bc_matrix.h5"
    adata = sc.read_10x_h5(file_path)
    adata.obs = adata.obs.assign(**sample_meta)
    adata.var_names_make_unique()
    adata.X = csr_matrix(adata.X.astype(int))
    adata.obs["original_obs_names"] = adata.obs_names
    adata.obs_names = (
        adata.obs["SRA_sample_accession"] + "-" + adata.obs_names.str.replace("-1", "")
    )
    assert adata.obs_names.is_unique
    return adata


# %%
adatas = process_map(
    load_counts,
    [r for _, r in meta.iterrows()],
    itertools.repeat(d10_1186_s13059_020_02210_0),
    itertools.repeat("raw"),
    max_workers=cpus,
)

# %%
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

# %%
adata.obs = adata.obs.assign(
    study_id="He_2020_Genome_Biol",
    study_doi="10.1186/s13059-020-02210-0",
    study_pmid="33287869",
    tissue_processing_lab="Jin-Xin Bei lab",
    dataset="He_2020",
    medical_condition="healthy",
    NCBI_BioProject_accession="PRJNA670909",
    enrichment_cell_types="naive",
    matrix_type="raw counts",
    # Paper: mRNA transcripts from each sample were ligated with barcoded indexes at 5′-end and reverse transcribed into cDNA, using GemCode technology (10x Genomics, USA).
    platform="10x 5' v1",
    cellranger_version="7.1.0",
    reference_genome="gencode.v44",
    hospital_location="Sun Yat-Sen Memorial Hospital, Sun Yat-Sen University, Guangzhou, Guangdong",
    country="China",
    # Paper methods: All tissue samples were kept on ice and delivered to the laboratory within 40 min for further processing.
    tissue_cell_state="fresh",
    # Paper methods: An adult male donor who died of a traumatic brain injury
    sample_type="normal",
    treatment_status_before_resection="naive",
    patient_id="donor_1",
)

# %%
adata.obs["sample_id"] = adata.obs["SRA_sample_accession"]
adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]

adata.obs["age"] = adata.obs["age"].str.split(" ").str[0].astype("Int64")

# we collected tissues from 15 organs in sequence, including the blood, bone marrow, liver, common bile duct, lymph node (hilar and mesenteric),
# spleen, heart (apical), urinary bladder, trachea, esophagus, stomach, small intestine, rectum, skin, and muscle (thigh).
mapping = {
    "Blood_cDNA": "blood",
    "Rectum_cDNA": "colon",
    "Small.intestine_cDNA": "small intestine",
    "Lymph.node_cDNA": "lymph node",
}
adata.obs["sample_tissue"] = adata.obs["sample_title"].map(mapping)

mapping = {
    "Blood_cDNA": "blood",
    "Rectum_cDNA": "rectum",
    "Small.intestine_cDNA": "small intestine",
    "Lymph.node_cDNA": "lymph node",
}
adata.obs["anatomic_location"] = adata.obs["sample_title"].map(mapping)

mapping = {
    "Blood_cDNA": "blood",
    "Rectum_cDNA": "distal colon",
    "Small.intestine_cDNA": "small intestine",
    "Lymph.node_cDNA": "lymph node",
}
adata.obs["anatomic_region"] = adata.obs["sample_title"].map(mapping)

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
adata.obs["sample_id"].value_counts()

# %%
adata

# %%
# Save unfiltered raw counts
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
