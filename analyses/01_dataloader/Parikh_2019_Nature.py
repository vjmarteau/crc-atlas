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
# # Dataloader: Parikh_2019_Nature

# %%
import os

import anndata
import fast_matrix_market as fmm
import numpy as np
import pandas as pd
import sc_atlas_helpers as ah
import scanpy as sc
import scipy.sparse
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

# %%
d10_1038_s41586_019_0992_y = f"{databases_path}/d10_1038_s41586_019_0992_y"
gtf = f"{annotation}/ensembl.v76_gene_annotation_table.csv"
GSE116222 = f"{d10_1038_s41586_019_0992_y}/GSE116222"

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
adata = sc.read_text(f"{GSE116222}/GSE116222_Expression_matrix.txt.gz").transpose()
adata.X = csr_matrix(adata.X)
adata.obs["sample_id"] = adata.obs_names.str.split("-").str[1]

# %%
ah.pp.undo_log_norm(adata)

# %% [markdown]
# ### 2. Load UCSC ensemblToGeneName table for gene mapping
# Dowloaded from [UCSC hgTables](https://genome.ucsc.edu/cgi-bin/hgTables). See also: https://www.biostars.org/p/92939/.
# Roughly 2000 genes do not have an ensembl id. I don't think it is a good idea to map these using the gene symbols to gencode/ensembl gtf files - which is possible. This UCSC GRCh38/hg38 assembly is from Dec 2013 and I don't think that the gene coordinates match any more. Will only use the official UCSC to ensembl mapping and drop the rest!

# %%
UCSC_ensemblToGeneName = pd.read_csv(f"{d10_1038_s41586_019_0992_y}/UCSC_hg38_ensemblToGeneName.txt", sep='\t')
UCSC_ensemblToGeneName=UCSC_ensemblToGeneName.drop_duplicates(subset=['hg38.wgEncodeGencodeAttrsV20.geneId'])
gene_ids = UCSC_ensemblToGeneName.set_index("hg38.wgEncodeGencodeBasicV20.name2")["hg38.wgEncodeGencodeAttrsV20.geneId"].to_dict()

# %%
adata.var = adata.var.rename_axis("symbol").reset_index()
adata.var["ensembl"] = (
    adata.var["symbol"].map(gene_ids).fillna(value=adata.var["symbol"])
)
adata.var_names = adata.var["ensembl"].apply(ah.pp.remove_gene_version)
adata.var_names.name = None

# %%
unmapped_genes = ah.pp.find_unmapped_genes(adata)
len(unmapped_genes)

# %%
# gene_annotation_table.csv for ensembl.v76:
# zcat Homo_sapiens.GRCh37.76.gtf.gz | awk 'BEGIN{FS="\t";OFS=","}$3=="gene"{split($9,a,";");split(a[1],gene_id,"\"");split(a[2],gene_name,"\""); print gene_id[2],gene_name[2]}' | sed '1i\gene_id,gene_name' > ensembl.v76_gene_annotation_table.csv

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ### 2a. Compile sample/patient metadata from original study supplemetary tables
# Not really available ...

# %% [markdown]
# ### 2b. Get SRA and GEO accession numbers

# %%
fetchngs = pd.read_csv(
    f"{d10_1038_s41586_019_0992_y}/fetchngs/PRJNA477814/samplesheet/samplesheet.csv"
).dropna(axis=1, how="all")

# %%
geofetch = pd.read_csv(f"{GSE116222}/GSE116222_PEP/GSE116222_PEP_raw.csv").dropna(
    axis=1, how="all"
)

geofetch["sample_id"] = geofetch["sample_name"].str.split("_").str[0].str.upper()

# %%
accessions = (
    pd.merge(
        fetchngs,
        geofetch,
        how="left",
        left_on=["experiment_accession"],
        right_on=["srx"],
        validate="m:1",
    )
    .drop(["sample_description_y", "sample_title_y"], axis=1)
    .rename(
        columns={
            "sample_description_x": "sample_description",
            "sample_title_x": "sample_title",
        }
    )
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
sample_meta = (
    adata.obs[
        adata.obs.columns[adata.obs.nunique() <= adata.obs["sample_id"].nunique()]
    ]
    .groupby("sample_id")
    .first()
    .reset_index()
)

# %%
sample_meta.to_csv(f"{artifact_dir}/Parikh_2019-original_sample_meta.csv", index=False)

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="Parikh_2019_Nature",
    study_doi="10.1038/s41586-019-0992-y",
    study_pmid="30814735",
    tissue_processing_lab="Alison Simmons lab",
    dataset="Parikh_2019",
    medical_condition="healthy",
    NCBI_BioProject_accession="PRJNA477814",
    sample_tissue="colon",
    tissue_cell_state="fresh",
    platform="10x 3' v2",
    matrix_type="log norm",
    reference_genome="UCSC GRCh38/hg38",
    cellranger_version="2.1.1",
    hospital_location="John Radcliffe Hospital, University of Oxford",
    country="United Kingdom",
)

adata.obs["SRA_sample_accession"] = adata.obs["secondary_sample_accession"]
adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]

# 3 repeat experiments (A, B, C) with biopsy samples taken from colonic biopsies collected from healthy patients (Healthy) and those with UC inflammation from an inflamed area of colon and adjacent non-inflamed area.
# All biopsies underwent an epithelial cell isolation protocol by digestion and then were loaded onto the 10x single cell RNA sequencing platform with subsequent GEM generation
mapping = {
    "A1": "A_healthy",
    "A2": "A_UC",
    "A3": "A_UC",
    "B1": "B_healthy",
    "B2": "B_UC",
    "B3": "B_UC",
    "C1": "C_healthy",
    "C2": "C_UC",
    "C3": "C_UC",
}
adata.obs["patient_id"] = adata.obs["sample_id"].map(mapping)

# %%
mapping = {
    "healthy control": "normal",
    "Ulcerative colitis": "Ulcerative colitis",
}
adata.obs["condition"] = adata.obs["subject_status"].map(mapping)

mapping = {
    "colonic biopsy": "normal",
    "adjacent non-inflamed area": "normal",
    "inflamed area of colon": "inflamed",
}
adata.obs["sample_type"] = adata.obs["tissue"].map(mapping)

adata.obs["enrichment_cell_types"] = adata.obs["cell_type"].map(
    {"epithelial cells": "epithelial"}
)

mapping = {
    "A_healthy": "median: 50 [47-74]",
    "B_healthy": "median: 50 [47-74]",
    "C_healthy": "median: 50 [47-74]",
    "A_UC": "median: 55 [36-80]",
    "B_UC": "median: 55 [36-80]",
    "C_UC": "median: 55 [36-80]",
}
adata.obs["age"] = adata.obs["patient_id"].map(mapping)

mapping = {
    "A_healthy": "n (%) male: 1 (33%)",
    "B_healthy": "n (%) male: 1 (33%)",
    "C_healthy": "n (%) male: 1 (33%)",
    "A_UC": "n (%) male: 2 (66%)",
    "B_UC": "n (%) male: 2 (66%)",
    "C_UC": "n (%) male: 2 (66%)",
}
adata.obs["sex"] = adata.obs["patient_id"].map(mapping)

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
# Add back potentially interesting metadata that does not fit the reference meta
ref_meta_cols += [
    "condition",
]

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["GEO_sample_accession"]
    + "-"
    + adata.obs_names.str.split("-").str[0]
)

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
adata

# %%
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
