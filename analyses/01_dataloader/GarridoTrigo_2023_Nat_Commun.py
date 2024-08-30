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
# # Dataloader: GarridoTrigo_2023_Nat_Commun

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
d10_1038_s41467_023_40156_6 = f"{databases_path}/d10_1038_s41467_023_40156_6"

# %%
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

# %%
# Read the YAML file and load it into a dictionary
with open(reference_meta, "r") as f:
    ref_meta_dict = yaml.load(f, Loader=yaml.Loader)

# %% [markdown] tags=[]
# ## 1. Load raw adata counts

# %%
SraRunTable = pd.read_csv(f"{d10_1038_s41467_023_40156_6}/fetchngs/PRJNA886695/samplesheet/samplesheet.csv")#.drop_duplicates(subset='sample').reset_index(drop=True)

geofetch = pd.read_csv(
    f"{d10_1038_s41467_023_40156_6}/GSE214695/GSE214695_PEP/GSE214695_PEP_raw.csv"
).dropna(axis=1, how="all")

meta = pd.merge(
    SraRunTable,
    geofetch,
    how="left",
    left_on=["sample"],
    right_on=["big_key"],
    validate="m:1",
)

samplesheet = pd.read_csv(f"{d10_1038_s41467_023_40156_6}/samplesheet.csv").drop(columns=['fastq_1', 'fastq_2'])
patient_meta = pd.read_csv(f"{d10_1038_s41467_023_40156_6}/metadata/tableS1.csv")

meta2 = pd.merge(
    samplesheet,
    patient_meta,
    how="left",
    left_on=["type"],
    right_on=["Sample"],
    validate="m:1",
).rename(columns={"sample": "SRA_sample_accession"})

metadata = pd.merge(
    meta2,
    meta,
    how="left",
    left_on=["SRA_sample_accession"],
    right_on=["run_accession"],
    validate="m:1",
).sort_values(by="SRA_sample_accession")


# %%
def load_counts(sample_meta, databases_path, mat_type="filtered"):

    file_path = f"{databases_path}/10_nfcore_scrnaseq/cellranger/count/{sample_meta['SRA_sample_accession']}/outs/{mat_type}_feature_bc_matrix.h5"
    adata = sc.read_10x_h5(file_path)
    adata.obs = adata.obs.assign(**sample_meta)
    adata.var_names_make_unique()
    adata.X = csr_matrix(adata.X.astype(int))
    adata.obs_names = (
        adata.obs["SRA_sample_accession"] + "_" + adata.obs_names.str.replace("-1", "")
    )
    assert adata.obs_names.is_unique
    return adata


# %%
adatas = process_map(
    load_counts,
    [r for _, r in metadata.iterrows()],
    itertools.repeat(d10_1038_s41467_023_40156_6),
    itertools.repeat("raw"),
    max_workers=cpus,
)

adata = anndata.concat(adatas, join="outer")

# %%
sample_meta = (
    adata.obs[
        adata.obs.columns[adata.obs.nunique() <= adata.obs["Sample"].nunique()]
    ]
    .groupby("Sample")
    .first()
    .reset_index()
)

# %%
sample_meta.to_csv(f"{artifact_dir}/GarridoTrigo_2023-original_sample_meta.csv", index=False)

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
# Paper methods: Healthy controls were individuals undergoing endoscopy for colorectal cancer screening as standard of care and presenting no signs of dysplasia or polyps at the time of endoscopy

# %%
adata.obs = adata.obs.assign(
    study_id="GarridoTrigo_2023_Nat_Commun",
    study_doi="10.1038/s41467-023-40156-6",
    study_pmid="37495570",
    tissue_processing_lab="Azucena Salas lab",
    dataset="GarridoTrigo_2023",
    medical_condition="healthy",
    NCBI_BioProject_accession="PRJNA886695",
    enrichment_cell_types="naive",
    matrix_type="raw counts",
    platform="10x 3' v3",
    cellranger_version="7.1.0",
    reference_genome="gencode.v44",
    hospital_location="Hospital Clinic Barcelon, Institut d’Investigacions Biomèdiques August Pi i Sunyer (IDIBAPS)",
    country="Spain",
    sample_tissue="colon",
    tissue_cell_state="fresh",
)

# %%
adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]
adata.obs["sample_id"] = adata.obs["Sample"]
adata.obs["patient_id"] = adata.obs["Sample"]

mapping = {
    "Female": "female",
    "Male": "male",
}
adata.obs["sex"] = adata.obs["Sex"].map(mapping)
adata.obs["age"] = adata.obs["Age"].fillna(np.nan).astype("Int64")

mapping = {
    "healthy control": "normal",
    "ulcerative colitis": "Ulcerative colitis",
    "Crohn’s disease": "Crohn’s disease",
}
adata.obs["condition"] = adata.obs["disease_state"].map(mapping)

mapping = {
    "normal": "normal",
    "Crohn’s disease": "inflamed",
    "Ulcerative colitis": "inflamed",
}
adata.obs["sample_type"] = adata.obs["condition"].map(mapping)

mapping = {
    'Sigmoid colon': "sigmoid colon",
    'Descending colon': "descending colon",
    'Ascending colon': "ascending colon",
    'Rectum': "rectum",
    'Transvers colon': "transverse colon",
}
adata.obs["anatomic_location"] = adata.obs["Biopsy/surgical resection area"].map(mapping)

mapping = {
    'Sigmoid colon': "distal colon",
    'Descending colon': "distal colon",
    'Ascending colon': "proximal colon",
    'Rectum': "distal colon",
    'Transvers colon': "proximal colon",
}
adata.obs["anatomic_region"] = adata.obs["Biopsy/surgical resection area"].map(mapping)

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
    + adata.obs["SRA_sample_accession"]
    + "-"
    + adata.obs_names.str.split("_").str[1]
)
adata.obs_names_make_unique()

# %%
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
# Save unfiltered raw counts
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
