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
# # Dataloader: deVries_2023_Nature

# %%
import anndata
import fast_matrix_market as fmm
import numpy as np
import pandas as pd
import sc_atlas_helpers as ah
import scanpy as sc
import yaml
from nxfvars import nxfvars
from scipy.sparse import csr_matrix
from tqdm.contrib.concurrent import process_map

# %%
artifact_dir = nxfvars.get("artifact_dir", "/data/scratch/marteau/downloads/crc-atlas/")
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")
cpus = nxfvars.get("cpus", 2)

# %%
d10_1038_s41586_022_05593_1 = f"{databases_path}/d10_1038_s41586_022_05593_1"

# %%
# Read the YAML file and load it into a dictionary
with open(reference_meta, "r") as f:
    ref_meta_dict = yaml.load(f, Loader=yaml.Loader)

# %% [markdown]
# ## 1. Load raw adata counts

# %%
geofetch = pd.read_csv(
    f"{d10_1038_s41586_022_05593_1}/GSE216534/GSE216534_PEP/GSE216534_PEP_raw.csv"
).dropna(axis=1, how="all")

# %%

adata = sc.read_10x_h5(
    f"{d10_1038_s41586_022_05593_1}/GSE216534/GSE216534_RAW//GSE216534_filtered_feature_bc_matrix.h5"
)
adata.X = csr_matrix(adata.X.astype(int))
adata.var["symbol"] = adata.var_names
adata.var_names = adata.var["gene_ids"]
adata.var_names.name = None

# %%
cell_meta = pd.read_csv(
    f"{d10_1038_s41586_022_05593_1}/GSE216534/GSE216534_RAW//GSE216534_CRC_GDT_metadata.csv"
)

adata.obs["cell_id"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    cell_meta,
    how="left",
    on=["cell_id"],
    validate="m:1",
).set_index("cell_id")
adata.obs_names.name = None

# %%
#scRNA-seq was performed on sorted γδ T cells from colon cancers (MMR-d) of five patients from the LUMC in the presence of hashtag oligos (HTOs) 
adata.obs["sample_id"] = adata.obs["MULTI_id"]
# Maybe possible to assign LUMC cohort patient ids from supp figure3 ? (CRC159, CRC134, CRC154, CRC167, CRC96)
adata.obs["patient_id"] = adata.obs["sample_id"]

adata = adata[adata.obs["sample_id"].isin(["HTO1", "HTO6", "HTO7", "HTO8", "HTO9"])].copy()

# %%
adata.obs = adata.obs.assign(
    study_id="deVries_2023_Nature",
    study_doi="10.1038/s41586-022-05593-1",
    study_pmid="36631610",
    tissue_processing_lab="Emile Voest lab",
    dataset="deVries_2023_LUMC",
    medical_condition="colorectal cancer",
# We performed single-cell RNA-sequencing on γδ T cells isolated from five treatment-naive MMR-deficient colon cancers ...
    treatment_status_before_resection="naive",
    sample_tissue="colon",
    sample_type="tumor",
    tissue_cell_state="frozen",
    enrichment_cell_types="CD3+/TCRgd+",
    mismatch_repair_deficiency_status="dMMR",
    microsatellite_status="MSI",
    platform="10x 5' v1",
    cellranger_version="3.1.0",
    #reference_genome="",
    matrix_type="raw counts",
# Paper reporting summary: Patients from St. Vincent's ... coloctomy ... eligibale for this study
    hospital_location="Leiden University Medical Center",
    country="Netherlands",
    NCBI_BioProject_accession="PRJNA894171",
    GEO_sample_accession="GSM6675892",
    SRA_sample_accession="SRX18012614",
)

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
    + adata.obs["sample_id"]
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
adata.obs["sample_id"].value_counts()

# %%
# Save unfiltered raw counts
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
