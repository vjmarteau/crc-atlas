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
# # Dataloader: Harmon_2023_Nat_Cancer

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
d10_1038_s43018_023_00589_w = f"{databases_path}/d10_1038_s43018_023_00589_w"

# %%
# Read the YAML file and load it into a dictionary
with open(reference_meta, "r") as f:
    ref_meta_dict = yaml.load(f, Loader=yaml.Loader)

# %% [markdown]
# ## 1. Load raw adata counts

# %%
geofetch = pd.read_csv(
    f"{d10_1038_s43018_023_00589_w}/GSE210040/GSE210040_PEP/GSE210040_PEP_raw.csv"
).dropna(axis=1, how="all")

fetchngs = pd.read_csv(
    f"{d10_1038_s43018_023_00589_w}/fetchngs/PRJNA863562/samplesheet/samplesheet.csv"
).dropna(axis=1, how="all").drop_duplicates("sample_alias")

meta = pd.merge(
    geofetch,
    fetchngs,
    how="left",
    left_on=["gsm_id"],
    right_on=["sample_alias"],
    validate="m:1",
)
meta["file"] = meta["sample_supplementary_file_1"].str.rsplit("/", n=1).str[1]
meta = meta.loc[meta["sample_source_name_ch1"].isin(["Colorectal tumor", "Healthy colon"])].reset_index(drop=True)

# %%
def load_counts(sample_meta):

    adata = sc.read_10x_h5(
        f"{d10_1038_s43018_023_00589_w}/GSE210040/GSE210040_RAW/{sample_meta['file']}"
    )
    adata.X = csr_matrix(adata.X.astype(int))
    adata.var["symbol"] = adata.var_names
    adata.var_names = adata.var["gene_ids"]
    adata.var_names.name = None
    adata.obs = adata.obs.assign(**sample_meta)
    
    return adata

# %%
adatas = process_map(
    load_counts,
    [r for _, r in meta.iterrows()],
    max_workers=cpus,
)

# %%
adata = anndata.concat(adatas, join="outer")

# %%
# Unfortunately don't have access to the paper
adata.obs = adata.obs.assign(
    study_id="Harmon_2023_Nat_Cancer",
    study_doi="10.1038/s43018-023-00589-w",
    study_pmid="37474835",
    tissue_processing_lab="Lydia Lynch lab",
    dataset="Harmon_2023_CD56Pos",
    medical_condition="colorectal cancer",
    #treatment_status_before_resection="",
    sample_tissue="colon",
    tissue_cell_state="frozen",
    enrichment_cell_types="CD56+", # Lymphocytes
    platform="10x 3' v3",
    cellranger_version="2.1.1",
    #reference_genome="",
    matrix_type="raw counts",
# Paper reporting summary: Patients from St. Vincent's ... coloctomy ... eligibale for this study
    hospital_location="St. Vincent's University Hospital, Dublin",
    country="Ireland",
    NCBI_BioProject_accession="PRJNA863562",
)

# %%
# P3-6 -> I think this might be a mix of different patients?
adata.obs["patient_id"] = "P" + adata.obs["sample_name"].str.rsplit("_", n=1).str[1]

# define the dictionary mapping
mapping = {
    "Colorectal tumor": "tumor",
    "Healthy colon": "normal",
}
adata.obs["sample_type"] = adata.obs["sample_source_name_ch1"].map(mapping)

adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]
adata.obs["SRA_sample_accession"] = adata.obs["big_key"]
adata.obs["sample_id"] = adata.obs["GEO_sample_accession"]

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
    + adata.obs["GEO_sample_accession"]
    + "-"
    + adata.obs_names.str.split("-").str[0]
)

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
adata.obs["sample_id"].value_counts()

# %%
# Save unfiltered raw counts
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
