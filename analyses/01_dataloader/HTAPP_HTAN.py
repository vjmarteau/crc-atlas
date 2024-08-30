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
# # Dataloader: HTAPP_HTAN
#
# Is this a subset from Pelka 2021 ?! on HTAN with raw counts -> see: https://www.protocols.io/view/htapp-dissociation-of-human-primary-colorectal-can-batcieiw
#
# Would be nice to get the patient/sample mapping between HTAN Ids and SCP Ids

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
from tqdm.contrib.concurrent import process_map

# %% tags=["parameters"]
artifact_dir = nxfvars.get("artifact_dir", "/data/scratch/marteau/downloads/crc-atlas/")
cpus = nxfvars.get("cpus", 6)
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")

# %%
gtf = f"{annotation}/gencode.v28_lift37_gene_annotation_table.csv"
d10_1016_j_cell_2021_08_003 = f"{databases_path}/d10_1016_j_cell_2021_08_003"
syn24181445 = f"{d10_1016_j_cell_2021_08_003}/synapse/syn24181445"

# %%
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

# %%
# Read the YAML file and load it into a dictionary
with open(reference_meta, "r") as f:
    ref_meta_dict = yaml.load(f, Loader=yaml.Loader)

# %% [markdown]
# ## 1. Load raw adata counts from Synapse and append HTAN metadata

# %%
meta = pd.read_csv(
    f"{d10_1016_j_cell_2021_08_003}/metadata/syn24181445_sample_meta.csv"
)
meta = meta.dropna(thresh=5)

# %%
# Set display options to show all rows and columns
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


# %%
def _load_mtx(file_meta):
    FileName = f"{syn24181445}/{file_meta['Filename']}"
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
    adata.X = csr_matrix(adata.X)
    return adata


# %%
adatas = process_map(_load_mtx, [r for _, r in meta.iterrows()], max_workers=cpus)
adata = anndata.concat(adatas, index_unique="-", join="outer", fill_value=0)

# %% [markdown]
# ### 1. Gene annotations

# %%
# get id mappings from one of the samples
features = pd.read_csv(f"{syn24181445}/HTAPP-556-SMP-5761_none_channel1_features.tsv.gz",
            delimiter="\t",
            header=None,
            names=["ensembl", "symbol"],
            index_col=0)

# map back gene symbols to .var
gene_symbols = features["ensembl"].to_dict()

adata.var["ensembl"] = adata.var_names
adata.var["symbol"] = (
    adata.var["ensembl"]
    .map(gene_symbols)
    .fillna(value=adata.var["ensembl"])
)
adata.var_names = adata.var["ensembl"]
adata.var_names.name = None

# %% [markdown]
# ### 2. sample/patient metadata

# %%
patient_meta_meta = pd.read_csv(
    f"{d10_1016_j_cell_2021_08_003}/metadata/syn24181445_patient_meta.csv"
)

# %%
adata.obs = adata.obs.assign(
    study_id="Pelka_2021_syn24181445", #?
    dataset="HTAPP_HTAN",
    medical_condition="colorectal cancer",
    platform = "10x", #?
)

adata.obs["synapse_sample_accession"] = adata.obs["SynID"]
adata.obs["sample_id"] = adata.obs["HTAN Biospecimen ID"]
adata.obs["patient_id"] = adata.obs["HTAN Participant ID"]

# see: https://seer.cancer.gov/tools/solidtumor/Colon_STM.pdf and https://training.seer.cancer.gov/coding/structure/nos.html for meaning of "NOS"
# define the dictionary mapping
mapping = {
    "8140/3": "adenocarcinoma",
    "8510/3": "medullary carcinoma",
    "8480/3": "mucinous adenocarcinoma",
}
adata.obs["histological_type"] = adata.obs["Histologic Morphology Code"].map(
    mapping
)
# adata.obs["Primary Diagnosis"] same info?

adata.obs["tissue_cell_state"] = adata.obs["Preservation Method"].map(
    {"Fresh dissociated": "fresh"}
)

adata.obs["sample_type"] = adata.obs["Tumor Tissue Type"].map({"Primary": "tumor"})
adata.obs["sample_tissue"] = adata.obs["Tissue or Organ of Origin"].map(
    {"Colon NOS": "colon"}
)

adata.obs["age"] = adata.obs["Age at Diagnosis (years)"].fillna(np.nan).astype("Int64")
adata.obs["sex"] = adata.obs["Gender"]

mapping = {
    "Cecum": "cecum",
    "Ascending colon": "ascending colon",
    "Rectosigmoid junction": "rectosigmoid junction",
    "Sigmoid colon": "sigmoid colon",
    "Transverse colon": "transverse colon",
}
adata.obs["anatomic_location"] = adata.obs["Site of Resection or Biopsy"].map(
    mapping
)

mapping = {
    "Cecum": "proximal colon",
    "Ascending colon": "proximal colon",
    "Rectosigmoid junction": "distal colon",
    "Sigmoid colon": "distal colon",
    "Transverse colon": "proximal colon",
}
adata.obs["anatomic_region"] = adata.obs[
    "Site of Resection or Biopsy"
].map(mapping)

adata.obs["tumor_grade"] = adata.obs["Tumor Grade"]

adata.obs["tumor_stage_TNM_T"] = adata.obs["AJCC Pathologic T"]
adata.obs["tumor_stage_TNM_N"] = adata.obs["AJCC Pathologic N"]
adata.obs["tumor_stage_TNM_M"] = adata.obs["AJCC Pathologic M"]

adata.obs["tumor_stage_TNM"] = ah.pp.tnm_tumor_stage(adata.obs, "patient_id")

adata.obs["ethnicity"] = adata.obs["Race"]

# %%
adata.obs_names = adata.obs["dataset"] + "-" + adata.obs["SynID"] + "-" + adata.obs_names.str.split("-").str[0]
adata.obs_names_make_unique()

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
# Add back potentially interesting metadata that does not fit the reference meta
ref_meta_cols += [
    "Progression or Recurrence",
    "Last Known Disease Status",
    "Days to Last Follow up",
    "Days to Last Known Disease Status",
    "Vital Status",
    "Cause of Death",
    "Year of Death",
]

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
adata

# %%
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")

# %%
# Have been unable to map the 22 HTAN samples back to the original Pelka dataset sample ids
# adatas["Pelka_2021_HTAN"].obs["tmp_id"] = adatas["Pelka_2021_HTAN"].obs["sex"].astype(str) + "-" + adatas["Pelka_2021_HTAN"].obs["age"].astype(str) + "-" + adatas["Pelka_2021_HTAN"].obs["anatomic_location"].str.split(" ").str[0] + "-" + adatas["Pelka_2021_HTAN"].obs_names.str.rsplit("_", n=1).str[1]
# adatas["Pelka_2021_10Xv2"].obs["tmp_id"] = adatas["Pelka_2021_10Xv2"].obs["sex"].astype(str) + "-" + adatas["Pelka_2021_10Xv2"].obs["age"].astype(str) + "-" + adatas["Pelka_2021_10Xv2"].obs["anatomic_location"].str.split(" ").str[0] + "-" + adatas["Pelka_2021_10Xv2"].obs_names.str.rsplit("_", n=1).str[1]
# adatas["Pelka_2021_10Xv2"][adatas["Pelka_2021_10Xv2"].obs["tmp_id"].isin(adatas["Pelka_2021_HTAN"].obs["tmp_id"])].obs.groupby("sample_id", observed=False).first()

# Tried using "sex"+"age"+"location"+"barcodes" -> only 3 samples that kind of work:

#"C109_T_0_1_0_c1_v2": "female-60-ascending",
#"C135_N_1_1_0_c1_v2": "male-90-cecum",
#"C143_N_1_1_0_c1_v2": "female-55-cecum"

# Also no luck with sc.pp.filter_cells and comparing adata.obs["sample_id"].value_counts()
# FileNames indicate its a mix of enriched/naive samples
