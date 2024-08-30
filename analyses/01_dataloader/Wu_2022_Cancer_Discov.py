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
# # Dataloader: Wu_2022_Cancer_Discov

# %%
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

# %% tags=["parameters"]
artifact_dir = nxfvars.get("artifact_dir", "/data/scratch/marteau/downloads/crc-atlas/")
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")
cpus = nxfvars.get("cpus", 6)
d10_1158_2159_8290_CD_21_0316 = f"{databases_path}/d10_1158_2159_8290_CD_21_0316"

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
mat = fmm.mmread(f"{d10_1158_2159_8290_CD_21_0316}/matrix.mtx.gz")

features = pd.read_csv(f"{d10_1158_2159_8290_CD_21_0316}/genes.tsv.gz", delimiter="\t", header=None)
features = features.rename(columns={0: "symbol"})

barcodes = pd.read_csv(f"{d10_1158_2159_8290_CD_21_0316}/barcodes.tsv.gz", delimiter="\t", header=None)
barcodes = barcodes.rename(columns={0: "barcodes"})

adata = sc.AnnData(X=mat.T)
adata.X = csr_matrix(adata.X)

adata.var = pd.DataFrame(index=features["symbol"])
adata.obs = pd.DataFrame(index=barcodes["barcodes"])

# %%
adata.obs["barcodes"] = adata.obs_names
adata.obs_names.name=None

# %%
cell_meta = pd.read_csv(f"{d10_1158_2159_8290_CD_21_0316}/cell_meta.tsv", delimiter="\t")

adata.obs = pd.merge(
    adata.obs,
    cell_meta,
    how="left",
    left_on=["barcodes"],
    right_on=["old_obs_names"],
    validate="m:1",
).set_index("barcodes")
adata.obs_names.name=None

# %%
patient_meta = pd.read_csv(f"{d10_1158_2159_8290_CD_21_0316}/metadata/patient_meta.csv")

adata.obs = adata.obs.reset_index(names="obs_names")
adata.obs = pd.merge(
    adata.obs,
    patient_meta,
    how="left",
    left_on=["patient"],
    right_on=["Patient number"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
# Get original sample/patient meta and remove cell level meta
sample_meta = (
    adata.obs[
        adata.obs.columns[adata.obs.nunique() <= adata.obs["patient_tissue"].nunique()]
    ]
    .groupby("patient_tissue")
    .first()
    .reset_index()
)

# %%
sample_meta.to_csv(f"{artifact_dir}/Wu_2022-original_sample_meta.csv", index=False)

# %% [markdown]
# ### Gene annotations

# %%
# load gtf for gene mapping
gtf = pd.read_csv(f"{annotation}/ensembl.v98_gene_annotation_table.csv")
gtf = ah.pp.append_duplicate_suffix(df=gtf, column="gene_name", sep=".")
gene_ids = gtf.set_index("gene_name")["gene_id"].to_dict()

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

# %% [markdown]
# ### sample/patient metadata

# %%
adata.obs = adata.obs.assign(
    study_id="Wu_2022_Cancer_Discov",
    study_doi="10.1158/2159-8290.CD-21-0316",
    study_pmid="34417225",
    tissue_processing_lab="Qiang Gao lab",
    dataset="Wu_2022_CD45Pos",
    medical_condition="colorectal cancer",
    tissue_cell_state="fresh",
    enrichment_cell_types="CD45+",
    # Paper Results: All the tumors were microsatellite-stable
    microsatellite_status="MSS",
    hospital_location="Zhongshan Hospital, Fudan University",
    country="China",
    platform="10x 3' v3",
    matrix_type="raw counts",
    cellranger_version="3.0.0",
    reference_genome="ensembl.v98",
)

# %%
adata.obs["original_obs_names"] = adata.obs["new_obs_names"]
adata.obs["cell_type_study"] = adata.obs["sub_cell_type"]
adata.obs["cell_type_coarse_study"] = adata.obs["main_cell_type"]

adata.obs["patient_id"] = adata.obs["patient"]
adata.obs["sample_id"] = adata.obs["patient_tissue"]

mapping = {
    "Female": "female",
    "Male": "male",
}
adata.obs["sex"] = adata.obs["Sex"].map(mapping)
adata.obs["age"] = adata.obs["Age"].fillna(np.nan).astype("Int64")

# %%
# According to table S1 P19 has no matched normal colon -> Colon_T= tumor; Colon_P = normal
mapping = {
    "Colon_T": "tumor",
    "Colon_P": "normal",
    "Liver_T": "metastasis",
    "Liver_T1": "metastasis",
    "Liver_T2": "metastasis",
    "Liver_P": "normal",
    "LN": "lymph node",
    "PBMC": "blood",
}
adata.obs["sample_type"] = adata.obs["tissue"].map(mapping)

mapping = {
    "Colon_T": "colon",
    "Colon_P": "colon",
    "Liver_T": "liver",
    "Liver_T1": "liver",
    "Liver_T2": "liver",
    "Liver_P": "liver",
    "LN": "lymph node",
    "PBMC": "blood",
}
adata.obs["sample_tissue"] = adata.obs["tissue"].map(mapping)

# %%
adata.obs["tumor_stage_TNM_T"] = (
    adata.obs["TNM"]
    .astype(str)
    .apply(
        lambda x: re.search(r"(T[0-4][a-c]?)", x).group(1)
        if re.search(r"(T[0-4][a-c]?)", x)
        else None
    )
)
adata.obs["tumor_stage_TNM_N"] = (
    adata.obs["TNM"]
    .astype(str)
    .apply(
        lambda x: re.search(r"(N[0-4][a-c]?)", x).group(1)
        if re.search(r"(N[0-4][a-c]?)", x)
        else None
    )
)
adata.obs["tumor_stage_TNM_M"] = (
    adata.obs["TNM"]
    .astype(str)
    .apply(
        lambda x: re.search(r"((M[0-1]|Mx)[a-c]?)", x).group(1)
        if re.search(r"((M[0-1]|Mx)[a-c]?)", x)
        else None
    )
)

# Matches "Stage AJCC" column from original meta
adata.obs["tumor_stage_TNM"] = ah.pp.tnm_tumor_stage(adata.obs, "patient_id")

# %%
# Paper results: treatment naive patients + neoadjuvant chemotherapy
# Table S1 specifies patients in "Neoadjuvant chemotherapy" as "N"!
mapping = {
    "pretreated": "naive",
    "treated-PR": "treated",
    "treated-PD/SD": "treated",
}
adata.obs["treatment_status_before_resection"] = adata.obs["chemo"].map(mapping)

mapping = {
    "Y (PR/ FOLFOX)": "FOLFOX",
    "N": "naive",
    "Y (PR/ XELOX)": "FOLFOX",
    "nan": "naive",
    "Y (PD/ XELOX)": "FOLFOX",
    "Y (SD/ FOLFOX)": "FOLFOX",
}
adata.obs["treatment_drug"] = adata.obs["Neoadjuvant chemotherapy"].astype(str).map(mapping)

mapping = {
    "pretreated": "nan",
    "treated-PR": "responder",  # PR: partial response
    "treated-PD/SD": "non-responder",  # PD/SD: progressive disease/stable disease
}
adata.obs["treatment_response"] = adata.obs["chemo"].map(mapping)

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
    + adata.obs["sample_id"]
    + "-"
    + adata.obs["original_obs_names"].str.rsplit("_", n=1).str[1]
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
adata.obs["patient_id"].unique()

# %%
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
