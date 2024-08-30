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
# # Dataloader: Wu_2024_Cell

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

# %%
cpus = nxfvars.get("cpus", 8)
d10_1016_j_cell_2024_02_005 = f"{databases_path}/d10_1016_j_cell_2024_02_005"
gtf = f"{annotation}/ensembl.v98_gene_annotation_table.csv"

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
adata = sc.AnnData(
    X=fmm.mmread(f"{d10_1016_j_cell_2024_02_005}/seurat/NEU_exprmat_matrix.mtx.gz"),
    obs=pd.read_csv(
        f"{d10_1016_j_cell_2024_02_005}/seurat/NEU_exprmat_barcodes.tsv",
        delimiter="\t",
        header=None,
        index_col=0,
    ),
    var=pd.read_csv(
        f"{d10_1016_j_cell_2024_02_005}/seurat/NEU_exprmat_features.tsv",
        delimiter="\t",
        header=None,
        index_col=0,
    ),
)
adata.X = csr_matrix(adata.X)
adata.var_names.name = None
adata.obs_names.name = None

# %%
cell_meta = pd.read_csv(
    f"{d10_1016_j_cell_2024_02_005}/seurat/NEU_metadata.tsv", sep="\t"
)

adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    cell_meta,
    how="left",
    on=["obs_names"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
sample_meta = pd.read_csv(
    f"{d10_1016_j_cell_2024_02_005}/metadata/sample_meta.csv"
).drop("Notes", axis=1)
patient_meta = pd.read_csv(
    f"{d10_1016_j_cell_2024_02_005}/metadata/patient_meta.csv"
).drop("Data Source", axis=1)

# %%
adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    sample_meta,
    how="left",
    left_on=["orig.ident"],
    right_on=["Sample name"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
adata.obs["obs_names"] = adata.obs_names
adata.obs = (
    pd.merge(
        adata.obs,
        patient_meta,
        how="left",
        on=["Patient name"],
        validate="m:1",
    )
    .set_index("obs_names")
    .drop("Cancer type_y", axis=1)
    .rename(columns={"Cancer type_x": "Cancer type"})
)
adata.obs_names.name = None

# %% [markdown]
# ### Gene annotations

# %%
# load gtf for gene mapping
gtf = pd.read_csv(gtf)
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

# %%
# Replace last score with underscore in unmapped_genes
adata.var.loc[adata.var["symbol"].isin(unmapped_genes), "symbol"] = adata.var.loc[
    adata.var["symbol"].isin(unmapped_genes), "symbol"
].apply(lambda x: ".".join(x.rsplit("-", maxsplit=1)))

# %%
adata.var["ensembl"] = (
    adata.var["symbol"].map(gene_ids).fillna(value=adata.var["symbol"])
)

mapping = {
    "Metazoa.SRP": "ENSG00000292350", # Metazoa_SRP
    "Y.RNA": "ENSG00000276932", # Y_RNA
    "DWORF": "ENSG00000240045",
}
adata.var["ensembl"] = (
    adata.var["ensembl"].map(mapping).fillna(value=adata.var["ensembl"])
)

adata.var_names = adata.var["ensembl"].apply(ah.pp.remove_gene_version)
adata.var_names.name = None

# %%
unmapped_genes = ah.pp.find_unmapped_genes(adata)
len(unmapped_genes)

# %% [markdown]
# -> Not quite sure what gtf reference was used here. To me it seems like depending on the cancer type, different annotations were used. Subsequently, when trying to map the remaining 1067 genes I get duplicate ensembl ids.
# Will just drop these genes!
# Also tried to subset by cancer_type/study and filter genes by min counts=10 -> did not remove duplicate genes ... I have really no idea what is going on! was the same data mapped to multiple gtfs?

# %% [markdown]
# ### sample/patient metadata
# column "Reference": Wu Y. et al. barcodes already present in Wu_2022 dataloader; Qian J. et al. present in Qian_2022 dataloader. Will only use new "In house" CRC samples from this dataset.

# %%
adata = adata[
    (adata.obs["cancer"] == "COAD") & (adata.obs["Reference"] == "This study")
].copy()

# %%
adata.obs = adata.obs.assign(
    study_id="Wu_2024_Cell",
    study_doi="10.1016/j.cell.2024.02.005",
    study_pmid="38447573",
    tissue_processing_lab="Qiang Gao lab",
    dataset="Wu_2024_CD66bPos",
    medical_condition="colorectal cancer",
    matrix_type="raw counts",
    # Paper methods: The sorted cells were sequenced using the 10x Chromium single-cell platform with 5' Reagent Kits following the manufacturer’s protocol. ChromiumTM Next GEM Single Cell 5' Library and Gel Bead Kit v1.1
    platform="10x 5' v1",
    cellranger_version="7.0.0",
    sample_tissue="colon",
    # Paper results: colon adenocarcinoma (COAD)
    histological_type="adenocarcinoma",
    # reference_genome="", ??
    # Paper methods: Patients with treatment naive primary tumors who underwent surgery
    treatment_status_before_resection="naive",
    tissue_cell_state="fresh",
    enrichment_cell_types="CD66b+",  # Some samples also CD45+, for example older in house Wu Y. et al.
    hospital_location="Zhongshan Hospital, Fudan University",
    country="China",
)

# %%
adata.obs["sample_id"] = adata.obs["orig.ident"]
adata.obs["patient_id"] = adata.obs["patient"]
adata.obs["cell_type_study"] = adata.obs["celltype_l1"]

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["orig.ident"]
    + "-"
    + adata.obs_names.str.rsplit("_", n=1).str[1]
)

# %%
mapping = {
    "Female": "female",
    "Male": "male",
}
adata.obs["sex"] = adata.obs["Gender"].map(mapping)
adata.obs["age"] = adata.obs["Age"].fillna(np.nan).astype("Int64")

# Same as in house Wu Y. et al: T= tumor; P = normal
mapping = {
    "T": "tumor",
    "P": "normal",
    "B": "blood",
}
adata.obs["sample_type"] = adata.obs["orig.ident"].str.split("_").str[1].map(mapping)

# %%
adata.obs["tumor_stage_TNM_T"] = (
    adata.obs["Stage"]
    .astype(str)
    .apply(
        lambda x: re.search(r"(T[0-4][a-c]?)", x).group(1)
        if re.search(r"(T[0-4][a-c]?)", x)
        else None
    )
)
adata.obs["tumor_stage_TNM_N"] = (
    adata.obs["Stage"]
    .astype(str)
    .apply(
        lambda x: re.search(r"(N[0-4][a-c]?)", x).group(1)
        if re.search(r"(N[0-4][a-c]?)", x)
        else None
    )
)
adata.obs["tumor_stage_TNM_M"] = (
    adata.obs["Stage"]
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
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

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
adata.obs["patient_id"].value_counts()

# %%
adata.obs["sample_id"].value_counts()

# %%
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
