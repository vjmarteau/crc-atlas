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
# # Dataloader: UZH_Zurich - Arnold lab

# %%
import os
import re
from pathlib import Path

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
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")
cpus = nxfvars.get("cpus", 2)

# %%
data_path = nxfvars.get("data_path", "../../data/own_datasets/")
bd_output = f"{data_path}/arnold_lab/10_rhapsody_pipeline_v2/Fixed/"

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
patient_metadata = pd.read_csv(
    f"{data_path}/arnold_lab/metadata/Clinical_Details_CRC_scRNAseq.csv"
)


# %%
def load_adata_from_mtx(sample_meta):
    # Load data into memory
    matrix = fmm.mmread(
        f"{bd_output}/{sample_meta['Experiment ID']}/{sample_meta['Experiment ID']}_RSEC_MolsPerCell_Unfiltered_MEX/matrix.mtx.gz"
    )

    barcodes = pd.read_csv(
        f"{bd_output}/{sample_meta['Experiment ID']}/{sample_meta['Experiment ID']}_RSEC_MolsPerCell_Unfiltered_MEX/barcodes.tsv.gz",
        delimiter="\t",
        header=None,
        names=["Cell_Index"],
    )
    barcodes["obs_names"] = "Cell_Index_" + barcodes["Cell_Index"].astype(str)
    barcodes.set_index("obs_names", inplace=True)
    barcodes.index.name = None

    features = pd.read_csv(
        f"{bd_output}/{sample_meta['Experiment ID']}/{sample_meta['Experiment ID']}_RSEC_MolsPerCell_Unfiltered_MEX/features.tsv.gz",
        delimiter="\t",
        header=None,
        names=["var_names", "symbol", "type"],
    )

    # Assemble adata
    adata = sc.AnnData(X=matrix.T)
    adata.X = csr_matrix(adata.X)
    adata.var = features
    adata.obs = barcodes
    adata.obs = adata.obs.assign(**sample_meta)
    adata.var_names = adata.var["var_names"]
    adata.var_names.name = None

    adata_tmp = sc.read_h5ad(
        f"{bd_output}/{sample_meta['Experiment ID']}/{sample_meta['Experiment ID']}.h5ad"
    )
    adata_tmp.obs_names = "Cell_Index_" + adata_tmp.obs_names

    for col in ["Cell_Type_Experimental", "Sample_Tag", "Sample_Name"]:
        sample_tag_dict = adata_tmp.obs[col].to_dict()
        adata.obs[col] = adata.obs_names.map(sample_tag_dict)

    return adata


# %%
adatas = process_map(
    load_adata_from_mtx,
    [r for _, r in patient_metadata.iterrows()],
    max_workers=cpus,
)

# %%
adata = anndata.concat(adatas, index_unique="-", join="outer", fill_value=0)

# %% [markdown]
# ## 2. Gene annotations

# %%
gtf = f"{annotation}/gencode.v44_gene_annotation_table.csv"
gtf = pd.read_csv(gtf)
gtf["ensembl"] = gtf["Geneid"].apply(ah.pp.remove_gene_version)

# %%
# map back gene symbols to .var
gene_symbols = gtf.set_index("GeneSymbol")["ensembl"].to_dict()

# %%
adata.var["symbol"] = adata.var_names
adata.var["ensembl"] = (
    adata.var["symbol"].map(gene_symbols).fillna(value=adata.var["symbol"])
)

# %%
adata.var_names = adata.var["ensembl"]
adata.var_names.name = None

# %%
unmapped_genes = ah.pp.find_unmapped_genes(adata)
len(unmapped_genes)

# %%
# Reorder by gtf (i.e. chromosome position)
gene_index = gtf[gtf["ensembl"].isin(adata.var_names)]["ensembl"].values
missing_genes = adata[:, ~adata.var_names.isin(gtf["ensembl"])].var_names.values
gene_index = np.append(gene_index, missing_genes)
adata = adata[:, gene_index]

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="UZH_Zurich",
    tissue_processing_lab="Isabelle Arnold lab",
    dataset="UZH_Zurich_CD45Pos",
    medical_condition="colorectal cancer",
    # hospital_location="Basel/Zürich?",
    country="Switzerland",
    tissue_cell_state="fresh",
    platform="BD Rhapsody",
    reference_genome="gencode.v44",
    matrix_type="raw counts",
)

adata.obs["sample_id"] = adata.obs["Experiment ID"] + "_" + adata.obs["Sample_Name"]
adata.obs["patient_id"] = adata.obs["Patient ID"].fillna(adata.obs["sample_id"])

adata.obs["multiplex_batch"] = adata.obs["Experiment ID"]

# Update dataset column
adata.obs.loc[
    adata.obs["multiplex_batch"] == "Experiment6", "dataset"
] = "UZH_Zurich_healthy_blood"

# Update medical_condition column
adata.obs.loc[
    adata.obs["multiplex_batch"] == "Experiment6", "medical_condition"
] = "healthy"

mapping = {
    "Undetermined": "Undetermined",
    "Multiplet": "Multiplet",
    "T": "tumor",
    "N": "normal",
    "B": "blood",
    "HB": "blood",
    "HB6": "blood",
    "HB7": "blood",
    "HB8": "blood",
    "HB9": "blood",
    "HB10": "blood",
    "HB11": "blood",
    "SampleTag02_hs": "nan",
    "SampleTag04_hs": "nan",
    "SampleTag06_hs": "nan",
    "nan": "nan",
}
adata.obs["sample_type"] = adata.obs["Sample_Name"].astype(str).map(mapping)

mapping = {
    "Undetermined": "Undetermined",
    "Multiplet": "Multiplet",
    "tumor": "colon",
    "normal": "colon",
    "blood": "blood",
    "nan": "nan",
}
adata.obs["sample_tissue"] = adata.obs["sample_type"].map(mapping)

mapping = {
    "Undetermined": "Undetermined",
    "Multiplet": "Multiplet",
    "T": "CD45+",
    "N": "CD45+",
    "B": "Eosinophils",
    "HB": "Eosinophils",
    "HB6": "Eosinophils",
    "HB7": "Eosinophils",
    "HB8": "Eosinophils",
    "HB9": "Eosinophils",
    "HB10": "Eosinophils",
    "HB11": "Eosinophils",
    "SampleTag02_hs": "nan",
    "SampleTag04_hs": "nan",
    "SampleTag06_hs": "nan",
    "nan": "nan",
}
adata.obs["enrichment_cell_types"] = adata.obs["Sample_Name"].astype(str).map(mapping)

mapping = {
    "M": "male",
    "F": "female",
    "nan": "nan",
}

mapping = {
    "Rectum": "rectum",
    "Colon": "nan",
    "nan": "nan",
}
adata.obs["anatomic_location"] = adata.obs["Tumor type"].astype(str).map(mapping)

mapping = {
    "rectum": "distal colon",
    "nan": "nan",
}
adata.obs["anatomic_region"] = adata.obs["anatomic_location"].map(mapping)

adata.obs["sex"] = adata.obs["Sex"].astype(str).map(mapping)
adata.obs["age"] = adata.obs["Age"].fillna(np.nan).astype("Int64")
adata.obs["microsatellite_status"] = adata.obs["Genetic alteration"]
adata.obs["tumor_grade"] = adata.obs["Tumor grading (WHO)"]

mapping = {
    "Chemotherapy": "treated",
    "not treated": "naive",
    "nan": "nan",
}
adata.obs["treatment_status_before_resection"] = (
    adata.obs["Treatment before resection"].astype(str).map(mapping)
)

mapping = {
    "Chemotherapy": "chemotherapy",
    "not treated": "nan",
    "nan": "nan",
}
adata.obs["treatment_drug"] = (
    adata.obs["Treatment before resection"].astype(str).map(mapping)
)

# %%
mapping = {
    "pT2 pN0(0/44) L0 V0 Pn0 R0": "T2",
    "pT2 pN0(0/18) L0 V0 Pn0 R0": "T2",
    "ypT4a pN1b (2/24) L1 V1 Pn0 R0": "T4a",
    "pT3 pN0(0/35) L0 V0 Pn0 R0": "T3",
    "nan": "nan",
}

adata.obs["tumor_stage_TNM_T"] = (
    adata.obs["TNM classification: tumor(T) node (N) metastasis (M)"]
    .astype(str)
    .map(mapping)
)

mapping = {
    "pT2 pN0(0/44) L0 V0 Pn0 R0": "N0",
    "pT2 pN0(0/18) L0 V0 Pn0 R0": "N0",
    "ypT4a pN1b (2/24) L1 V1 Pn0 R0": "N1b",
    "pT3 pN0(0/35) L0 V0 Pn0 R0": "N0",
    "nan": "nan",
}

adata.obs["tumor_stage_TNM_N"] = (
    adata.obs["TNM classification: tumor(T) node (N) metastasis (M)"]
    .astype(str)
    .map(mapping)
)

adata.obs["tumor_stage_TNM_M"] = "M0"

adata.obs["tumor_stage_TNM"] = ah.pp.tnm_tumor_stage(adata.obs, "patient_id")

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
ref_meta_cols += [
    "multiplex_batch",
    "Cell_Type_Experimental",
    "Sample_Tag",
    "Sample_Name",
    "BD-Rhapsody File ID",
]

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
adata.obs_names = (
    adata.obs["dataset"].astype(str)
    + "-"
    + adata.obs["multiplex_batch"].astype(str)
    + "-"
    # + adata.obs["Sample_Tag"].astype(str)
    # + "_"
    + adata.obs_names.str.split("-").str[0].str.split("Cell_Index_").str[1]
)

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
adata

# %%
# Get original sample/patient meta and remove cell level meta
sample_meta = (
    adata.obs[
        adata.obs.columns[adata.obs.nunique() <= adata.obs["sample_id"].nunique()]
    ]
    .groupby("sample_id")
    .first()
    .reset_index()
)

# %%
sample_meta.to_csv(
    f"{artifact_dir}/UZH_Zurich-original_sample_meta.csv",
    index=False,
)

# %% [markdown]
# ## 4. Save adata by dataset

# %%
datasets = [
    adata[adata.obs["dataset"] == dataset, :].copy()
    for dataset in adata.obs["dataset"].unique()
    if adata[adata.obs["dataset"] == dataset, :].shape[0] != 0
]

# %%
datasets

# %%
for _adata in datasets:
    _adata.obs = _adata.obs.dropna(
        axis=1, how="all"
    )  # Remove all nan columns, else write_h5ad will fail
    dataset = _adata.obs["dataset"].values[0]  # Get the dataset name from the subset
    _adata.write_h5ad(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
