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
# # Dataloader: Elmentaite_2021_Nature

# %%
import itertools
import os

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

# %%
cpus = nxfvars.get("cpus", 16)
d10_1038_s41586_021_03852_1 = f"{databases_path}/d10_1038_s41586_021_03852_1"
gtf = f"{annotation}/ensembl.v93_gene_annotation_table.csv"

# %%
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

# %%
# Read the YAML file and load it into a dictionary
with open(reference_meta, "r") as f:
    ref_meta_dict = yaml.load(f, Loader=yaml.Loader)

# %% [markdown]
# ## 1. Load raw counts

# %%
# Get available metadata
ENA_meta = pd.read_csv(
    f"{d10_1038_s41586_021_03852_1}/metadata/E-MTAB-9543_sample_meta.csv"
).drop(["ENA", "FASTQ", "time unit", "Scan Name"], axis=1)
# Remove duplicates
ENA_meta = ENA_meta[ENA_meta["read_type"] == "read1"].copy()

sample_meta = pd.read_csv(
    f"{d10_1038_s41586_021_03852_1}/metadata/Supplementary Table 2 Sample metadata.csv"
)

meta = pd.merge(
    sample_meta,
    ENA_meta,
    how="left",
    left_on=["Sanger.sample.ID"],
    right_on=["Source Name"],
    validate="m:1",
)

meta = meta[meta["Diagnosis"] == "Healthy adult"]

# %%
meta.to_csv(f"{artifact_dir}/Elmentaite_2021-original_sample_meta.csv", index=False)

# %% [markdown]
# ## Load gutcellatlas.org .h5ad

# %%
# Colon_cell_atlas = sc.read_h5ad(f"{d10_1038_s41586_021_03852_1}/Colon_cell_atlas.h5ad")

# %%
Full_obj_adata = sc.read_h5ad(
    f"{d10_1038_s41586_021_03852_1}/gutcellatlas.org/Full_obj_raw_counts_nosoupx_v2.h5ad"
)
Full_obj_adata = Full_obj_adata[
    Full_obj_adata.obs["Diagnosis"] == "Healthy adult"
].copy()

# %%
# Switch var_names to ensembl
Full_obj_adata.var["symbol"] = Full_obj_adata.var_names
Full_obj_adata.var_names = Full_obj_adata.var["gene_ids"]
Full_obj_adata.var_names.name = None

# %%
# Merge all available metadata
Full_obj_adata.obs["obs_names"] = Full_obj_adata.obs_names
Full_obj_adata.obs = pd.merge(
    Full_obj_adata.obs,
    meta.drop(["Diagnosis", "X10X", "Fraction"], axis=1),
    how="left",
    left_on=["batch"],
    right_on=["Sanger.sample.ID"],
    validate="m:1",
).set_index("obs_names")
Full_obj_adata.obs_names.name = None

# Rename obs_names
Full_obj_adata.obs["sample_id"] = Full_obj_adata.obs["ENA_SAMPLE"].fillna(
    Full_obj_adata.obs["batch"]
)
Full_obj_adata.obs_names = (
    Full_obj_adata.obs["sample_id"]
    + "-"
    + Full_obj_adata.obs_names.str.split("-").str[0]
)

# %%
Full_obj_adata.obs = Full_obj_adata.obs.assign(
    study_id="Elmentaite_2021_Nature",
    study_doi="10.1038/s41586-021-03852-1",
    study_pmid="34497389",
    tissue_processing_lab="Sarah A. Teichmann lab",
    dataset="Elmentaite_2021",
    medical_condition="healthy",
    sample_type="normal",
    tumor_source="normal",
    matrix_type="raw counts",
    reference_genome="ensembl.v93",
    treatment_status_before_resection="naive",
    tissue_cell_state="fresh",
    NCBI_BioProject_accession="PRJEB40273",
    hospital_location="Cambridge Biorepository of Translational Medicine",
    country="United Kingdom",
)

Full_obj_adata.obs["patient_id"] = (
    Full_obj_adata.obs["Sample name"]
    .str.replace(r"\(|\)", "", regex=True)
    .str.replace(" ", "_")
)
Full_obj_adata.obs["ENA_sample_accession"] = Full_obj_adata.obs["ENA_SAMPLE"]

# define the dictionary mapping
mapping = {
    "5'": "10x 5' v2",
    "3'": "10x 3' v2",
}
Full_obj_adata.obs["platform"] = Full_obj_adata.obs["10X"].map(mapping)

# define the dictionary mapping
mapping = {
    "5'": "3.0.0",
    "3'": "3.0.2",
}
Full_obj_adata.obs["cellranger_version"] = Full_obj_adata.obs["10X"].map(mapping)


# define the dictionary mapping
mapping = {
    "F": "female",
    "M": "male",
}
Full_obj_adata.obs["sex"] = Full_obj_adata.obs["Gender_x"].map(mapping)
Full_obj_adata.obs["age"] = Full_obj_adata.obs["Age_x"]

# define the dictionary mapping
mapping = {
    "SC": "naive",
    "SC-45N": "CD45-",
    "SC-45P": "CD45+",
}
Full_obj_adata.obs["enrichment_cell_types"] = Full_obj_adata.obs["Fraction"].map(
    mapping
)

Full_obj_adata.obs["dataset"] = (
    Full_obj_adata.obs["dataset"].astype(str)
    + "_"
    + Full_obj_adata.obs["platform"].astype(str)
    + "_"
    + Full_obj_adata.obs["enrichment_cell_types"].astype(str)
)

mapping = {
    "Elmentaite_2021_10x 5' v2_CD45-": "Elmentaite_2021_10x5p_CD45Neg",
    "Elmentaite_2021_10x 5' v2_CD45+": "Elmentaite_2021_10x5p_CD45Pos",
    "Elmentaite_2021_10x 5' v2_naive": "Elmentaite_2021_10x5p",
    "Elmentaite_2021_10x 3' v2_CD45+": "Elmentaite_2021_10x3p_CD45Pos",
    "Elmentaite_2021_10x 3' v2_CD45-": "Elmentaite_2021_10x3p_CD45Neg",
    "Elmentaite_2021_10x 3' v2_naive": "Elmentaite_2021_10x3p",
}
Full_obj_adata.obs["dataset"] = Full_obj_adata.obs["dataset"].map(mapping)


# Paper methods: Samples were collected from 11 distinct locations, including the duodenum (two locations, DUO1 and DUO2, which were pooled in the analysis),
# jejunum (JEJ), ileum (two locations, ILE1 and ILE2, which were pooled in the analysis), appendix (APD), caecum (CAE), ascending colon (ACL),
# transverse colon (TCL), descending colon (DCL), sigmoid colon (SCL), rectum (REC) and mesenteric lymph nodes (mLN).
# Fresh mucosal intestinal tissue and lymph nodes from the intestinal mesentery were excised within 1 h of circulatory arrest.

# define the dictionary mapping
mapping = {
    "DUO": "duodenum",
    "JEJ": "jejunum",
    "ILE": "ileum",
    "ILE1": "ileum",
    "ILE2": "ileum",
    "APD": "appendix",
    "CAE": "cecum",
    "ACL": "ascending colon",
    "TCL": "transverse colon",
    "DCL": "descending colon",
    "SCL": "sigmoid colon",
    "REC": "rectum",
    "MLN": "mesenteric lymph nodes",
}
Full_obj_adata.obs["anatomic_location"] = Full_obj_adata.obs["Region code"].map(mapping)


# define the dictionary mapping
mapping = {
    "DUO": "proximal small intestine",  # small intestine
    "JEJ": "middle small intestine",
    "ILE": "middle small intestine",
    "ILE1": "middle small intestine",
    "ILE2": "middle small intestine",
    "APD": "appendix",
    "CAE": "proximal colon",  # large intestine
    "ACL": "proximal colon",
    "TCL": "proximal colon",
    "DCL": "distal colon",
    "SCL": "distal colon",
    "REC": "distal colon",
    "MLN": "mesenteric lymph nodes",
}
Full_obj_adata.obs["anatomic_region"] = Full_obj_adata.obs["Region code"].map(mapping)

# define the dictionary mapping
mapping = {
    "DUO": "small intestine",
    "JEJ": "small intestine",
    "ILE": "small intestine",
    "ILE1": "small intestine",
    "ILE2": "small intestine",
    "APD": "appendix",
    "CAE": "colon",
    "ACL": "colon",
    "TCL": "colon",
    "DCL": "colon",
    "SCL": "colon",
    "REC": "colon",
    "MLN": "lymph node",
}
Full_obj_adata.obs["sample_tissue"] = Full_obj_adata.obs["Region code"].map(mapping)

Full_obj_adata.obs["cell_type_study"] = Full_obj_adata.obs["Integrated_05"]
Full_obj_adata.obs["cell_type_coarse_study"] = Full_obj_adata.obs["category"]

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
# Subset adata for columns in the reference meta
Full_obj_adata.obs = Full_obj_adata.obs[
    [col for col in ref_meta_cols if col in Full_obj_adata.obs.columns]
].copy()

# %%
Full_obj_adata.obs["patient_id"].unique()

# %%
# Get compiled sample/patient meta from gutcellatlas.org for raw adata
sample_meta = (
    Full_obj_adata.obs[
        Full_obj_adata.obs.columns[
            Full_obj_adata.obs.nunique() <= Full_obj_adata.obs["sample_id"].nunique()
        ]
    ]
    .groupby("sample_id")
    .first()
    .reset_index()
)


# %% [markdown]
# ## Load raw counts from fastq

# %%
def load_counts(sample_meta, databases_path, mat_type="filtered"):

    file_path = f"{databases_path}/10_nfcore_scrnaseq/cellranger/count/{sample_meta['ENA_SAMPLE']}/outs/{mat_type}_feature_bc_matrix.h5"
    adata = sc.read_10x_h5(file_path)
    adata.obs = adata.obs.assign(**sample_meta)
    adata.var_names_make_unique()
    adata.X = csr_matrix(adata.X.astype(int))
    adata.obs_names = (
        adata.obs["ENA_SAMPLE"] + "-" + adata.obs_names.str.replace("-1", "")
    )
    assert adata.obs_names.is_unique
    return adata


# %%
adatas = process_map(
    load_counts,
    [r for _, r in meta.dropna(subset=["ENA_SAMPLE"]).iterrows()],
    itertools.repeat(d10_1038_s41586_021_03852_1),
    itertools.repeat("raw"),
    max_workers=cpus,
)
adata = anndata.concat(adatas, join="outer")

# %% [markdown]
# ### Gene annotations

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
# ### Merge already compiled metadata from gutcellatlas_adata

# %%
# Merge all available metadata
adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    sample_meta,
    how="left",
    left_on=["ENA_SAMPLE"],
    right_on=["sample_id"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
adata.obs["cellranger_version"] = "7.1.0"
adata.obs["reference_genome"] = "gencode.v44"

# Map back study cell type annotations
adata.obs["cell_type_study"] = adata.obs_names.map(
    Full_obj_adata.obs["cell_type_study"].to_dict()
)
adata.obs["cell_type_coarse_study"] = adata.obs_names.map(
    Full_obj_adata.obs["cell_type_coarse_study"].to_dict()
)

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
# Append dataset to obs_names
Full_obj_adata.obs_names = (
    Full_obj_adata.obs["dataset"] + "-" + Full_obj_adata.obs_names
)
adata.obs_names = adata.obs["dataset"] + "-" + adata.obs_names

# %% [markdown]
# ### Save raw counts from additional samples (10x 3 prime)
# "Full_obj_adata" contains additional samples! Also will need to filter out non colon samples!

# %%
Full_obj_adata[
    Full_obj_adata.obs["anatomic_region"].isin(
        ["distal colon", "mesenteric lymph nodes", "proximal colon"]
    )
].obs["sample_id"].value_counts()

# %%
# Keep only additional samples not available in Elmentaite_2021
Full_obj_adata = Full_obj_adata[
    ~(Full_obj_adata.obs_names.isin(adata.obs_names))
].copy()

Full_obj_adata.obs = Full_obj_adata.obs.dropna(axis=1, how="all")

# %%
Full_obj_adata

# %%
datasets = [
    Full_obj_adata[Full_obj_adata.obs["dataset"] == dataset, :].copy()
    for dataset in Full_obj_adata.obs["dataset"].unique()
    if Full_obj_adata[Full_obj_adata.obs["dataset"] == dataset, :].shape[0] != 0
]

# %%
datasets

# %%
for _adata in datasets:
    _adata.obs = _adata.obs.dropna(
        axis=1, how="all"
    )  # Remove all nan columns, else write_h5ad will fail
    dataset = _adata.obs["dataset"].values[0]  # Get the dataset name from the subset
    _adata.write_h5ad(
        f"{artifact_dir}/{dataset}_gutcellatlas-adata.h5ad", compression="lzf"
    )

# %% [markdown]
# ### Save raw counts from fastq (10x 5 prime)

# %%
adata

# %%
adata[
    adata.obs["anatomic_region"].isin(
        ["distal colon", "mesenteric lymph nodes", "proximal colon"]
    )
].obs["sample_id"].value_counts()

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
