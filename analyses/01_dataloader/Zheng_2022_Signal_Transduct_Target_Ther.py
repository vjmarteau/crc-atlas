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
# # Dataloader: Zheng_2022_Signal_Transduct_Target_Ther

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

# %% tags=["parameters"]
artifact_dir = nxfvars.get("artifact_dir", "/data/scratch/marteau/downloads/crc-atlas/")
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../tables/annotation")
cpus = nxfvars.get("cpus", 2)
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")

# %%
d10_1038_s41392_022_00881_8 = f"{databases_path}/d10_1038_s41392_022_00881_8"

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
# Get all file names in dir by sample
files = [
    x.stem.split("_matrix")[0]
    for x in Path(f"{d10_1038_s41392_022_00881_8}/GSE161277/GSE161277_RAW").glob(
        "*.mtx.gz"
    )
]
files.sort()

# %%
files


# %%
def _load(stem):
    mat = fmm.mmread(
        f"{d10_1038_s41392_022_00881_8}/GSE161277/GSE161277_RAW/{stem}_matrix.mtx.gz"
    )

    barcodes = pd.read_csv(
        f"{d10_1038_s41392_022_00881_8}/GSE161277/GSE161277_RAW/{stem}_barcodes.tsv.gz",
        header=None,
    )
    features = pd.read_csv(
        f"{d10_1038_s41392_022_00881_8}/GSE161277/GSE161277_RAW/{stem}_features.tsv.gz",
        header=None,
        delimiter="\t",
    )

    features = features.set_index(0)
    barcodes = barcodes.set_index(0)
    features.index.name = None
    barcodes.index.name = None

    adata = sc.AnnData(X=mat.T)
    adata.X = csr_matrix(adata.X)
    adata.var = features
    adata.obs = barcodes
    adata.obs["file name"] = stem
    return adata


# %%
adatas = [_load(stem) for stem in files]

# %%
adata = anndata.concat(adatas, index_unique="_", join="outer", fill_value=0)
adata.obs_names = adata.obs_names.str.split("_").str[0]
adata.obs_names_make_unique()
# Append back gene symbols
adata.var["symbol"] = adata.var_names.map(adatas[0].var[1].to_dict())

adata.obs["sample_accession_number"] = adata.obs["file name"].str.split("_").str[0]
adata.obs["patient_id"] = (
    adata.obs["file name"].str.split("_").str[1].str.replace("atient", "")
)
adata.obs["Sample"] = adata.obs["file name"].str.split("_").str[2]
adata.obs["sample_id"] = adata.obs["patient_id"] + "_" + adata.obs["Sample"]


# %%
# quick way to find used annotation
# gtf = pd.read_csv(f"{annotation}/ensembl.v93_gene_annotation_table.csv")
# gtf["ensembl"] = gtf["gene_id"].apply(ah.pp.remove_gene_version)
# gtf = ah.pp.append_duplicate_suffix(df=gtf, column="gene_name", sep=".")
# gene_ids = gtf.set_index("ensembl")["gene_name"].to_dict()

# adata.var["ensembl"] = adata.var_names
# adata.var["var_names"] = adata.var["ensembl"].map(gene_ids)
# adata.var['var_names'].isna().sum()

# %% [markdown]
# ### 2a. Compile sample/patient metadata from original study supplemetary

# %%
def merge_columns(row):
    values = [str(value) for value in row if pd.notnull(value)]
    if len(values) == 1:
        return values[0]
    elif len(values) > 1:
        return " ".join(values)
    else:
        return np.nan


# %%
# import tabula
# tables = tabula.read_pdf(f"{d10_1038_s41392_022_00881_8}/41392_2022_881_MOESM1_ESM_Table_S1.pdf")
# meta = tables[0]
# meta = meta.T

# Reset the index of the DataFrame
# meta = meta.reset_index()
# Fill NaN values in the first row with column names
# for col in meta.columns:
#    if pd.isna(meta.loc[0, col]):
#        meta.loc[0, col] = col

# meta.columns = meta.iloc[0]
# meta = meta[1:]

# meta["tissue processed"] = meta.iloc[:, 6:9].apply(merge_columns, axis=1)
# meta["IHC"] = meta.iloc[:, 18:22].apply(merge_columns, axis=1)

# meta = meta[
#    [
#        "Patient ID",
#        "Age",
#        "Gender",
#        "Site of tumor",
#        "Histological type of tumor",
#        "Histological type of polyp",
#        "tissue processed",
#        "pTNM: T",
#        "pTNM: N",
#        "pTNM: M",
#        "pTNM",
#        "Stage",
#        "Tumor size",
#        "Grade",
#        "MSI statusb",
#        "Genotype",
#        "IHC",
#    ]
# ]

# Load table from csv instead of from the pdf above
meta = pd.read_csv(f"{d10_1038_s41392_022_00881_8}/sample_meta.csv")
meta["patient_id"] = meta["Patient ID"].str.extract(r"\((.*?)\)")

# %%
geofetch = pd.read_csv(
    f"{d10_1038_s41392_022_00881_8}/GSE161277/GSE161277_PEP/GSE161277_PEP_raw.csv"
).dropna(axis=1, how="all")

# %%
# Merge accession numbers
adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    geofetch,
    how="left",
    left_on=["sample_accession_number"],
    right_on=["sample_geo_accession"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# Merge meta
adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    meta,
    how="left",
    on=["patient_id"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

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
    f"{artifact_dir}/Zheng_2022-original_sample_meta.csv",
    index=False,
)

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="Zheng_2022_Signal_Transduct_Target_Ther",
    study_doi="10.1038/s41392-022-00881-8",
    study_pmid="35221332",
    dataset="Zheng_2022",
    medical_condition="colorectal cancer",
    tissue_processing_lab="Hubing Shi lab",
    enrichment_cell_types="naive",
    tissue_cell_state="fresh",
    histological_type="adenocarcinoma",
    # Paper Methods: Only patients with untreated, non-metastatic CRCs that underwent radical resection were included in this study.
    treatment_status_before_resection="naive",
    tumor_stage_TNM_M="M0",
    platform="10x 3' v2",
    matrix_type="raw counts",
    reference_genome="ensembl.v93",
    cellranger_version="3.0.0",
    NCBI_BioProject_accession="PRJNA676119",
    hospital_location="West China Hospital, Sichuan University",
    country="China",
)

adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]

# define the dictionary mapping
mapping = {
    "carcinoma": "tumor",
    "adenoma": "polyp",
    "normal": "normal",
    "para-cancer": "tumor",  # ? "para-" often means "beside" or "near" -> normal? -> From figure 1 it looks more like tumor border
    "blood": "blood",
}
adata.obs["sample_type"] = adata.obs["Sample"].map(mapping)

# define the dictionary mapping
mapping = {
    "carcinoma": "core",
    "adenoma": "na",
    "normal": "normal",
    "para-cancer": "border",
    "blood": "na",
}
adata.obs["tumor_source"] = adata.obs["Sample"].map(mapping)

# define the dictionary mapping
mapping = {
    "tumor": "colon",
    "polyp": "colon",
    "para-cancer": "colon", # -> From figure 1 it looks more like tumor border (need to ask pathologist!)
    "normal": "colon",
    "pbmc": "blood",
}
adata.obs["sample_tissue"] = adata.obs["sample_type"].map(mapping)


# define the dictionary mapping
mapping = {
    "none": "nan",
    "tubulovillous adenoma": "tubulovillous conventional adenoma",
}
adata.obs["histological_type"] = adata.obs["Histological type of polyp"].map(
    mapping
)
adata.obs.loc[
    adata.obs["sample_type"].isin(["tumor", "normal", "pbmc"]),
    "histological_type",
] = "nan"

adata.obs.loc[
    adata.obs["sample_type"].isin(["tumor"]),
    "histological_type",
] = "adenocarcinoma"

mapping = {
    "Female": "female",
    "Male": "male",
}
adata.obs["sex"] = adata.obs["Gender"].map(mapping)

adata.obs["age"] = adata.obs["Age"].fillna(np.nan).astype("Int64")

adata.obs["tumor_grade"] = adata.obs["Grade"]

adata.obs["tumor_stage_TNM_T"] = adata.obs["pTNM"].apply(
    lambda x: re.search(r"(T[0-4][a-c]?)", x).group(1)
    if re.search(r"(T[0-4][a-c]?)", x)
    else None
)
adata.obs["tumor_stage_TNM_N"] = adata.obs["pTNM"].apply(
    lambda x: re.search(r"(N[0-4][a-c]?)", x).group(1)
    if re.search(r"(N[0-4][a-c]?)", x)
    else None
)
adata.obs["tumor_stage_TNM"] = ah.pp.tnm_tumor_stage(adata.obs, "patient_id")

# define the dictionary mapping
mapping = {
    "P0": "nan",
    "P1": "nan",
    "P2": "rectum",
    "P3": "nan",
}
adata.obs["anatomic_location"] = adata.obs["patient_id"].map(mapping)

# define the dictionary mapping
mapping = {
    "P0": "nan",
    "P1": "nan",
    "P2": "distal colon",
    "P3": "nan",
}
adata.obs["anatomic_region"] = adata.obs["patient_id"].map(mapping)

# define the dictionary mapping
mapping = {
    "NAc": "nan",
    "MSS": "MSS",
}
adata.obs["microsatellite_status"] = adata.obs["MSI statusb"].map(mapping)

adata.obs["tumor_dimensions"] = adata.obs["Tumor size"].str.replace(" ", "")

# define the dictionary mapping
mapping = {
    "4.3×3.8×1.7cm": 4.3,
    "6.1×2.3×0.4cm": 6.1,
    "5.5×2.5×1.3cm": 5.5,
    "4.5×3.5×1.2cm": 4.5,
}
adata.obs["tumor_size"] = adata.obs["tumor_dimensions"].map(mapping)
adata.obs["tumor_size"] = round(adata.obs["tumor_size"].astype("float"), 2)

# %%
adata.obs[["patient_id", "Genotype", "MSI statusb", "IHC"]].groupby(
    "patient_id"
).first()

# %%
# mismatch repair-proficient (MMRp) -> "(+)" should be MSS!
mapping = {
    "MLH1(+), MSH2(+) MSH6(+), PMS2(+) Ki67(+), CDX2(+)": "pMMR",
    "MLH1(+), MSH2(+) MSH6(+), PMS2(+) Ki67(+), CDX2(+) Desmin(+)": "pMMR",
    "MLH1(+), MSH2(+) MSH6(+), PMS2(+) Ki67(+), CDX2(+) CK20(+)": "pMMR",
    "MLH1(+), MSH2(+) MSH6(+), PMS2(+) Ki67(+)": "pMMR",
}

adata.obs["mismatch_repair_deficiency_status"] = adata.obs["IHC"].map(mapping)

# %%
# define the dictionary mapping
mapping = {
    "P0": "nan",
    "P1": "mut",
    "P2": "wt",
    "P3": "mut",
}
adata.obs["APC_status_driver_mut"] = adata.obs["patient_id"].map(mapping)

# define the dictionary mapping
mapping = {
    "P0": "nan",
    "P1": "mut",
    "P2": "wt",
    "P3": "mut",
}
adata.obs["KRAS_status_driver_mut"] = adata.obs["patient_id"].map(mapping)

# define the dictionary mapping
mapping = {
    "P0": "nan",
    "P1": "mut",
    "P2": "wt",
    "P3": "wt",
}
adata.obs["PIK3CA_status_driver_mut"] = adata.obs["patient_id"].map(mapping)

# define the dictionary mapping
mapping = {
    "P0": "nan",
    "P1": "wt",
    "P2": "mut",
    "P3": "wt",
}
adata.obs["TP53_status_driver_mut"] = adata.obs["patient_id"].map(mapping)

# define the dictionary mapping
mapping = {
    "P0": "nan",
    "P1": "wt",
    "P2": "wt",
    "P3": "mut",
}
adata.obs["FAT4_status_driver_mut"] = adata.obs["patient_id"].map(mapping)


# Get the column names that contain "_status_driver_mut"
driver_mutation_columns = [
    col for col in adata.obs.columns if "_status_driver_mut" in col
]
# Set the values to NaN for "normal" sample_type in the driver mutation columns
adata.obs.loc[adata.obs["sample_type"] == "normal", driver_mutation_columns] = "nan"

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
# Add newly created driver_mutation columns
ref_meta_cols += [
    col for col in adata.obs.columns if col.endswith("_status_driver_mut")
]

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
adata.obs["sample_id"] = adata.obs["sample_id"].str.replace("-", "_")

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
adata.obs["sample_id"].value_counts()

# %%
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
