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
# # Dataloader: Lee_2020_Nat_Genet

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
cpus = nxfvars.get("cpus", 6)
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")

# %%
d10_1038_s41588_020_0636_z = f"{databases_path}/d10_1038_s41588_020_0636_z"
gtf = f"{annotation}/ensembl.v84_gene_annotation_table.csv"

# %%
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

# %%
# Read the YAML file and load it into a dictionary
with open(reference_meta, "r") as f:
    ref_meta_dict = yaml.load(f, Loader=yaml.Loader)

# %%
# get original metadata
file_path = f"{d10_1038_s41588_020_0636_z}/metadata/41588_2020_636_MOESM3_ESM.xlsx"
meta = pd.read_excel(file_path, sheet_name="Table S1")

SMC_meta = meta[2:26]
SMC_meta.columns = SMC_meta.iloc[0]
SMC_meta = SMC_meta.iloc[1:].reset_index(drop=True).rename_axis(None, axis=1)

KUL3_meta = meta[28:]
KUL3_meta.columns = KUL3_meta.iloc[0]
KUL3_meta = (
    KUL3_meta.iloc[1:].reset_index(drop=True).dropna(axis=1).rename_axis(None, axis=1)
)

# This is such a bad abbrev. for mut and wt! Had to look it up in Fig1 ...
for col in ["TP53", "APC", "KRAS", "BRAF", "SMAD4"]:
    KUL3_meta[col] = KUL3_meta[col].replace({"T": "mut", "F": "wt"})

# %%
# get original metadata for missing 3 patients from Joanito_2022_Nat_Genet
file_path = f"{d10_1038_s41588_020_0636_z}/metadata/Joanito_2022_Nat_Genet/41588_2022_1100_MOESM3_ESM.xlsx"
meta = pd.read_excel(file_path, sheet_name="Supplementary Table 13")
meta = meta.loc[meta["dataset"] == "KUL3"]

# %% [markdown]
# ## SMC cohort

# %% [markdown]
# ## 1. Load raw adata counts from GEO

# %%
GEO = [
    "GSE132257/GSE132257_GEO_processed_protocol_and_fresh_frozen_raw_UMI_count_matrix.txt.gz",
    "GSE132465/GSE132465_GEO_processed_CRC_10X_raw_UMI_count_matrix.txt.gz",
    "GSE144735/GSE144735_processed_KUL3_CRC_10X_raw_UMI_count_matrix.txt.gz",
]


# %%
def _load_mtx(mtx_path):
    adata = sc.read_csv(f"{d10_1038_s41588_020_0636_z}/{mtx_path}/", delimiter="\t")
    adata = adata.T
    adata.X = csr_matrix(adata.X)
    adata.obs["id"] = adata.obs_names.str.split("_").str[0]
    return adata


# %% [markdown]
# ### 1a. SMC cohort
# 23 Korean patients with CRC from the Samsung Medical Center (SMC).
# Samples were cryopreserved in CELLBANKER 1 (Zenoaq Resource) before scRNA-seq.
#
# ### 1b. extra cryo test setup samples
# To compare sample preparation methods, CRC and matched normal tissues from one patient were processed using three dissociation protocols: SMC; KUL3; or Singapore.
# A single-cell suspension was generated from another patient sample using the SMC protocol and was subjected to scRNA-seq immediately or after a freezethaw cycle.

# %%
adata_SMC = _load_mtx(
    "GSE132465/GSE132465_GEO_processed_CRC_10X_raw_UMI_count_matrix.txt.gz"
)
adata_SMC.obs = adata_SMC.obs.assign(
    dataset="Lee_2020_SMC",
    tissue_cell_state="frozen",
    NCBI_BioProject_accession="PRJNA548146",
)

# %%
geofetch = pd.read_csv(
    f"{d10_1038_s41588_020_0636_z}/GSE132465/GSE132465_PEP/GSE132465_PEP_raw.csv"
).dropna(axis=1, how="all")

# %%
adata_SMC.obs["obs_names"] = adata_SMC.obs_names
adata_SMC.obs = pd.merge(
    adata_SMC.obs,
    geofetch,
    how="left",
    left_on=["id"],
    right_on=["sample_title"],
    validate="m:1",
).set_index("obs_names")
adata_SMC.obs_names.name = None

# %%
SMC_extra = _load_mtx(
    "GSE132257/GSE132257_GEO_processed_protocol_and_fresh_frozen_raw_UMI_count_matrix.txt.gz"
)

SMC_extra.obs = SMC_extra.obs.assign(NCBI_BioProject_accession="PRJNA546616")

# %%
geofetch = pd.read_csv(
    f"{d10_1038_s41588_020_0636_z}/GSE132257/GSE132257_PEP/GSE132257_PEP_raw.csv"
).dropna(axis=1, how="all")
geofetch["sample_title"] = geofetch["sample_title"].str.replace("SMC13T", "SMC13T-A1")
geofetch["sample_title"] = geofetch["sample_title"].str.replace("SMC13N", "SMC13N-A1")
geofetch["sample_title"] = geofetch["sample_title"].str.replace("-", ".")

# %%
SMC_extra.obs["obs_names"] = SMC_extra.obs_names
SMC_extra.obs = pd.merge(
    SMC_extra.obs,
    geofetch,
    how="left",
    left_on=["id"],
    right_on=["sample_title"],
    validate="m:1",
).set_index("obs_names")
SMC_extra.obs_names.name = None

mapping = {
    "Cryopreserved cell dissociates": "frozen",
    "Fresh cell dissociates": "fresh",
}
SMC_extra.obs["tissue_cell_state"] = SMC_extra.obs["cell_preparation"].map(mapping)

SMC_extra.obs["dataset"] = (
    "Lee_2020_SMC_test_setup_" + SMC_extra.obs["tissue_cell_state"]
)

# %%
adata = anndata.concat(
    [adata_SMC, SMC_extra], index_unique="-", join="outer", fill_value=0
)

# %%
adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    SMC_meta,
    how="left",
    left_on=["patient_id"],
    right_on=["Patient"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
# Get original sample/patient meta and remove cell level meta
SMC_sample_meta = (
    adata.obs[adata.obs.columns[adata.obs.nunique() <= adata.obs["id"].nunique()]]
    .groupby("id")
    .first()
    .reset_index()
)

# %%
SMC_sample_meta.to_csv(
    f"{artifact_dir}/Lee_2020_SMC-original_sample_meta.csv",
    index=False,
)

# %% [markdown]
# ## 2. Gene annotations

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
duplicated_ids = adata.var_names[adata.var_names.duplicated()].unique()
adata = ah.pp.aggregate_duplicate_gene_ids(adata, duplicated_ids)

# %%
unmapped_genes = ah.pp.find_unmapped_genes(adata)
len(unmapped_genes)

# %% [markdown]
# ## 3. Harmonize SMC cohort metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="Lee_2020_Nat_Genet",
    study_doi="10.1038/s41588-020-0636-z",
    study_pmid="32451460",
    tissue_processing_lab="Woong-Yang Park lab",
    medical_condition="colorectal cancer",
    sample_tissue="colon",
    tumor_source="core",
    hospital_location="Samsung Genome Institute, Samsung Medical Center, Seoul",
    country="South Korea",
    platform="10x 3' v2",
    cellranger_version="2.1.0",
    reference_genome="ensembl.v84",
    enrichment_cell_types="naive",
    matrix_type="raw counts",
    treatment_status_before_resection="naive",
)

adata.obs["sample_id"] = (
    adata.obs["sample_title"]
    .str.replace(".", "-")
    .str.replace("-A1", "")
    .str.replace("SMC13N", "SMC13-N")
    .str.replace("SMC13T", "SMC13-T")
)

adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]

# define the dictionary mapping
mapping = {
    "T": "tumor",
    "N": "normal",
}
adata.obs["sample_type"] = adata.obs["sample_id"].str.split("-").str[1].map(mapping)


# define the dictionary mapping
mapping = {
    "hepatic flexure": "hepatic flexure",
    "sigmoid": "sigmoid colon",
    "rectosigmoid": "rectosigmoid junction",
    "ascending": "ascending colon",
    "rectum": "rectum",
}
adata.obs["anatomic_location"] = adata.obs["Anatomic region"].map(mapping)

mapping = {
    "hepatic flexure": "proximal colon",
    "ascending colon": "proximal colon",
    "rectum": "distal colon",
    "descending colon": "distal colon",
    "rectosigmoid junction": "distal colon",
    "sigmoid colon": "distal colon",
}
adata.obs["anatomic_region"] = adata.obs["anatomic_location"].map(
    mapping
)

mapping = {
    "F": "female",
    "M": "male",
}
adata.obs["sex"] = adata.obs["Gender"].map(mapping)
adata.obs["age"] = adata.obs["Age"].fillna(np.nan).astype("Int64")


mapping = {
    "Adenocarcinoma, well differentiated": "adenocarcinoma",
    "Adenocarcinoma, poorly differentiated": "adenocarcinoma",
    "Adenocarcinoma, moderately differentiated": "adenocarcinoma",
    "Adenocarcinoma, well differentiated with mucin production (<10%)": "adenocarcinoma",
    "Mucinous adenocarcinoma": "mucinous adenocarcinoma",
    "Adenocarcinoma, well differentiated with mucin production (10%)": "adenocarcinoma",
}
adata.obs["histological_type"] = adata.obs["Pathological subtype"].map(mapping)

mapping = {
    "Adenocarcinoma, well differentiated": "G1",
    "Adenocarcinoma, poorly differentiated": "G3",
    "Adenocarcinoma, moderately differentiated": "G2",
    "Adenocarcinoma, well differentiated with mucin production (<10%)": "G1",
    "Mucinous adenocarcinoma": "nan",
    "Adenocarcinoma, well differentiated with mucin production (10%)": "G1",
}
adata.obs["tumor_grade"] = adata.obs["Pathological subtype"].map(mapping)


adata.obs["microsatellite_status"] = adata.obs["MSI"]


adata.obs["tumor_stage_TNM_T"] = (
    adata.obs["TNM stage"]
    .astype(str)
    .apply(
        lambda x: re.search(r"(T[0-4][a-c]?)", x).group(1)
        if re.search(r"(T[0-4][a-c]?)", x)
        else None
    )
)
adata.obs["tumor_stage_TNM_N"] = (
    adata.obs["TNM stage"]
    .astype(str)
    .apply(
        lambda x: re.search(r"(N[0-4][a-c]?)", x).group(1)
        if re.search(r"(N[0-4][a-c]?)", x)
        else None
    )
)
adata.obs["tumor_stage_TNM_M"] = (
    adata.obs["TNM stage"]
    .astype(str)
    .apply(
        lambda x: re.search(r"((M[0-1]|Mx)[a-c]?)", x).group(1)
        if re.search(r"((M[0-1]|Mx)[a-c]?)", x)
        else None
    )
)

adata.obs["tumor_stage_TNM"] = ah.pp.tnm_tumor_stage(adata.obs, "patient_id")


adata.obs = adata.obs.rename(
    columns={
        "KRAS": "KRAS_status_driver_mut",
        "BRAF": "BRAF_status_driver_mut",
        "TP53": "TP53_status_driver_mut",
        "APC": "APC_status_driver_mut",
        "SMAD4": "SMAD4_status_driver_mut",
    }
)

mapping = {"Mutant": "mut", "Wildtype": "wt"}

for col in adata.obs.columns:
    if col.endswith("_status_driver_mut"):
        adata.obs[col] = adata.obs[col].replace(mapping)

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)
# Add newly created driver_mutation columns
ref_meta_cols += [col for col in adata.obs.columns if "_status_driver_mut" in col]

# %%
# Add back potentially interesting metadata that does not fit the reference meta
ref_meta_cols += [
    "No.of mutations",
    "nearestCMS (RF)",
    "predictedCMS (RF)",
]

# %%
adata.obs["No.of mutations"] = adata.obs["No.of mutations"].astype(str)

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]]

# %%
adata.obs["sample_id"] = adata.obs["sample_id"].str.replace("-", "_")

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["GEO_sample_accession"]
    + "-"
    + adata.obs_names.str.split("_").str[1].str.split("-").str[0]
)

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique
assert adata.obs.columns.nunique() == adata.obs.shape[1], "Column names are not unique."

# %%
adata

# %%
adata.obs["sample_id"].value_counts()

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
    _adata.obs = _adata.obs.dropna(axis=1, how="all")  # Remove all nan columns, else write_h5ad will fail
    dataset = _adata.obs["dataset"].values[0]  # Get the dataset name from the subset
    _adata.write_h5ad(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")

# %% [markdown]
# ## KUL3 cohort
#
# fastqs available!

# %%
E_MTAB_8410_meta = pd.read_csv(
    f"{d10_1038_s41588_020_0636_z}/fetchngs/E-MTAB-8410/E-MTAB-8410_sample_meta.csv"
)
E_MTAB_8410_meta["sample_id"] = E_MTAB_8410_meta["ENA_SAMPLE"]
samplesheet = E_MTAB_8410_meta


# %%
def load_counts(sample_meta, databases_path, mat_type="filtered"):

    file_path = f"{databases_path}/10_nfcore_scrnaseq/cellranger/count/{sample_meta['sample_id']}/outs/{mat_type}_feature_bc_matrix.h5"
    adata = sc.read_10x_h5(file_path)
    adata.obs = adata.obs.assign(**sample_meta)
    adata.var_names_make_unique()
    adata.X = csr_matrix(adata.X)
    adata.obs_names = (
        adata.obs["sample_id"] + "-" + adata.obs_names.str.replace("-1", "")
    )
    assert adata.obs_names.is_unique
    return adata


# %%
adatas = process_map(
    load_counts,
    [r for _, r in samplesheet.iterrows()],
    itertools.repeat(d10_1038_s41588_020_0636_z),
    itertools.repeat("raw"),
    max_workers=cpus,
)

adata_KUL3 = anndata.concat(adatas, join="outer")

# %% [markdown]
# ### 2. Gene annotations

# %%
# Append gtf gene info
gtf = f"{annotation}/gencode.v44_gene_annotation_table.csv"
gtf = pd.read_csv(gtf)

adata_KUL3.var.reset_index(names="symbol", inplace=True)
gtf["symbol"] = adata_KUL3.var["symbol"]
gtf["ensembl"] = gtf["Geneid"].apply(ah.pp.remove_gene_version)
gtf.set_index("ensembl", inplace=True)
adata_KUL3.var = gtf.copy()
adata_KUL3.var_names.name = None

# %%
# Sum up duplicate gene ids
duplicated_ids = adata_KUL3.var_names[adata_KUL3.var_names.duplicated()].unique()
adata_KUL3 = ah.pp.aggregate_duplicate_gene_ids(adata_KUL3, duplicated_ids)

# %%
geofetch = pd.read_csv(
    f"{d10_1038_s41588_020_0636_z}/GSE144735/GSE144735_PEP/GSE144735_PEP_raw.csv"
).dropna(axis=1, how="all")

# %%
adata_KUL3.obs["obs_names"] = adata_KUL3.obs_names
adata_KUL3.obs = pd.merge(
    adata_KUL3.obs,
    geofetch,
    how="left",
    left_on=["sample"],
    right_on=["sample_title"],
    validate="m:1",
).set_index("obs_names")
adata_KUL3.obs_names.name = None

# %%
adata_KUL3.obs = adata_KUL3.obs.rename(columns={"sample_id": "ENA_sample_accession"})

# %% [markdown]
# ### Compile original meta from study supplements

# %%
adata_KUL3.obs["patient_id"] = adata_KUL3.obs["patient_id"].fillna(
    adata_KUL3.obs["individual"]
)

# %%
# Get meta for 3 missing samples from subsequent study Joanito_2022_Nat_Genet
m = meta.loc[meta["patient.ID"].isin(adata_KUL3.obs["patient_id"])]
m = m.rename(
    columns={
        "patient.ID": "Patient",
        "Stage TNM": "TNM stage",
        "Age at recruitment": "Age",
        "Site": "Anatomic region",
        "Sidedness": "Left/Right-sided",
        "MSS/MSI": "MSI",
    }
)

# Add new columns to the DataFrame
for col in [
    "Stage",
    "Pathological subtype",
    "nearestCMS (SSP) in Core samples",
    "nearestCMS (SSP) in Border samples",
    "SMAD4",
]:
    if col not in m.columns:
        m[col] = None

m = m[
    [
        "Patient",
        "Gender",
        "Age",
        "TNM stage",
        "Stage",
        "Anatomic region",
        "Left/Right-sided",
        "MSI",
        "Pathological subtype",
        "nearestCMS (SSP) in Core samples",
        "nearestCMS (SSP) in Border samples",
        "TP53",
        "APC",
        "KRAS",
        "BRAF",
        "SMAD4",
    ]
].copy()

# %%
KUL3_meta = KUL3_meta[
    [
        "Patient",
        "Gender",
        "Age",
        "TNM stage",
        "Stage",
        "Anatomic region",
        "Left/Right-sided",
        "MSI",
        "Pathological subtype",
        "nearestCMS (SSP) in Core samples",
        "nearestCMS (SSP) in Border samples",
        "TP53",
        "APC",
        "KRAS",
        "BRAF",
        "SMAD4",
    ]
].copy()

KUL3_meta = pd.concat([KUL3_meta, m], ignore_index=True)

# %%
adata_KUL3.obs["obs_names"] = adata_KUL3.obs_names
adata_KUL3.obs = pd.merge(
    adata_KUL3.obs,
    KUL3_meta,
    how="left",
    left_on=["patient_id"],
    right_on=["Patient"],
    validate="m:1",
).set_index("obs_names")
adata_KUL3.obs_names.name = None

# %%
# Get original sample/patient meta and remove cell level meta
KUL3_sample_meta = (
    adata_KUL3.obs[
        adata_KUL3.obs.columns[
            adata_KUL3.obs.nunique() <= adata_KUL3.obs["specimen"].nunique()
        ]
    ]
    .groupby("specimen")
    .first()
    .reset_index()
)

# %%
KUL3_sample_meta.to_csv(
    f"{artifact_dir}/Lee_2020_KUL3-original_sample_meta.csv",
    index=False,
)

# %% [markdown]
# ## 3. Harmonize KUL3 cohort metadata to ref_meta_dict

# %%
adata_KUL3.obs = adata_KUL3.obs.assign(
    study_id="Lee_2020_Nat_Genet",
    study_doi="10.1038/s41588-020-0636-z",
    study_pmid="32451460",
    tissue_processing_lab="Sabine Tejpar lab",
    dataset="Lee_2020_KUL3",
    medical_condition="colorectal cancer",
    NCBI_BioProject_accession="PRJNA604751",
    sample_tissue="colon",
    treatment_status_before_resection="naive",
    hospital_location="Department of Oncology, Katholieke Universiteit Leuven",
    country="Belgium",
    tissue_cell_state="fresh",
    cellranger_version="7.1.0",
    reference_genome="gencode.v44",
    matrix_type="raw counts",
    enrichment_cell_types="naive",
    histological_type="adenocarcinoma",
)

adata_KUL3.obs["GEO_sample_accession"] = adata_KUL3.obs["sample_geo_accession"]

# define the dictionary mapping
mapping = {
    "neoplasm": "T",
    "normal tissue adjacent to neoplasm": "N",
}
adata_KUL3.obs["sample_type"] = adata_KUL3.obs["sampling site"].map(mapping)

adata_KUL3.obs.loc[
    adata_KUL3.obs["specimen"].isin(
        [
            "EXT002",
            "EXT010",
            "EXT013",
            "EXT016",
            "EXT019",
            "EXT022",
            "EXT025",
            "EXT028",
            "EXT031",
        ]
    ),
    "sample_type",
] = "B"

adata_KUL3.obs["individual"] = (
    adata_KUL3.obs["individual"] + "-" + adata_KUL3.obs["sample_type"]
)

adata_KUL3.obs["sample_id"] = adata_KUL3.obs["sample"].fillna(
    adata_KUL3.obs["individual"]
)

mapping = {
    "B": "border",
    "T": "core",
    "N": "normal",
}
adata_KUL3.obs["tumor_source"] = adata_KUL3.obs["sample_type"].map(mapping)

mapping = {
    "neoplasm": "tumor",
    "normal tissue adjacent to neoplasm": "normal",
}
adata_KUL3.obs["sample_type"] = adata_KUL3.obs["sampling site"].map(mapping)

# %%
# define the dictionary mapping
mapping = {
    "Sigmoid colon": "sigmoid colon",
    "sigmoid": "sigmoid colon",
    "rectosigmoid": "rectosigmoid junction",
    "ascending": "ascending colon",
    "Caecum": "cecum",
    "caecum": "cecum",
}
adata_KUL3.obs["anatomic_location"] = adata_KUL3.obs["Anatomic region"].map(
    mapping
)

mapping = {
    "cecum": "proximal colon",
    "ascending colon": "proximal colon",
    "sigmoid colon": "distal colon",
    "rectosigmoid junction": "distal colon",
}
adata_KUL3.obs["anatomic_region"] = adata_KUL3.obs[
    "anatomic_location"
].map(mapping)

mapping = {
    "F": "female",
    "M": "male",
}
adata_KUL3.obs["sex"] = adata_KUL3.obs["Gender"].map(mapping)
adata_KUL3.obs["age"] = adata_KUL3.obs["Age"].fillna(np.nan).astype("Int64")

# %%
adata_KUL3.obs["microsatellite_status"] = adata_KUL3.obs["MSI"]


adata_KUL3.obs["tumor_stage_TNM_T"] = (
    adata_KUL3.obs["TNM stage"]
    .astype(str)
    .apply(
        lambda x: re.search(r"(T[0-4][a-c]?)", x).group(1)
        if re.search(r"(T[0-4][a-c]?)", x)
        else None
    )
)
adata_KUL3.obs["tumor_stage_TNM_N"] = (
    adata_KUL3.obs["TNM stage"]
    .astype(str)
    .apply(
        lambda x: re.search(r"(N[0-4][a-c]?)", x).group(1)
        if re.search(r"(N[0-4][a-c]?)", x)
        else None
    )
)
adata_KUL3.obs["tumor_stage_TNM_M"] = (
    adata_KUL3.obs["TNM stage"]
    .astype(str)
    .apply(
        lambda x: re.search(r"((M[0-1]|Mx)[a-c]?)", x).group(1)
        if re.search(r"((M[0-1]|Mx)[a-c]?)", x)
        else None
    )
)

# Switch sample id
adata_KUL3.obs["patient_id"] = adata_KUL3.obs["individual"]

# ! Additional patient meta from Borras_2023_Cell_Discov (PMID: 37968259)
mapping = {
    "SC001": "G2",
    "SC019": "G2",
    "SC021": "G2",
    "SC031": "G1",
}
adata_KUL3.obs["tumor_grade"] = adata_KUL3.obs["patient_id"].map(mapping)
adata_KUL3.obs["tumor_grade"] = adata_KUL3.obs["tumor_grade"].astype(str)

adata_KUL3.obs.loc[
    adata_KUL3.obs["patient_id"].isin(["SC001"]), "histological_type"
] = "mucinous adenocarcinoma" # 'Global_moderately_differentiated_adenocarcinoma_with_mixed_glandulair_mucinous_growth_patern_moderate_budding'

adata_KUL3.obs.loc[
    adata_KUL3.obs["patient_id"].isin(["SC001", "SC019", "SC031"]), "tumor_stage_TNM_M"
] = adata_KUL3.obs["tumor_stage_TNM_M"].fillna("M0")

# Replace None values in tumor_stage_TNM_M column with 'Mx'
adata_KUL3.obs["tumor_stage_TNM_M"] = (
    adata_KUL3.obs["tumor_stage_TNM_M"].astype(str).replace("None", "Mx")
)

# Matches "Stage" column from original meta
adata_KUL3.obs["tumor_stage_TNM"] = ah.pp.tnm_tumor_stage(adata_KUL3.obs, "patient_id")

adata_KUL3.obs["platform"] = adata_KUL3.obs["library construction"]


adata_KUL3.obs = adata_KUL3.obs.rename(
    columns={
        "KRAS": "KRAS_status_driver_mut",
        "BRAF": "BRAF_status_driver_mut",
        "TP53": "TP53_status_driver_mut",
        "APC": "APC_status_driver_mut",
        "SMAD4": "SMAD4_status_driver_mut",
    }
)

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)
# Add newly created driver_mutation columns
ref_meta_cols += [col for col in adata_KUL3.obs.columns if "_status_driver_mut" in col]

# %%
# Add back potentially interesting metadata that does not fit the reference meta
ref_meta_cols += [
    "nearestCMS (SSP) in Core samples",
    "nearestCMS (SSP) in Border samples",
]

# %%
# Subset adata for columns in the reference meta
adata_KUL3.obs = adata_KUL3.obs[
    [col for col in ref_meta_cols if col in adata_KUL3.obs.columns]
]

# %%
adata_KUL3.obs["sample_id"] = adata_KUL3.obs["sample_id"].str.replace("-", "_")
adata_KUL3.obs["patient_id"] = adata_KUL3.obs["patient_id"].str.split('-').str[0]

# %%
adata_KUL3.obs_names = (
    adata_KUL3.obs["dataset"]
    + "-"
    + adata_KUL3.obs_names
)

# %%
assert np.all(np.modf(adata_KUL3.X.data)[0] == 0), "X does not contain all integers"
assert adata_KUL3.var_names.is_unique
assert adata_KUL3.obs_names.is_unique
assert (
    adata_KUL3.obs.columns.nunique() == adata_KUL3.obs.shape[1]
), "Column names are not unique."

# %%
adata_KUL3

# %%
# Save unfiltered raw counts
dataset = adata_KUL3.obs["dataset"].unique()[0]
adata_KUL3.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
