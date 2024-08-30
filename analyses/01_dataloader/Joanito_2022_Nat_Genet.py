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
# # Dataloader: Joanito_2022_Nat_Genet

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
cpus = nxfvars.get("cpus", 2)
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")

# %%
d10_1038_s41588_022_01100_4 = f"{databases_path}/d10_1038_s41588_022_01100_4"
gtf = f"{annotation}/ensembl.v93_gene_annotation_table.csv"
SG_singlecell = f"{d10_1038_s41588_022_01100_4}/SG_singlecell"

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
def load_adata_from_mtx(mtx_path, barcodes_path, features_path, metadata_path):
    # Load data into memory
    matrix = fmm.mmread(mtx_path)
    barcodes = pd.read_csv(
        barcodes_path, delimiter="\t", header=None, names=["barcodes"]
    )
    features = pd.read_csv(
        features_path, delimiter="\t", header=None, names=["var_names"]
    )
    meta = pd.read_csv(metadata_path)

    barcodes = pd.merge(
        barcodes,
        meta,
        how="left",
        left_on=["barcodes"],
        right_on=["cell.ID"],
        validate="m:1",
    ).set_index("barcodes")
    barcodes.index.name = None

    # Assemble adata
    adata = sc.AnnData(X=matrix.T)
    adata.X = csr_matrix(adata.X.astype(int))
    adata.var = features
    adata.obs = barcodes
    adata.obs["original_obs_names"] = adata.obs_names
    adata.var_names = adata.var["var_names"]
    adata.var_names.name = None
    return adata


# %%
dataset_names = ["Epithelial", "NonEpithelial"]
adatas = []
for dataset_name in dataset_names:
    mtx_path = f"{SG_singlecell}/seurat/{dataset_name}_Count_matrix.mtx.gz"
    barcodes_path = f"{SG_singlecell}/seurat/{dataset_name}_Count_barcodes.tsv"
    features_path = f"{SG_singlecell}/seurat/{dataset_name}_Count_features.tsv"
    metadata_path = f"{SG_singlecell}/{dataset_name}_metadata.csv"
    adata = load_adata_from_mtx(mtx_path, barcodes_path, features_path, metadata_path)
    adatas.append(adata)
epithelial_adata, non_epithelial_adata = adatas

# %%
adata = anndata.concat(
    adatas,
    index_unique="-",
    join="outer",
    fill_value=0,
)
adata.obs_names = adata.obs_names.str.split("-").str[0]
adata.obs_names_make_unique()

# %% [markdown]
# ### 1a. Gene annotations

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
# ### 1b. Compile sample/patient metadata

# %%
meta = pd.read_csv(
    f"{SG_singlecell}/patient_clinical_information.csv", encoding="ISO-8859-1"
)
# Additional patient meta from Borras_2023_Cell_Discov (PMID: 37968259)
patient_meta = pd.read_csv(
    f"{SG_singlecell}/patient_meta_Borras_2023_Cell_Discov.csv")

merged_meta = pd.merge(
    meta,
    patient_meta,
    how="left",
    left_on=["patient.ID"],
    right_on=["Patient"],
    validate="m:1",
)

# %%
adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    merged_meta,
    how="left",
    on=["patient.ID"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
adata.obs = adata.obs.rename(
    columns={
        "dataset_x": "dataset",
    }
)
del adata.obs["dataset_y"]

# %%
# Get original sample/patient meta and remove cell level meta
sample_meta = (
    adata.obs[
        adata.obs.columns[adata.obs.nunique() <= adata.obs["sample.ID"].nunique()]
    ]
    .groupby("sample.ID")
    .first()
    .reset_index()
)

# %%
sample_meta.to_csv(f"{artifact_dir}/Joanito_2022-original_sample_meta.csv", index=False)

# %% [markdown] tags=[]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
# define the dictionary mapping
mapping = {
    "CRC-SG1": "Joanito_2022_Nat_Genet",
    "CRC-SG2": "Joanito_2022_Nat_Genet",
    "KUL3": "Lee_2020_Nat_Genet",
    "KUL5": "Joanito_2022_Nat_Genet",
    "SMC": "Lee_2020_Nat_Genet",
}
adata.obs["study_id"] = adata.obs["dataset"].map(mapping)

# define the dictionary mapping
mapping = {
    "CRC-SG1": "10.1038/s41588-022-01100-4",
    "CRC-SG2": "10.1038/s41588-022-01100-4",
    "KUL3": "10.1038/s41588-020-0636-z",
    "KUL5": "10.1038/s41588-022-01100-4",
    "SMC": "10.1038/s41588-020-0636-z",
}
adata.obs["study_doi"] = adata.obs["dataset"].map(mapping)

# define the dictionary mapping
mapping = {
    "CRC-SG1": "35773407",
    "CRC-SG2": "35773407",
    "KUL3": "32451460",
    "KUL5": "35773407",
    "SMC": "32451460",
}
adata.obs["study_pmid"] = adata.obs["dataset"].map(mapping)

adata.obs = adata.obs.rename(
    columns={
        "sample.ID": "sample_id",
        "patient.ID": "patient_id",
        "cell.type": "cell_type_study",
        "Age at recruitment": "age",
        "MSS/MSI": "microsatellite_status",
        "KRAS": "KRAS_status_driver_mut",
        "BRAF": "BRAF_status_driver_mut",
        "TP53": "TP53_status_driver_mut",
        "APC": "APC_status_driver_mut",
        "PIK3CA": "PIK3CA_status_driver_mut",
    }
)

adata.obs["age"] = adata.obs["age"].fillna(np.nan).astype("Int64")

# Paper methods: For both CRC-SG1 and CRC-SG2 samples, tissue specimens were processed similarly to the KUL samples.
# Does this mean they sampled tumor core + border seperately? Then what is Tumor and Tumor-2?
# From the KUL3 cohort I know that "Tumor" is tumor core and "Tumor-2" is tumor border -> will go with that!

mapping = {
    "Normal": "normal",
    "LymphNode": "nan",
    "Tumor": "core",
    "Tumor-2": "border",
}
adata.obs["tumor_source"] = adata.obs["sample.origin"].map(mapping)

# define the dictionary mapping
mapping = {
    "Normal": "normal",
    "LymphNode": "lymph node",
    "Tumor": "tumor",
    "Tumor-2": "tumor",
}
adata.obs["sample_type"] = adata.obs["sample.origin"].map(mapping)

adata.obs["sample_matched"] = ah.pp.add_matched_samples_column(adata.obs, "patient_id")

# define the dictionary mapping
mapping = {
    "tumor": "colon",
    "normal": "colon",
    "lymph node": "lymph node",
}
adata.obs["sample_tissue"] = adata.obs["sample_type"].map(mapping)

# define the dictionary mapping
mapping = {
    "F": "female",
    "M": "male",
}
adata.obs["sex"] = adata.obs["Gender"].map(mapping)

# define the dictionary mapping
mapping = {
    "CRC-SG1": "10x 5' v3",
    "CRC-SG2": "10x 3' v3",
    "KUL3": "10x 3' v2",
    "KUL5": "10x 5' v3",
    "SMC": "10x 3' v2",
}
adata.obs["platform"] = adata.obs["dataset"].map(mapping)

# define the dictionary mapping
mapping = {
    "L": "distal colon",
    "R": "proximal colon",
}
adata.obs["anatomic_region"] = adata.obs["Sidedness"].map(mapping)

# define the dictionary mapping
mapping = {
    "Descending colon": "descending colon",
    "Rectum": "rectum",
    "Distal Descending colon": "descending colon",
    "Low rectum": "rectum",
    "Sigmoid colon": "sigmoid colon",
    "Upper rectum": "rectum",
    "Caecum": "cecum",
    "Ascending colon": "ascending colon",
    "Transverse colon": "transverse colon",
    "Mid-rectum": "rectum",
    "Distal Sigmoid colon": "sigmoid colon",
    "Rectosigmoid": "rectosigmoid junction",
    "Distal Ascending colon": "ascending colon",
    "Hepatic Flexure": "hepatic flexure",
}
adata.obs["anatomic_location"] = adata.obs["Site"].map(mapping)

# define the dictionary mapping
mapping = {
    "nan": "nan",
    "Adenocarcinoma_moderately_differentiated": "adenocarcinoma",
    "Sigmoid.Adenocarcinoma_moderately_differentiated": "adenocarcinoma",
    "Adenocarcinoma_poorly_differentiated": "adenocarcinoma",
    "Mucinous": "mucinous adenocarcinoma",
    "moderately_differentiated": "nan",
    "Adenocarcinoma_well_differentiated": "adenocarcinoma",
    "Mucinous_adenocarcinoma": "mucinous adenocarcinoma",
    "Global_moderately_differentiated_adenocarcinoma_with_mixed_glandulair_mucinous_growth_patern_moderate_budding": "mucinous adenocarcinoma",
    "Moderately_differentiated_adenocarcinoma_NST": "adenocarcinoma",
    "Moderately_differentiated_adenocarcinoma": "adenocarcinoma",
    "Well_differentiated_adenocarcinoma_NST": "adenocarcinoma",
    "Low_differentiated_adenocarcinoma": "adenocarcinoma",
}
adata.obs["histological_type"] = adata.obs["histo"].astype(str).map(mapping)

# define the dictionary mapping
mapping = {
    "nan": "nan",
    "Adenocarcinoma_moderately_differentiated": "G2",
    "Sigmoid.Adenocarcinoma_moderately_differentiated": "G2",
    "Adenocarcinoma_poorly_differentiated": "G3",
    "Mucinous": "nan",
    "moderately_differentiated": "G2",
    "Adenocarcinoma_well_differentiated": "G1",
    "Mucinous_adenocarcinoma": "nan",
    "Global_moderately_differentiated_adenocarcinoma_with_mixed_glandulair_mucinous_growth_patern_moderate_budding": "G2",
    "Moderately_differentiated_adenocarcinoma_NST": "G2",
    "Moderately_differentiated_adenocarcinoma": "G2",
    "Well_differentiated_adenocarcinoma_NST": "G1",
    "Low_differentiated_adenocarcinoma": "G1",
}
adata.obs["tumor_grade"] = adata.obs["histo"].astype(str).map(mapping)


adata.obs["tumor_stage_TNM_T"] = adata.obs["Stage TNM"].apply(
    lambda x: re.search(r"(T[0-4][a-c]?)", x).group(1)
    if re.search(r"(T[0-4][a-c]?)", x)
    else None
)
adata.obs["tumor_stage_TNM_N"] = adata.obs["Stage TNM"].apply(
    lambda x: re.search(r"(N[0-4][a-c]?)", x).group(1)
    if re.search(r"(N[0-4][a-c]?)", x)
    else None
)
adata.obs["tumor_stage_TNM_M"] = adata.obs["Stage TNM"].apply(
    lambda x: re.search(r"((M[0-1]|Mx)[a-c]?)", x).group(1)
    if re.search(r"((M[0-1]|Mx)[a-c]?)", x)
    else None
)

# Fill nan values in tumor_stage_TNM_M column with M0 if N0, I think that is a reasonable assumption -> Turns out its not ...
# adata.obs.loc[adata.obs["tumor_stage_TNM_N"] == "N0", "tumor_stage_TNM_M"] = adata.obs[
#    "tumor_stage_TNM_M"
# ].fillna("M0")

# Replace None values in tumor_stage_TNM_M column with 'Mx'
adata.obs["tumor_stage_TNM_M"] = (
    adata.obs["tumor_stage_TNM_M"].astype(str).replace("None", "Mx")
)

adata.obs["tumor_stage_TNM"] = ah.pp.tnm_tumor_stage(adata.obs, "patient_id")

# In case function ah.pp.tnm_tumor_stage returns empty values use the provided ones
# adata.obs["tumor_stage_TNM"].fillna(adata.obs["Group Stage"], inplace=True)

adata.obs = adata.obs.assign(
    medical_condition="colorectal cancer",
    tissue_cell_state="fresh",
    enrichment_cell_types="naive",
    matrix_type="raw counts",
    reference_genome="ensembl.v93",
    cellranger_version="3.1.0",
    treatment_status_before_resection="naive",
)

# define the dictionary mapping
mapping = {
    "CRC-SG1": "Iain Beehuat Tan lab",
    "CRC-SG2": "Iain Beehuat Tan lab",
    "KUL3": "Sabine Tejpar lab",
    "KUL5": "Sabine Tejpar lab",
    "SMC": "Woong-Yang Park lab",
}
adata.obs["tissue_processing_lab"] = adata.obs["dataset"].map(mapping)

# define the dictionary mapping
mapping = {
    "CRC-SG1": "Singapore General Hospital",
    "CRC-SG2": "Singapore General Hospital",
    "KUL3": "Department of Oncology, Katholieke Universiteit Leuven",
    "KUL5": "Department of Oncology, Katholieke Universiteit Leuven",
    "SMC": "Samsung Genome Institute, Samsung Medical Center, Seoul",
}
adata.obs["hospital_location"] = adata.obs["dataset"].map(mapping)

# define the dictionary mapping
mapping = {
    "CRC-SG1": "Singapore",
    "CRC-SG2": "Singapore",
    "KUL3": "Belgium",
    "KUL5": "Belgium",
    "SMC": "South Korea",
}
adata.obs["country"] = adata.obs["dataset"].map(mapping)

# define the dictionary mapping
mapping = {
    "CRC-SG1": "Joanito_2022_SG1",
    "CRC-SG2": "Joanito_2022_SG2",
    "KUL3": "Lee_2020_KUL3",
    "KUL5": "Joanito_2022_KUL5",
    "SMC": "Lee_2020_SMC",
}
adata.obs["dataset"] = adata.obs["dataset"].map(mapping)

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
ref_meta_cols += [col for col in adata.obs.columns if "_status_driver_mut" in col]

# %%
ref_meta_cols += ["CMS"]

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
adata.obs["sample_id"] = adata.obs["sample_id"].str.replace("-", "_")

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["sample_id"]
    + "-"
    + adata.obs_names.str.rsplit("_", n=1).str[1]
)

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
adata

# %%
# Subset for new data from this study. Lee data is loaded in a seperate dataloader (see: Lee_2020_Nat_Genet)
adata = adata[
    adata.obs["study_id"] == "Joanito_2022_Nat_Genet",
]

# %%
adata

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
