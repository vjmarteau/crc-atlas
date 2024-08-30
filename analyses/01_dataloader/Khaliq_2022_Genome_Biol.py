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
# # Dataloader: Khaliq_2022_Genome_Biol

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
cpus = nxfvars.get("cpus", 6)
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")

# %%
d10_1186_s13059_022_02677_z = f"{databases_path}/d10_1186_s13059_022_02677_z"
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
# ## 1. Load raw adata counts

# %%
# Loading this as csv takes ages ...
# counts_mat = pd.read_csv(
#    Path(f"{d10_1186_s13059_022_02677_z}/zenodo/masoodlab-CRC-Single-Cell-1c6e07c/Data/CRC_Raw_counts.csv.gz")
# )
# mat = counts_mat.T
# mat.columns = mat.iloc[0]
# mat = mat[1:]
# mat = mat.rename_axis(None, axis=1)

# mat = np.ceil(mat).astype(int)
# adata = sc.AnnData(mat)
# adata.X = csr_matrix(adata.X)

# fmm.mmwrite(f"{d10_1186_s13059_022_02677_z}/zenodo/masoodlab-CRC-Single-Cell-1c6e07c/Data/raw.mat", adata.X) # seems to write as scipy.sparse._coo.coo_matrix
# pd.DataFrame(adata.obs_names, columns=["barcodes"]).to_csv(f"{d10_1186_s13059_022_02677_z}/zenodo/masoodlab-CRC-Single-Cell-1c6e07c/Data/barcodes.csv", index=False)
# pd.DataFrame(adata.var_names, columns=["features"]).to_csv(f"{d10_1186_s13059_022_02677_z}/zenodo/masoodlab-CRC-Single-Cell-1c6e07c/Data/features.csv", index=False)

# %%
adata = sc.AnnData(
    X=fmm.mmread(
        f"{d10_1186_s13059_022_02677_z}/zenodo/masoodlab-CRC-Single-Cell-1c6e07c/Data/CRC_Raw_counts.mat.gz"
    ),
    obs=pd.read_csv(
        f"{d10_1186_s13059_022_02677_z}/zenodo/masoodlab-CRC-Single-Cell-1c6e07c/Data/barcodes.csv.gz",
        index_col=0,
    ),
    var=pd.read_csv(
        f"{d10_1186_s13059_022_02677_z}/zenodo/masoodlab-CRC-Single-Cell-1c6e07c/Data/features.csv.gz",
        index_col=0,
    ),
)
adata.X = csr_matrix(adata.X) # convert coo_matrix back to csr_matrix
adata.var_names.name = None
adata.obs_names.name = None

# %%
meta = pd.read_csv(
    f"{d10_1186_s13059_022_02677_z}/zenodo/masoodlab-CRC-Single-Cell-1c6e07c/Data/CRC_Metadata.csv"
).rename({"Unnamed: 0": "obs_names"}, axis=1)

adata.obs = adata.obs.reset_index(names="obs_names")
adata.obs = pd.merge(
    adata.obs,
    meta,
    how="left",
    on=["obs_names"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
adata.obs["sample_id"] = adata.obs["samples"].str.upper()
adata.obs["patient_id"] = adata.obs["sample_id"].str.split("_").str[1]

# %%
patient_meta = pd.read_csv(f"{d10_1186_s13059_022_02677_z}/TableS1.csv")
patient_meta["patient_id"] = patient_meta["SAMPLE_ID"].str.split("_").str[1]

adata.obs = adata.obs.reset_index(names="obs_names")
adata.obs = pd.merge(
    adata.obs,
    patient_meta,
    how="left",
    on=["patient_id"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
geofetch = pd.read_csv(
    f"{d10_1186_s13059_022_02677_z}/GSE200997/GSE200997_PEP/GSE200997_PEP_raw.csv"
).dropna(axis=1, how="all")

# get sample ids from study
mapping = {
    "tumor": "T",
    "normal": "B",
}
geofetch["type"] = geofetch["sample_name"].str.split("_").str[2].map(mapping)

geofetch["sample_id"] = (
    geofetch["type"] + "_" + "CAC" + geofetch["sample_name"].str.split("_").str[1]
)

# %%
adata.obs = adata.obs.reset_index(names="obs_names")
adata.obs = pd.merge(
    adata.obs,
    geofetch,
    how="left",
    on=["sample_id"],
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
    f"{artifact_dir}/Khaliq_2022-original_sample_meta.csv",
    index=False,
)

# %% [markdown]
# ## 2. Gene annotations

# %%
# load gtf for gene mapping
gtf = pd.read_csv(gtf)
gtf = ah.pp.append_duplicate_suffix(df=gtf, column="gene_name", sep=".")
gene_ids = gtf.set_index("gene_name")["gene_id"].to_dict()

adata.var = adata.var.rename_axis("symbol").reset_index()
adata.var["ensembl"] = (
    adata.var["symbol"].map(gene_ids).fillna(value=adata.var["symbol"])
)
adata.var_names = adata.var["ensembl"].apply(ah.pp.remove_gene_version)
adata.var_names.name = None

unmapped_genes = ah.pp.find_unmapped_genes(adata)
len(unmapped_genes)

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="Khaliq_2022_Genome_Biol",
    study_doi="10.1186/s13059-022-02677-z",
    study_pmid="35538548",
    tissue_processing_lab="Ashiq Masood lab",
    dataset="Khaliq_2022",
    medical_condition="colorectal cancer",
    tissue_cell_state="fresh",
    enrichment_cell_types="naive",
    sample_tissue="colon",
    # Paper: we performed droplet-based scRNA-seq on 16 racially diverse, treatment naïve CRC patient tissue samples and seven adjacent normal colonic tissue samples
    treatment_status_before_resection="naive",
    platform="10x 5' v2",
    cellranger_version="3.1.0",
    reference_genome="ensembl.v93",
    matrix_type="raw counts",
    hospital_location="Rush University Medical Center, Chicago, Illinois",
    country="US",
    NCBI_BioProject_accession="PRJNA827831",
)

# %%
adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]
adata.obs["microsatellite_status"] = adata.obs["MSI Status"]
adata.obs["tumor_mutational_burden"] = adata.obs["TMB(m/MB)"]

# %%
adata.obs["age"] = adata.obs["Age"].fillna(np.nan).astype("Int64")
# define the dictionary mapping
mapping = {
    "Female": "female",
    "Male": "male",
}
adata.obs["sex"] = adata.obs["Gender"].map(mapping)

# define the dictionary mapping
mapping = {
    "B": "normal",
    "T": "tumor",
}
adata.obs["sample_type"] = adata.obs["type"].map(mapping)

# define the dictionary mapping
mapping = {
    "Cecum": "cecum",
    "Cecum/IC": "cecum",
    "Cecum/Appendix": "cecum",
    "Sigmoid": "sigmoid colon",
    "Hepatic Flexure": "hepatic flexure",
    "Ascending": "ascending colon",
    "Rectosigmoid": "rectosigmoid junction",
    "Descending": "descending colon",
}
adata.obs["anatomic_location"] = adata.obs["Location2"].map(mapping)

mapping = {
    "descending colon": "distal colon",
    "ascending colon": "proximal colon",
    "sigmoid colon": "distal colon",
    "cecum": "proximal colon",
    "rectosigmoid junction": "distal colon",
    "hepatic flexure": "proximal colon",
}
adata.obs["anatomic_region"] = adata.obs["anatomic_location"].map(mapping)

mapping = {
    "Moderately differentiated(Mucinous features)": "G2",
    "Moderately differentiated": "G2",
    "Adenocarcinoma insitu": "nan",
    "Poorly differentiated": "G3",
}
adata.obs["tumor_grade"] = adata.obs["Pathological subtype"].map(mapping)

mapping = {
    "Moderately differentiated(Mucinous features)": "mucinous adenocarcinoma",
    "Moderately differentiated": "nan",
    "Adenocarcinoma insitu": "adenocarcinoma",
    "Poorly differentiated": "nan",
}
adata.obs["histological_type"] = adata.obs["Pathological subtype"].map(mapping)

mapping = {
    "Hispanic": "hispanic",
    "Caucasian": "caucasian",
    "African American": "black or african american",
    "Other": "other",
    "Asian": "asian",
}
adata.obs["ethnicity"] = adata.obs["Race"].map(mapping)

adata.obs["tumor_stage_TNM_T"] = adata.obs["Stage"].apply(
    lambda x: re.search(r"(T[0-4][a-c]?)", x).group(1)
    if re.search(r"(T[0-4][a-c]?)", x)
    else None
)
adata.obs["tumor_stage_TNM_N"] = adata.obs["Stage"].apply(
    lambda x: re.search(r"(N[0-4][a-c]?)", x).group(1)
    if re.search(r"(N[0-4][a-c]?)", x)
    else None
)
adata.obs["tumor_stage_TNM_M"] = adata.obs["Stage"].apply(
    lambda x: re.search(r"((M[0-1]|Mx)[a-c]?)", x).group(1)
    if re.search(r"((M[0-1]|Mx)[a-c]?)", x)
    else None
)

# Replace None values in tumor_stage_TNM_M column with 'Mx'
adata.obs["tumor_stage_TNM_M"] = (
    adata.obs["tumor_stage_TNM_M"].astype(str).replace("None", "Mx")
)

adata.obs["tumor_stage_TNM"] = ah.pp.tnm_tumor_stage(adata.obs, "patient_id")

# %%
mutation_list = ["TP53", "APC", "KRAS", "BRAS", "NRAS", "PiK3CA", "ERBB2", "POLE"]
# List comprehension to create new columns for each unique mutation
mapping = {
    "Wildtype": "wt",
    "Mutant": "mut",
    "Copy Number Gain": "mut",
    "NaN": "nan",
}
for mutation in mutation_list:
    adata.obs[f"{mutation}_status_driver_mut"] = adata.obs[mutation].map(mapping)

# %%
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
# Add newly created driver_mutation columns
ref_meta_cols += [f"{mutation}_status_driver_mut" for mutation in mutation_list]

# %%
# Add back potentially interesting metadata that does not fit the reference meta
ref_meta_cols += [
    "tumor_mutational_burden",
]

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["GEO_sample_accession"]
    + "-"
    + adata.obs_names.str.split("_").str[2]
)

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
adata.obs["sample_id"].value_counts()

# %%
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
