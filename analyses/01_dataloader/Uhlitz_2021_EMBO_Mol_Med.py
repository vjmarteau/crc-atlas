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
# # Dataloader: Uhlitz_2021_EMBO_Mol_Med

# %%
import os
import re
from pathlib import Path

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
d10_15252_emmm_202114123 = f"{databases_path}/d10_15252_emmm_202114123"
gtf = f"{annotation}/ensembl.v93_gene_annotation_table.csv"
GSE166555 = f"{d10_15252_emmm_202114123}/GSE166555"

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
def _load_mtx(stem):
    adata = sc.read_csv(Path(f"{GSE166555}/GSE166555_RAW/{stem}.gz"), delimiter="\t")
    adata = adata.T
    adata.X = csr_matrix(adata.X)
    return adata


# %%
files = [x.stem for x in Path(f"{GSE166555}/GSE166555_RAW/").glob("*.tsv.gz")]
files.sort()
adatas = process_map(_load_mtx, files, max_workers=cpus)

# %%
adata = anndata.concat(adatas, index_unique="-", join="outer", fill_value=0)

# %% [markdown]
# ### 1a. Gene annotations
# Get ensembl ids without version numbers as var_names and aggregate counts of identical ids (e.g. _PAR_Y genes)

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
# Sum up duplicate gene ids
duplicated_ids = adata.var_names[adata.var_names.duplicated()].unique()
adata = ah.pp.aggregate_duplicate_gene_ids(adata, duplicated_ids)

# %% [markdown]
# ### 1b. Compile sample/patient metadata

# %%
# Seperate meta in obs_names to individual columns -> needed for mapping additional metadata
adata.obs["sample_id"] = adata.obs_names.str.split(":").str[0]
adata.obs_names = adata.obs_names.str.split(":").str[1]

adata.obs["Patient ID"] = adata.obs["sample_id"].apply(
    lambda x: re.sub(r"[tn]\d*", "", x)
)
adata.obs["Patient ID"] = adata.obs["Patient ID"].str.upper()


adata.obs["tissue"] = adata.obs["sample_id"].apply(lambda x: re.sub(r"[^nt]", "", x))

mapping = {
    "n": "normal",
    "t": "tumor",
}
adata.obs["tissue"] = adata.obs["tissue"].map(mapping)

adata.obs["replicate"] = adata.obs["sample_id"].str.split(r"[nt]", expand=True)[1]

# %%
patient_metadata = pd.read_csv(
    f"{d10_15252_emmm_202114123}/metadata/emmm202114123-sup-0003-tableev1.csv",
    encoding="utf-8",
    header=1,
    sep=",",
    quotechar='"',
    na_values="?",
)

# %%
adata.obs = adata.obs.reset_index(names="obs_names")
adata.obs = pd.merge(
    adata.obs,
    patient_metadata,
    how="left",
    on=["Patient ID"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
geofetch = pd.read_csv(f"{GSE166555}/GSE166555_PEP/GSE166555_PEP_raw.csv").dropna(
    axis=1, how="all"
)

# %%
adata.obs["sample_name"] = (
    adata.obs["Patient ID"].str.replace("P", "patient")
    + "_"
    + adata.obs["tissue"]
    + "_"
    + adata.obs["replicate"]
)
adata.obs["sample_name"] = adata.obs["sample_name"].str.strip("_")

adata.obs = adata.obs.reset_index(names="obs_names")
adata.obs = pd.merge(
    adata.obs,
    geofetch,
    how="left",
    on=["sample_name"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
adata.obs = adata.obs.assign(NCBI_BioProject_accession="PRJNA701264")

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
sample_meta.to_csv(f"{artifact_dir}/Uhlitz_2021-original_sample_meta.csv", index=False)

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="Uhlitz_2021_EMBO_Mol_Med",
    study_doi="10.15252/emmm.202114123",
    study_pmid="34409732",
    tissue_processing_lab="Markus Morkel lab",
    dataset="Uhlitz_2021",
    medical_condition="colorectal cancer",
    sample_tissue="colon",
    # Paper: invasive tumor front
    tumor_source="core",
    histological_type="adenocarcinoma",
    enrichment_cell_types="naive",
    platform="10x 3' v2",
    cellranger_version="3.0.2",
    reference_genome="ensembl.v93",
    matrix_type="raw counts",
    tissue_cell_state="fresh",
    tissue_dissociation="enzymatic",
    hospital_location="Charité University Hospital Berlin",
    country="Germany",
    treatment_status_before_resection="naive",
)

adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]
adata.obs["sample_type"] = adata.obs["tissue"]
adata.obs.loc[adata.obs["sample_type"] == "normal", "tumor_source"] = "normal"
adata.obs["patient_id"] = adata.obs["Patient ID"]

mapping = {
    "": "no_rep",
    "1": "rep1",
    "2": "rep2",
}
adata.obs["replicate"] = adata.obs["replicate"].map(mapping)

mapping = {
    "f": "female",
    "m": "male",
}
adata.obs["sex"] = adata.obs["Sex"].map(mapping)


# define the dictionary mapping
mapping = {
    "C": "cecum",
    "S": "sigmoid colon",
    "T": "transverse colon",
    "A": "ascending colon",
    "R": "rectum",
    "D": "descending colon",
}
adata.obs["anatomic_location"] = adata.obs["Localisation"].map(mapping)

mapping = {
    "cecum": "proximal colon",
    "sigmoid colon": "distal colon",
    "transverse colon": "proximal colon",
    "ascending colon": "proximal colon",
    "rectum": "distal colon",
    "descending colon": "distal colon",
}
adata.obs["anatomic_region"] = adata.obs["anatomic_location"].map(mapping)

adata.obs["Stage"] = adata.obs["Stage"].replace("TisN0", "T1N0")

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

adata.obs["tumor_grade"] = adata.obs["Grade"]

adata.obs["microsatellite_status"] = adata.obs["Microsatellite status"]

mapping = {
    "S": "serrated polyps",
    "I": "inflammation associated",
    "C": "conventional adenoma",
    "nan": "nan",
}
adata.obs["inferred_precursor_lesion"] = adata.obs["Inferred progression "].map(mapping)

adata.obs["sample_matched"] = ah.pp.add_matched_samples_column(adata.obs, "patient_id")

# %%
# Split and explode the "Key Driver Mutations" column to get unique mutations per patient
mutation_list = (
    adata.obs["Key Driver Mutations"]
    .str.split(",")
    .explode()
    .dropna()
    .str.strip()
    .unique()
)

# List comprehension to create new columns for each unique mutation
for mutation in mutation_list:
    adata.obs[f"{mutation}_status_driver_mut"] = adata.obs["Key Driver Mutations"].apply(
        lambda x: "mut" if mutation in str(x) else "wt" if x != "None" else "nan"
    )

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
    "inferred_precursor_lesion",
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
