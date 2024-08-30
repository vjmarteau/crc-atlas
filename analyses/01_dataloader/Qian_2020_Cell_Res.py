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
# # Dataloader: Qian_2020_Cell_Res

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
cpus = nxfvars.get("cpus", 2)
gtf = f"{annotation}/ensembl.v84_gene_annotation_table.csv"
d10_1038_s41422_020_0355_0 = f"{databases_path}/d10_1038_s41422_020_0355_0"

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
adata = anndata.AnnData(
    X=fmm.mmread(
        f"{d10_1038_s41422_020_0355_0}/lambrechtslab/CRC_counts/matrix.mtx.gz"
    ).T,
    obs=pd.read_csv(
        f"{d10_1038_s41422_020_0355_0}/lambrechtslab/CRC_counts/barcodes.tsv",
        delimiter="\t",
        header=None,
        index_col=0,
    ),
    var=pd.read_csv(
        f"{d10_1038_s41422_020_0355_0}/lambrechtslab/CRC_counts/genes.tsv",
        delimiter="\t",
        header=None,
        usecols=[0],
        index_col=0,
    ),
)
adata.obs_names.name = None
adata.var_names.name = None

adata.X = csr_matrix(adata.X)

# %% [markdown]
# This seems to be equivalent to the sum of all samples in E-MTAB-8107.

# %% [markdown]
# ###  Gene annotations

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

# %% [markdown]
# ### Get available metadata

# %%
Colorectalcancer_metadata = pd.read_csv(
    f"{d10_1038_s41422_020_0355_0}/lambrechtslab/2099-Colorectalcancer_metadata.csv"
)

# Merge cell meta
adata.obs["Cell"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    Colorectalcancer_metadata,
    how="left",
    on=["Cell"],
    validate="m:1",
).set_index("Cell")
adata.obs_names.name = None

adata.obs["Sample"] = adata.obs_names.str.split("_").str[0]

# %%
ENA_metadata = pd.read_csv(
    f"{d10_1038_s41422_020_0355_0}/metadata/E-MTAB-8107_sample_meta.csv"
)
ENA_metadata = ENA_metadata[ENA_metadata["disease"] == "colorectal cancer"]

adata.obs["Assay Name"] = adata.obs["Sample"].str.replace("scr", "")

adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    ENA_metadata,
    how="left",
    on=["Assay Name"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
sample_meta = pd.read_csv(
    f"{d10_1038_s41422_020_0355_0}/metadata/sample_meta.txt", delimiter="\t"
)
patient_meta = pd.read_csv(f"{d10_1038_s41422_020_0355_0}/metadata/patient_meta.csv")

meta = pd.merge(
    sample_meta,
    patient_meta,
    how="left",
    on=["Patient number"],
    validate="m:1",
)

# %%
# Use number of cells per sample to map back study meta
adata.obs["Cells"] = adata.obs["ENA_SAMPLE"].map(
    adata.obs["ENA_SAMPLE"].value_counts().to_dict()
)

adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    meta,
    how="left",
    on=["Cells"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
# Get original sample/patient meta and remove cell level meta
sample_meta = (
    adata.obs[
        adata.obs.columns[adata.obs.nunique() <= adata.obs["Assay Name"].nunique()]
    ]
    .groupby("Assay Name")
    .first()
    .reset_index()
)

# %%
sample_meta.to_csv(f"{artifact_dir}/Qian_2020-original_sample_meta.csv", index=False)

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="Qian_2020_Cell_Res",
    study_doi="10.1038/s41422-020-0355-0",
    study_pmid="32561858",
    tissue_processing_lab="Diether Lambrechts lab",
    dataset="Qian_2020",
    medical_condition="colorectal cancer",
    sample_tissue="colon",
    # Paper: All tumors were treatment-naïve 
    treatment_status_before_resection="naive",
    tissue_cell_state="fresh",
    cellranger_version="2.0.0",
    reference_genome="ensembl.v84",
    # Paper: Following resection, tissues were rapidly digested to a singlecell suspension and unbiasedly subjected to 3′-scRNA-seq
    enrichment_cell_types="naive",
    matrix_type="raw counts",
    hospital_location="University Hospitals Leuven",
    country="Belgium",
)

# %%
adata.obs["ENA_sample_accession"] = adata.obs["ENA_SAMPLE"]
adata.obs["sample_id"] = adata.obs["Assay Name"]
adata.obs["patient_id"] = "P" + adata.obs["PatientNumber"].astype(str)

mapping = {
    "C": "core",
    "B": "border",
    "N": "normal",
}
adata.obs["tumor_source"] = adata.obs["TumorSite"].map(mapping)

mapping = {
    "C": "tumor",
    "B": "tumor",
    "N": "normal",
}
adata.obs["sample_type"] = adata.obs["TumorSite"].map(mapping)

mapping = {
    "Right caecum": "cecum",
    "Left rectosigmoid": "rectosigmoid junction",
    "Left sigmoid": "sigmoid colon",
    "Right ascending": "ascending colon",
}
adata.obs["anatomic_location"] = adata.obs["Location"].map(mapping)

mapping = {
    "cecum": "proximal colon",
    "rectosigmoid junction": "distal colon",
    "sigmoid colon": "distal colon",
    "ascending colon": "proximal colon",
}
adata.obs["anatomic_region"] = adata.obs["anatomic_location"].map(mapping)

# define the dictionary mapping
mapping = {
    "global moderately differentiated adenocarcinoma with mixed glandulair/ mucinous growth pattern/ moderate budding": "G2",
    "moderately differentiated adenocarcinoma NST": "G2",
    "moderately differentiated adenocarcinoma": "G2",
    "moderately differentiated adenocarcinoma NOS": "G2",
}
adata.obs["tumor_grade"] = adata.obs["Pathological subtype"].map(mapping)

# define the dictionary mapping
mapping = {
    "global moderately differentiated adenocarcinoma with mixed glandulair/ mucinous growth pattern/ moderate budding": "mucinous adenocarcinoma",
    "moderately differentiated adenocarcinoma NST": "adenocarcinoma",
    "moderately differentiated adenocarcinoma": "adenocarcinoma",
    "moderately differentiated adenocarcinoma NOS": "adenocarcinoma",
}
adata.obs["histological_type"] = adata.obs["Pathological subtype"].map(mapping)

adata.obs["platform"] = adata.obs["10X version"].map({"3' V2": "10x 3' v2"})


adata.obs["age"] = adata.obs["Age range (years)"].astype(str)

# define the dictionary mapping
mapping = {
    "Female": "female",
    "Male": "male",
}
adata.obs["sex"] = adata.obs["Gender"].map(mapping)

# define the dictionary mapping
mapping = {
    "MSI-high": "MSI-H",
    "MSS": "MSS",
}
adata.obs["microsatellite_status"] = adata.obs["Molecular status"].map(mapping)

adata.obs["cell_type_study"] = adata.obs["CellType"]

# %%
adata.obs["tumor_stage_TNM_T"] = adata.obs["TNM classification"].apply(
    lambda x: re.search(r"(T[0-4][a-c]?)", x).group(1)
    if re.search(r"(T[0-4][a-c]?)", x)
    else None
)
adata.obs["tumor_stage_TNM_N"] = adata.obs["TNM classification"].apply(
    lambda x: re.search(r"(N[0-4][a-c]?)", x).group(1)
    if re.search(r"(N[0-4][a-c]?)", x)
    else None
)
adata.obs["tumor_stage_TNM_M"] = adata.obs["TNM classification"].apply(
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
# check if Stage column and tumor_stage_TNM match -> they do
adata.obs[
    [
        "patient_id",
        "TNM classification",
        "tumor_stage_TNM_T",
        "tumor_stage_TNM_N",
        "tumor_stage_TNM_M",
        "tumor_stage_TNM",
        "Stage",
    ]
].groupby("patient_id").first()

# %%
# Split and explode the "Key Driver Mutations" column to get unique mutations per patient
mutation_list = (
    adata.obs["Mutations"]
    .str.split("/")
    .explode()
    .dropna()
    .str.strip()
    .unique()
)

# List comprehension to create new columns for each unique mutation
for mutation in mutation_list:
    adata.obs[f"{mutation}_status_driver_mut"] = adata.obs["Mutations"].apply(
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
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["ENA_sample_accession"]
    + "-"
    + adata.obs_names.str.split("_").str[1]
)

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
