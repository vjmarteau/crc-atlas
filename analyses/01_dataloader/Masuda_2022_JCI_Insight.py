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
# # Dataloader: Masuda_2022_JCI_Insight

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
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")
cpus = nxfvars.get("cpus", 6)
d10_1172_jci_insight_154646 = f"{databases_path}/d10_1172_jci_insight_154646"

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
ENA_metadata = pd.read_csv(
    f"{d10_1172_jci_insight_154646}/metadata/ArrayExpress_sample_data_table_E-MTAB-9455.csv"
)
ENA_metadata = ENA_metadata[
    ENA_metadata["single cell library construction"] == "10x 5' v2 sequencing"
]
ENA_metadata = (
    ENA_metadata.groupby("Source Name")
    .first()
    .reset_index()
    .drop(columns=["ENA", "FASTQ"])
)

# %%
samplesheet = (
    pd.read_csv(f"{d10_1172_jci_insight_154646}/samplesheet.csv")
    .drop(columns=["fastq_1", "fastq_2"])
    .rename(columns={"source_name": "Assay Name"})
)

# %%
meta = pd.merge(
    samplesheet,
    ENA_metadata,
    how="left",
    on=["Assay Name"],
    validate="m:1",
).rename(columns={"sample": "sample_id"})

# %%
fetchngs = pd.read_csv(
    f"{d10_1172_jci_insight_154646}/fetchngs/PRJEB40726/samplesheet/samplesheet.csv"
).dropna(axis=1, how="all").drop_duplicates(subset=["secondary_sample_accession"])

# %%
meta = pd.merge(
    meta,
    fetchngs,
    how="left",
    left_on=["ENA_SAMPLE"],
    right_on=["secondary_sample_accession"],
    validate="m:1",
)
# Handle performance warning: DataFrame is highly fragmented
meta = meta.copy()

# %%
meta.to_csv(
    f"{artifact_dir}/Masuda_2022-original_sample_meta.csv",
    index=False,
)


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
    [r for _, r in meta.iterrows()],
    itertools.repeat(d10_1172_jci_insight_154646),
    itertools.repeat("raw"),
    max_workers=cpus,
)

adata = anndata.concat(adatas, join="outer")

# %% [markdown]
# ## 2. Gene annotations

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
# ## 3. Compile original meta from study supplements
# Need to be carefull -> Besides having been frozen, a fraction of the cells was stimulated in culture before sequencing. Idieally, would like to remove those!

# %%
adata.obs = adata.obs.assign(
    study_id="Masuda_2022_JCI_Insight",
    study_doi="10.1172/jci.insight.154646",
    study_pmid="35192548",
    tissue_processing_lab="Arnold Han lab",
    dataset="Masuda_2022_CD3Pos",
    medical_condition="colorectal cancer",
    histological_type="adenocarcinoma",
    sample_tissue="colon",
    # Paper: Here, we sorted and analyzed 37,931 viable TCRαβ+ CD3+
    enrichment_cell_types="CD3+",
    tissue_cell_state="frozen",
    matrix_type="raw counts",
    cellranger_version="7.1.0",
    reference_genome="gencode.v44",
    treatment_status_before_resection="naive",
    hospital_location="Columbia University Irving Medical Center, New York",
    country="US",
    NCBI_BioProject_accession="PRJEB40726",
)

adata.obs["ENA_sample_accession"] = adata.obs["sample_id"]
adata.obs["SRA_sample_accession"] = adata.obs["secondary_sample_accession"]

mapping = {
    "10x 5' v2 sequencing": "10x 5' v2",
}

adata.obs["platform"] = adata.obs["single cell library construction"].map(mapping)

# %%
# Supplemental tables are so bad -> Impossible to get the patient to source name mapping except for "MaycDNA", and a few others by deduction

mapping = {
    "ERR4667456": "nan",
    "ERR4667457": "nan",
    "ERR4667464": "nan",
    "ERR4667468": "nan",
    "ERR4667469": "nan",
    "ERR4667470": "nan",
    "ERR4667480": "TC-040",
}
adata.obs["patient_id"] = adata.obs["sample_id"].map(mapping)

mapping = {
    "ERR4667456": "nan",
    "ERR4667457": "nan",
    "ERR4667464": "nan",
    "ERR4667468": "nan",
    "ERR4667469": "nan",
    "ERR4667470": "nan",
    "ERR4667480": "IIA",
}
adata.obs["tumor_stage_TNM"] = adata.obs["sample_id"].map(mapping)

mapping = {
    "ERR4667456": "nan",
    "ERR4667457": "nan",
    "ERR4667464": "male",
    "ERR4667468": "nan",
    "ERR4667469": "nan",
    "ERR4667470": "nan",
    "ERR4667480": "male",
}
adata.obs["sex"] = adata.obs["sample_id"].map(mapping)

mapping = {
    "ERR4667456": np.nan,
    "ERR4667457": np.nan,
    "ERR4667464": np.nan,
    "ERR4667468": np.nan,
    "ERR4667469": np.nan,
    "ERR4667470": np.nan,
    "ERR4667480": 66,
}
adata.obs["age"] = adata.obs["sample_id"].map(mapping)
adata.obs["age"] = adata.obs["age"].fillna(np.nan).astype("Int64")

# %%
adata.obs_names = adata.obs["dataset"] + "-" + adata.obs_names

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]]

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
adata

# %%
# Save unfiltered raw counts
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
