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
# # Dataloader: Terekhanova_2023_Nature

# %%
import os

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
cpus = nxfvars.get("cpus", 6)
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")

# %%
gtf = f"{annotation}/gencode.v32_gene_annotation_table.csv"
d10_1038_s41586_023_06682_5 = f"{databases_path}/d10_1038_s41586_023_06682_5"

# %%
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

# %%
# Read the YAML file and load it into a dictionary
with open(reference_meta, "r") as f:
    ref_meta_dict = yaml.load(f, Loader=yaml.Loader)

# %% [markdown]
# ## 1. Load raw adata counts from Synapse and append HTAN metadata

# %% [markdown]
# The paper states "sc/snRNA-seq" multiple times -> HTAN Assay column says "scRNA-seq", so I assume these 3 samples are single cell rnaseq and not single nuclei! TableS1 is ambigoius in this regard, will see downstream if these samples pop out!

# %%
meta = pd.read_csv(
    f"{d10_1038_s41586_023_06682_5}/metadata/HTAN_WUSTL_meta.csv"
).dropna(axis=1, how="all")

# %%
patient_meta = pd.read_csv(
    f"{d10_1038_s41586_023_06682_5}/metadata/TableS1_subset_CRC.csv"
)

# %%
def _load_mtx(file_meta):
    FileName = f"{d10_1038_s41586_023_06682_5}/HTAN_WUSTL/{file_meta['Filename']}"
    adata = anndata.AnnData(
        X=fmm.mmread(f"{FileName}-matrix.mtx.gz").T,
        obs=pd.read_csv(
            f"{FileName}-barcodes.tsv",
            delimiter="\t",
            header=None,
            names=["obs_names"],
            index_col=0,
        ),
        var=pd.read_csv(
            f"{FileName}-features.tsv",
            delimiter="\t",
            header=None,
            names=["ensembl", "symbol"],
            index_col=0,
        ),
    )
    adata.obs = adata.obs.assign(**file_meta)
    adata.obs_names.name = None
    adata.X = csr_matrix(adata.X)
    return adata

# %%
meta_sub = meta.loc[meta["Filename"].isin(["CM556C1-T1", "CM618C1-S1", "CM618C2-T1"])]

# %%
adatas = process_map(_load_mtx, [r for _, r in meta_sub.iterrows()], max_workers=cpus)
adata = anndata.concat(adatas, index_unique="-", join="outer", fill_value=0)

# %% [markdown]
# ### 1. Gene annotations

# %%
# get id mappings from one of the samples
features = pd.read_csv(
    f"{d10_1038_s41586_023_06682_5}/HTAN_WUSTL/CM556C1-T1-features.tsv",
    delimiter="\t",
    header=None,
    names=["ensembl", "symbol"],
    index_col=0,
)

# map back gene symbols to .var
gene_symbols = features["ensembl"].to_dict()

adata.var["ensembl"] = adata.var_names
adata.var["symbol"] = (
    adata.var["ensembl"].map(gene_symbols).fillna(value=adata.var["ensembl"])
)
adata.var_names = adata.var["ensembl"]
adata.var_names.name = None

# %%
unmapped_genes = ah.pp.find_unmapped_genes(adata)
len(unmapped_genes)

# %% [markdown]
# ## 2. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="Terekhanova_2023_Nature",
    study_doi="10.1038/s41586-023-06682-5",
    study_pmid="37914932",
    tissue_processing_lab="Li Ding lab",
    dataset="WUSTL_HTAN",
    medical_condition="colorectal cancer",
    matrix_type="raw counts",
    cellranger_version="6.0.2",
    reference_genome="gencode.v32",
    platform="10x 3' v2",
    tissue_cell_state="frozen",
    # TableS1: "treatment_category"
    treatment_status_before_resection="treated",
    tumor_stage_TNM_M="M1",
    tumor_stage_TNM="IVA",
    enrichment_cell_types="naive",
    hospital_location="Department of Surgery, Washington University in St Louis, St Louis, MO",
    country="US",
)

# %%
adata.obs["synapse_sample_accession"] = adata.obs["Synapse Id"]
adata.obs["sample_id"] = adata.obs["HTAN Biospecimen ID"]
adata.obs["patient_id"] = adata.obs["HTAN Participant ID"]

adata.obs["age"] = adata.obs["Age at Diagnosis (years)"]
adata.obs["sex"] = adata.obs["Gender"]
adata.obs["ethnicity"] = adata.obs["Race"]

# %%
mapping = {
    "Metastatic": "metastasis",
    "Primary": "tumor",
}
adata.obs["sample_type"] = adata.obs["Tumor Tissue Type"].map(mapping)

mapping = {
    "Metastatic": "liver",
    "Primary": "colon",
}
adata.obs["sample_tissue"] = adata.obs["Tumor Tissue Type"].map(mapping)

mapping = {
    "8140": "adenocarcinoma",
}
adata.obs["histological_type"] = (
    adata.obs["Histologic Morphology Code"].astype(str).map(mapping)
)

# From Table S1: patient_meta -> died during surgery
mapping = {
    "HTA12_105": "CAPOX + bevacizumab x 12 cycles",
    "HTA12_106": "FOLFOX + bevacizumab",
}
adata.obs["treatment_drug"] = (
    adata.obs["patient_id"].astype(str).map(mapping)
)

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["sample_id"]
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
# Save unfiltered raw counts
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")

# %% [markdown]
# ## 3. Load additional samples (seurat converted)
These are already filtered

# %%
def _load_mtx(file_meta):
    FilePath = f"{d10_1038_s41586_023_06682_5}/HTAN_WUSTL/seurat/mat/{file_meta['Filename']}"
    adata = sc.AnnData(
        X=fmm.mmread(
            f"{FilePath}_matrix.mtx.gz"
        ),
        obs=pd.read_csv(
            f"{FilePath}_barcodes.tsv",
            delimiter="\t",
            header=None,
            index_col=0,
        ),
        var=pd.read_csv(
            f"{FilePath}_features.tsv",
            delimiter="\t",
            header=None,
            index_col=0,
        ),
    )
    adata.X = csr_matrix(adata.X) # convert coo_matrix back to csr_matrix
    adata.var_names.name = None
    adata.obs_names.name = None

    cell_meta = pd.read_csv(
            f"{FilePath}_cell_meta.tsv",
            delimiter=" ",
        )
    
    adata.obs = adata.obs.reset_index(names="obs_names")
    adata.obs = pd.merge(
        adata.obs,
        cell_meta,
        how="left",
        on=["obs_names"],
        validate="m:1",
    ).set_index("obs_names")
    
    adata.obs = adata.obs.assign(**file_meta)
    adata.obs_names.name = None
    return adata

# %%
meta_sub = meta.loc[~meta["Filename"].isin(["CM556C1-T1", "CM618C1-S1", "CM618C2-T1"])]

# %%
adatas = process_map(_load_mtx, [r for _, r in meta_sub.iterrows()], max_workers=cpus)
adata = anndata.concat(adatas, index_unique="-", join="outer", fill_value=0)

# %%
adata.var["symbol"] = adata.var_names
features = ah.pp.append_duplicate_suffix(df=features, column="ensembl", sep=".")
adata.var["ensembl"] = (
    adata.var["symbol"].map(features.reset_index().set_index("ensembl")["index"].to_dict())
).fillna(value=adata.var["symbol"])
adata.var_names = adata.var["ensembl"]
adata.var_names.name = None

# %%
unmapped_genes = ah.pp.find_unmapped_genes(adata)
len(unmapped_genes)

# %%
adata.obs = adata.obs.assign(
    study_id="Terekhanova_2023_Nature",
    study_doi="10.1038/s41586-023-06682-5",
    study_pmid="37914932",
    tissue_processing_lab="Li Ding lab",
    dataset="Terekhanova_2023_HTAN",
    medical_condition="colorectal cancer",
    matrix_type="raw counts",
    cellranger_version="6.0.2",
    reference_genome="gencode.v32",
    platform="10x 3' v2",
    tissue_cell_state="frozen",
    enrichment_cell_types="naive",
    hospital_location="Department of Surgery, Washington University in St Louis, St Louis, MO",
    country="US",
)

# %%
adata.obs["synapse_sample_accession"] = adata.obs["Synapse Id"]
adata.obs["sample_id"] = adata.obs["HTAN Biospecimen ID"]
adata.obs["patient_id"] = adata.obs["HTAN Participant ID"]

adata.obs["age"] = adata.obs["Age at Diagnosis (years)"]
adata.obs["sex"] = adata.obs["Gender"]
adata.obs["ethnicity"] = adata.obs["Race"]
adata.obs["cell_type_study"] = adata.obs["cell_type"]
adata.obs["original_obs_names"] = adata.obs["Original_barcode"]

# %%
mapping = {
    "Metastatic": "metastasis",
    "Primary": "tumor",
    "Normal distant": "normal",
}
adata.obs["sample_type"] = adata.obs["Tumor Tissue Type"].map(mapping)

mapping = {
    "Metastatic": "liver",
    "Primary": "colon",
    "Normal distant": "colon",
}
adata.obs["sample_tissue"] = adata.obs["Tumor Tissue Type"].map(mapping)

mapping = {
    "8140": "adenocarcinoma",
    "not applicable": "nan",
}
adata.obs["histological_type"] = (
    adata.obs["Histologic Morphology Code"].astype(str).map(mapping)
)

# %%
mapping = {
    "Ascending colon": "ascending colon",
    "Sigmoid colon": "sigmoid colon",
    "Transverse colon": "transverse colon",
}
adata.obs["anatomic_location"] = (
    adata.obs["Site of Resection or Biopsy"].astype(str).map(mapping)
)

mapping = {
    "Ascending colon": "proximal colon",
    "Sigmoid colon": "distal colon",
    "Transverse colon": "proximal colon",
}
adata.obs["anatomic_region"] = (
    adata.obs["Site of Resection or Biopsy"].astype(str).map(mapping)
)

# %%
# From Table S1: patient_meta 
mapping = {
    "CM1563C1-S1Y1": "IVA", # this is paired primary/mestastasis samples
    "CM1563C1-T1Y1": "IVA",
    "CM268C1-S1": "IVA", # same here
    "CM268C1-T1": "IVA",
    "CM663C1-T1Y1": "IVA", # metastasis sample
    "SP369H1-Mc1": "nan",
}
adata.obs["tumor_stage_TNM"] = (
    adata.obs["Filename"].astype(str).map(mapping)
)
mapping = {
    "CM1563C1-S1Y1": "M1", # this is paired primary/mestastasis samples
    "CM1563C1-T1Y1": "M1",
    "CM268C1-S1": "M1", # same here
    "CM268C1-T1": "M1",
    "CM663C1-T1Y1": "M1", # metastasis sample
    "SP369H1-Mc1": "nan",
}
adata.obs["tumor_stage_TNM"] = (
    adata.obs["Filename"].astype(str).map(mapping)
)

mapping = {
    "CM1563C1-S1Y1": "treated",
    "CM1563C1-T1Y1": "treated",
    "CM268C1-S1": "naive", # From Table S1: patient_meta -> died during surgery
    "CM268C1-T1": "naive",
    "CM663C1-T1Y1": "treated",
    "SP369H1-Mc1": "nan",
}
adata.obs["treatment_status_before_resection"] = (
    adata.obs["Filename"].astype(str).map(mapping)
)

mapping = {
    "CM1563C1-S1Y1": "FOLFOX + bevacizumab x 6 cycles",
    "CM1563C1-T1Y1": "FOLFOX + bevacizumab x 6 cycles",
    "CM268C1-S1": "naive",
    "CM268C1-T1": "naive",
    "CM663C1-T1Y1": "FOLFOX + bevacizumab x 6 cycles",
    "SP369H1-Mc1": "nan",
}
adata.obs["treatment_drug"] = (
    adata.obs["Filename"].astype(str).map(mapping)
)

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["sample_id"]
    + "-"
    + adata.obs["Original_barcode"].str.split("-").str[0]
)

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
adata.obs["sample_id"].value_counts()

# %%
# Save unfiltered raw counts
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
