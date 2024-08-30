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
# # Dataloader: Huang_2024_Nat_Cancer

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

# %% tags=["parameters"]
artifact_dir = nxfvars.get("artifact_dir", "/data/scratch/marteau/downloads/crc-atlas/")
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")
cpus = nxfvars.get("cpus", 8)

# %%
d10_1038_s43018_023_00691_z = f"{databases_path}/d10_1038_s43018_023_00691_z"
gtf = f"{annotation}/ensembl.v84_gene_annotation_table.csv"

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
# Converted csv file to sparse mtx file -> csv takes ages to load
adata = sc.AnnData(
    X=fmm.mmread(
        f"{d10_1038_s43018_023_00691_z}/data/allImmuneCell_pure_counts_matrix.mtx.gz"
    ),
    obs=pd.read_csv(
        f"{d10_1038_s43018_023_00691_z}/data/allImmuneCell_pure_barcodes.tsv.gz",
        delimiter="\t",
        header=None,
        index_col=0,
    ),
    var=pd.read_csv(
        f"{d10_1038_s43018_023_00691_z}/data/allImmuneCell_pure_features.tsv.gz",
        delimiter="\t",
        header=None,
        index_col=0,
    ),
)
adata.X = csr_matrix(adata.X.astype(int))
adata.var_names.name = None
adata.obs_names.name = None

# %%
meta = pd.read_csv(f"{d10_1038_s43018_023_00691_z}/data/OMIX921-99-03.csv").rename(
    columns={"Unnamed: 0": "obs_names"}
)

adata.obs = adata.obs.reset_index(names="obs_names")
adata.obs = pd.merge(
    adata.obs,
    meta,
    how="left",
    on=["obs_names"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %% [markdown]
# ## 1a. Gene annotations

# %%
# load gtf for gene mapping
gtf = pd.read_csv(gtf)
gtf["ensembl"] = gtf["gene_id"].apply(ah.pp.remove_gene_version)
gtf = ah.pp.append_duplicate_suffix(df=gtf, column="gene_name", sep=".")
gene_ids = gtf.set_index("gene_name")["gene_id"].to_dict()

# %%
# replace scores with underscores to match gtf symbols
adata.var_names = adata.var_names.str.replace("--", "__")
adata.var_names = adata.var_names.str.replace("XXyac-YX65C7-A.2", "XXyac-YX65C7_A.2")
adata.var_names = adata.var_names.str.replace("XXyac-YX65C7-A.3", "XXyac-YX65C7_A.3")

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
# Reorder by gtf (i.e. chromosome position)
gene_index = gtf[gtf["ensembl"].isin(adata.var_names)]["ensembl"].values
adata = adata[:, gene_index]

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="Huang_2024_Nat_Cancer",
    study_doi="10.1038/s43018-023-00691-z",
    study_pmid="38200243",
    tissue_processing_lab="Wenfei Jin lab",
    dataset="Huang_2024",
    medical_condition="colorectal cancer",
    enrichment_cell_types="naive",  # but data includes only immune cells ...
    matrix_type="raw counts",
    # Paper methods: 10x Genomics Single Cell 3′ Library Construction kit v.3
    platform="10x 3' v3",
    cellranger_version="3.0.2",
    reference_genome="ensembl.v84",
    sample_tissue="colon",
    hospital_location="Shenzhen People’s Hospital, The First Affiliated Hospital, Southern University of Science and Technology, Shenzhen",
    country="China",
    tissue_cell_state="fresh",
    # Paper methods: None of them received chemotherapy or radiation before tumor resection
    treatment_status_before_resection="naive",
    # Paper methods: three patients with CRC diagnosed with pMMR adenocarcinoma
    histological_type="adenocarcinoma",
    mismatch_repair_deficiency_status="pMMR",
    microsatellite_status="MSS",
    # Paper methods: The three patients signed ... and they were women.
    sex="female",
    # Paper methods: Two samples were classified as stage IIA disease and the other was classified as stage IIB
    tumor_stage_TNM="II",
)

# %%
adata.obs["patient_id"] = adata.obs["Tsample"]
adata.obs["sample_id"] = adata.obs["sample"]
adata.obs["cell_type_study"] = adata.obs["Tcelltype"]

mapping = {
    "CRCT": "tumor",
    "CRCN": "normal",
    # Paper methods: We collected tumor tissue, paracancerous tissue and normal tissue
    "CRCP": "normal", # CRCP: para-cancer -> according to pathologist can be treated as normal!
}
adata.obs["sample_type"] = adata.obs["tissue"].map(mapping)

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
    f"{artifact_dir}/Huang_2024-original_sample_meta.csv",
    index=False,
)

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]]

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["sample_id"]
    + "-"
    + adata.obs_names
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
