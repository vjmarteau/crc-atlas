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
# # Dataloader: Liu_2022_Cancer_Cell

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
d10_1016_j_ccell_2022_02_013 = f"{databases_path}/d10_1016_j_ccell_2022_02_013"
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
def load_counts(file):

    adata = sc.AnnData(
        X=fmm.mmread(
            f"{d10_1016_j_ccell_2022_02_013}/GSE164522/GSE164522_CRLM_{file}_expression.mat.gz"
        ),
        obs=pd.read_csv(
            f"{d10_1016_j_ccell_2022_02_013}/GSE164522/GSE164522_CRLM_{file}_expression_barcodes.csv.gz",
            index_col=0,
        ),
        var=pd.read_csv(
            f"{d10_1016_j_ccell_2022_02_013}/GSE164522/GSE164522_CRLM_{file}_expression_features.csv.gz",
            index_col=0,
        ),
    )
    adata.X = csr_matrix(adata.X) # convert coo_matrix back to csr_matrix
    adata.var_names.name = None
    adata.obs_names.name = None
    adata.obs["type"] = file
    return adata

# %%
adatas = process_map(
    load_counts,
    ["LN", "MN", "MT", "PBMC", "PN", "PT"],
    max_workers=cpus,
)
adata = anndata.concat(adatas, join="outer")
adata.obs_names = adata.obs_names.str.replace(".", "-")

# %%
meta = pd.read_csv(f"{d10_1016_j_ccell_2022_02_013}/GSE164522/GSE164522_CRLM_metadata.csv.gz").rename({"Unnamed: 0": "obs_names"}, axis=1)

# %%
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
#ah.pp.undo_log_norm(adata)

# %%
#assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"

# %%
patient_meta = pd.read_csv(f"{d10_1016_j_ccell_2022_02_013}/metadata/TableS2.csv")
adata.obs["patient_id"] = adata.obs["patient"].str.replace("patient", "CRLM")

# %%
adata.obs = adata.obs.reset_index(names="obs_names")
adata.obs = pd.merge(
    adata.obs,
    patient_meta,
    how="left",
    left_on=["patient_id"],
    right_on=["Patient ID"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

geofetch = pd.read_csv(
    f"{d10_1016_j_ccell_2022_02_013}/GSE164522/GSE164522_PEP/GSE164522_PEP_raw.csv"
).dropna(axis=1, how="all")

adata.obs = adata.obs.reset_index(names="obs_names")
adata.obs = pd.merge(
    adata.obs,
    geofetch,
    how="left",
    left_on=["patient_id"],
    right_on=["sample_title"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
adata.obs["sample_id"] = adata.obs["patient_id"] + "_" + adata.obs["type"]

# %% [markdown]
# ## 2. Gene annotations

# %%
# load gtf for gene mapping
gtf = pd.read_csv(gtf)
gtf = ah.pp.append_duplicate_suffix(df=gtf, column="gene_name", sep="-")
gene_ids = gtf.set_index("gene_name")["gene_id"].to_dict()

# %%
# Some gene symbols have scores instead of underscores ?!
adata.var_names =adata.var_names.str.replace("--", "__")
adata.var_names =adata.var_names.str.replace("XXyac-YX65C7-A.2", "XXyac-YX65C7_A.2")

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
adata.obs = adata.obs.assign(
    study_id="Liu_2022_Cancer_Cell",
    study_doi="10.1016/j.ccell.2022.02.013",
    study_pmid="35303421",
    tissue_processing_lab="Zemin Zhang lab",
    dataset="Liu_2022_CD45Pos",
    medical_condition="colorectal cancer",
    # Paper methods: Fresh tumor and adjacent normal tissue samples (at least 2 cm from matched tumor tissues) from both primary tumor and liver metastasis were surgically resected from the above-described patients
    tissue_cell_state="fresh",
    # Paper: we generated scRNA-seq data for CD45+ cells
    enrichment_cell_types="CD45+",
    
    # Paper: Here, we performed scRNA-seq analyses on 201 clinical samples of 51 treatment-naive patients with CRC liver metastasis (CRLM) and primary HCC/CRC
    # Paper methods: None of them were treated with chemotherapy or radiation prior to tumor resection
    treatment_status_before_resection="naive",
    cellranger_version="3.0.0",
    reference_genome="ensembl.v84",
    matrix_type="raw counts",
    hospital_location="Beijing Shijitan Hospital | Peking University Cancer Hospital & Institute",
    country="China",
    NCBI_BioProject_accession="PRJNA691050",
)

# %%
adata

# %%
adata.obs["sample_id"].value_counts()

# %%
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
