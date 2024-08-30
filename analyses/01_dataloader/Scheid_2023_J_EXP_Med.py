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
# # Dataloader: Scheid_2023_J_EXP_Med

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
cpus = nxfvars.get("cpus", 2)

# %%
d10_1084_jem_20220538 = f"{databases_path}/d10_1084_jem_20220538"
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
# ## 1. Load raw counts

# %%
adata = sc.AnnData(
    X=fmm.mmread(f"{d10_1084_jem_20220538}/SCP1690/rawcounts_expression_data.mtx").T,
    obs=pd.read_csv(
        f"{d10_1084_jem_20220538}/SCP1690/barcodes.tsv",
        delimiter="\t",
        header=None,
        index_col=0,
    ),
    var=pd.read_csv(
        f"{d10_1084_jem_20220538}/SCP1690/features.tsv",
        delimiter="\t",
        header=None,
        names=["ensembl", "symbol", "type"],
        index_col=0,
    ),
)
adata.X = csr_matrix(adata.X)
adata.var_names.name = None
adata.obs_names.name = None

# %%
cell_meta = pd.read_csv(
    f"{d10_1084_jem_20220538}/SCP1690/MetaData.txt", delimiter="\t"
).rename({"NAME": "barcodes"}, axis=1)

adata.obs["barcodes"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    cell_meta,
    how="left",
    on=["barcodes"],
    validate="m:1",
).set_index("barcodes")
adata.obs_names.name = None

# One cell baarcode is missing
adata = adata[~adata.obs_names.isin(["AAACCTGCAAGTAATG_H111"])].copy()

# %%
patient_meta = pd.read_csv(f"{d10_1084_jem_20220538}/metadata/TableS1.csv")
patient_meta["SUBJECT ID"] = patient_meta["SUBJECT ID"].str.replace("HC", "Healthy")

adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    patient_meta,
    how="left",
    left_on=["donor_id"],
    right_on=["SUBJECT ID"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
gtf = pd.read_csv(gtf)
gtf["ensembl"] = gtf["gene_id"].apply(ah.pp.remove_gene_version)

# %%
adata.var["ensembl"] = adata.var_names

for column in [col for col in gtf.columns if col != "ensembl"]:
    adata.var[column] = adata.var["ensembl"].map(gtf.set_index("ensembl")[column])

# Reorder by gtf (i.e. chromosome position)
gene_index = gtf[gtf["ensembl"].isin(adata.var_names)]["ensembl"].values
adata = adata[:, gene_index]

# %%
adata.obs = adata.obs.assign(
    study_id="Scheid_2023_J_EXP_Med",
    study_doi="10.1084/jem.20220538",
    study_pmid="36752797",
    tissue_processing_lab="Aviv Regev lab",
    dataset="Scheid_2023_CD27PosCD38Pos",
    medical_condition="healthy",
    sample_tissue="colon",
    # Paper methods: HCs were overall healthy individuals without any history of IBD, other autoimmune diseases, infectious colitis, or colon cancer.
    # sample_type="normal",
    treatment_status_before_resection="naive",
    matrix_type="raw counts",
    cellranger_version="3.0.2",
    reference_genome="ensembl.v84",  # probably -> hard to tell with the amount of genes ...
    hospital_location="Massachusetts General Hospital",
    country="US",
    # Paper methods: Biopsy bites or mucosal resection samples were immediately processed
    tissue_cell_state="fresh",
    # Paper methods: sorted using a Sony MA900 cell sorter by gating on live cells in the forward scatter and side scatter and on CD38-FITC and CD27-PE double-positive cells
    enrichment_cell_types="CD27+CD38+",
)

# %%
adata.obs["patient_id"] = adata.obs["donor_id"].str.replace("Healthy", "HC")
adata.obs["sample_id"] = adata.obs["biosample_id"]
adata.obs["platform"] = adata.obs[
    "library_preparation_protocol__ontology_label"
].str.replace("10X 5' v1 sequencing", "10x 5' v1")
adata.obs["cell_type_coarse_study"] = adata.obs["cell_type__ontology_label"]
adata.obs["original_obs_names"] = adata.obs_names

# %%
adata.obs["age"] = adata.obs["AGE"].fillna(np.nan).astype("Int64")

# define the dictionary mapping
mapping = {
    "F": "female",
    "M": "male",
}
adata.obs["sex"] = adata.obs["SEX"].map(mapping)

# define the dictionary mapping
mapping = {
    "Healthy": "normal",
    "UC_NonINF": "UC_NonINF",
    "UC_LessINF": "UC_LessINF",
    "UC_INF": "UC_INF",
}
adata.obs["sample_type"] = adata.obs["sample_type"].map(mapping)

# define the dictionary mapping
mapping = {
    "Right": "nan",
    "Left": "nan",
    "Rectum": "rectum",
    "Sigmoid": "sigmoid colon",
    "Transverse": "transverse colon",
    "Cecum": "cecum",
}
adata.obs["anatomic_location"] = adata.obs["gut_region"].map(mapping)

# define the dictionary mapping
mapping = {
    "Right": "proximal colon",
    "Left": "distal colon",
    "Rectum": "distal colon",
    "Sigmoid": "distal colon",
    "Transverse": "proximal colon",
    "Cecum": "proximal colon",
}
adata.obs["anatomic_region"] = adata.obs["gut_region"].map(mapping)

# %%
adata.obs.loc[adata.obs["sample_id"].str.contains("Prism"), "sample_id"] = (
    adata[adata.obs["sample_id"].str.contains("Prism")]
    .obs["original_obs_names"]
    .str.split("_", n=1)
    .str[1]
)
adata.obs["sample_id"] = adata.obs["sample_id"].str.replace("_", "")

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
# Add potentially interesting columns back
ref_meta_cols += [
    "CLINICAL STATUS",
    "DIAGNOSIS",
    "YEAR OF DIAGNOSIS",
    "EIM",
    "H/O INFECTIOUS COLITIS",
    "HISTOLOGY INFLAMED",
    "HISTOLOGY NON INFLAMED",
    "SAMPLE SOURCE",
    "IBD THERAPY",
]

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["sample_id"]
    + "-"
    + adata.obs_names.str.split("_", n=1).str[0]
)

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
adata

# %%
adata[adata.obs["sample_type"] == "normal"].obs["patient_id"].value_counts()

# %%
adata[adata.obs["sample_type"] == "normal"].obs["sample_id"].value_counts()

# %%
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
