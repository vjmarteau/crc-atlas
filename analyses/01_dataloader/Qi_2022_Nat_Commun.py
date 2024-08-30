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
# # Dataloader: Qi_2022_Nat_Commun

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
d10_1038_s41467_022_29366_6 = f"{databases_path}/d10_1038_s41467_022_29366_6"
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
# ## 1. Load raw counts

# %%
mat = fmm.mmread(f"{d10_1038_s41467_022_29366_6}/HRA000979/matrix.mtx")
barcodes = pd.read_csv(
    f"{d10_1038_s41467_022_29366_6}/HRA000979/barcodes.tsv",
    delimiter="\t",
    header=None,
    index_col=0,
)
barcodes.index.name = None
features = pd.read_csv(
    f"{d10_1038_s41467_022_29366_6}/HRA000979/features.tsv",
    delimiter="\t",
    header=None,
    index_col=0,
)
features.index.name = None

adata = sc.AnnData(X=mat.T)
adata.X = csr_matrix(adata.X)
adata.obs = barcodes
adata.var = features

# %%
xls_path = f"{d10_1038_s41467_022_29366_6}/HRA000979/Supplementary Data 4/41467_2022_29366_MOESM7_ESM.xlsx"
meta_xls = pd.read_excel(xls_path)

# Merge cell meta
adata.obs["Barcode"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    meta_xls,
    how="left",
    on=["Barcode"],
    validate="m:1",
).set_index("Barcode")
adata.obs_names.name = None

# %%
# import tabula
# pdf_path = f"{d10_1038_s41467_022_29366_6}/HRA000979/41467_2022_29366_MOESM1_ESM.pdf"
# meta = tabula.read_pdf(pdf_path, pages="17")[0]
# meta = meta.drop("Unnamed: 0", axis=1).T
# meta.columns = meta.iloc[0]
# meta.columns.name = None
# meta = meta.tail(-1)
# meta = meta.rename_axis("Patient")
# meta = meta.reset_index()
# meta["Tumour size"] = meta["Tumour size"].apply(
#    lambda x: re.sub(r"(\s+(?!cm))", "x", x)
# )
# meta.to_csv(f"{d10_1038_s41467_022_29366_6}/HRA000979/Table_S1.csv", index=False)

meta = pd.read_csv(f"{d10_1038_s41467_022_29366_6}/metadata/Table_S1.csv")

# %%
adata.obs["PatientID"] = adata.obs["PatientID"].str.upper()

adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    meta,
    how="left",
    left_on=["PatientID"],
    right_on=["Patient"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

del adata.obs["PatientID"]

adata.obs["sample_id"] = adata.obs["Patient"] + "_" + adata.obs["Tissues"]

# %%
xls_path = f"{d10_1038_s41467_022_29366_6}/metadata/41467_2022_29366_MOESM9_ESM.xlsx"
embeddings = pd.read_excel(xls_path)

# adata.obsm["X_umap"] = np.array(embeddings[["UMAP_1", "UMAP_2"]])

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
sample_meta.to_csv(f"{artifact_dir}/Qi_2022-original_sample_meta.csv", index=False)

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

# %%
unmapped_genes = ah.pp.find_unmapped_genes(adata)
len(unmapped_genes)

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="Qi_2022_Nat_Commun",
    study_doi="10.1038/s41467-022-29366-6",
    study_pmid="35365629",
    tissue_processing_lab="Bing Su lab",
    dataset="Qi_2022",
    medical_condition="colorectal cancer",
    tissue_cell_state="fresh",
    enrichment_cell_types="naive",
    # Paper: colonic adenocarcinoma (COAD), n = 2; rectal adenocarcinoma (READ), n = 3)
    histological_type="adenocarcinoma",
    sample_tissue="colon",
    treatment_status_before_resection="naive",
    platform="10x 3' v3",
    reference_genome="ensembl.v93",
    matrix_type="raw counts",
    hospital_location="Ruijin Hospital, Shanghai Jiao Tong University School of Medicine, Shanghai",
    country="China",
)

# %%
adata.obs["patient_id"] = adata.obs["Patient"]

adata.obs["age"] = adata.obs["Age"]

# define the dictionary mapping
mapping = {
    "Female": "female",
    "Male": "male",
}
adata.obs["sex"] = adata.obs["Gender"].map(mapping)

# define the dictionary mapping
mapping = {
    "N": "normal",
    "T": "tumor",
}
adata.obs["sample_type"] = adata.obs["Tissues"].map(mapping)


# define the dictionary mapping
mapping = {
    "Rectum": "distal colon",
    "Colon": "nan",
}
adata.obs["anatomic_region"] = adata.obs["Anatomical region"].map(mapping)

# define the dictionary mapping
mapping = {
    "Rectum": "rectum",
    "Colon": "nan",
}
adata.obs["anatomic_location"] = adata.obs["Anatomical region"].map(mapping)


# define the dictionary mapping
mapping = {
    "Low": "G1",
    "Moderately": "G2",
    "Low or moderately": "G1-G2",
}
adata.obs["tumor_grade"] = adata.obs["Grade (Differntiated)"].map(mapping)

adata.obs["tumor_dimensions"] = [
    item.replace(" ", "") for item in adata.obs["Tumour size"]
]

mapping = {
    "5x4x2cm": "5",
    "4.5x3x1.5cm": "4.5",
    "3x3x1.5cm": "3",
    "5x4x1.5cm": "5",
    "3.5x3.5x1.5cm": "3.5",
}
adata.obs["tumor_size"] = adata.obs["tumor_dimensions"].map(mapping)


adata.obs["tumor_stage_TNM_T"] = "T" + adata.obs["pTNM: T"].astype(str)
adata.obs["tumor_stage_TNM_N"] = "N" + adata.obs["pTNM: N"].astype(str)
adata.obs["tumor_stage_TNM_M"] = "M" + adata.obs["pTNM: M"].astype(str)

adata.obs["tumor_stage_TNM"] = ah.pp.tnm_tumor_stage(adata.obs, "patient_id")


adata.obs["cell_type_coarse_study"] = adata.obs["MainTypes"]
adata.obs["cell_type_study"] = adata.obs["Cell Types"]

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
    + adata.obs_names.str.rsplit("_", n=1).str[1]
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
