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
# # Dataloader: Li_2017_Nat_Genet

# %%
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
from sklearn.utils import sparsefuncs
from threadpoolctl import threadpool_limits

# %% tags=["parameters"]
artifact_dir = nxfvars.get("artifact_dir", "/data/scratch/marteau/downloads/crc-atlas/")
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")
cpus = nxfvars.get("cpus", 2)
d10_1038_ng_3818 = f"{databases_path}/d10_1038_ng_3818"
gtf = f"{annotation}/gencode.v19_gene_annotation_table.csv"

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
def load_adata_from_csv(counts):
    mat = pd.read_csv(f"{d10_1038_ng_3818}/GSE81861/{counts}.csv.gz")
    mat = mat.T
    mat.columns = mat.iloc[0]
    mat = mat[1:]
    mat = mat.rename_axis(None, axis=1)

    mat = mat.astype(int)
    adata = sc.AnnData(mat)
    adata.X = csr_matrix(adata.X)

    adata.obs["file"] = counts
    adata.obs["Title"] = adata.obs_names.str.split("__").str[0]
    adata.obs["cell_type"] = adata.obs_names.str.split("__").str[1]
    adata.obs["color"] = adata.obs_names.str.split("__").str[2]
    return adata

# %%
count_names = [
    "GSE81861_CRC_NM_all_cells_COUNT",
#    "GSE81861_CRC_NM_epithelial_cells_COUNT",
    "GSE81861_CRC_tumor_all_cells_COUNT",
#    "GSE81861_CRC_tumor_epithelial_cells_COUNT",
]
adatas = []
for counts in count_names:
    adata = load_adata_from_csv(counts)
    adatas.append(adata)

adata = anndata.concat(
    adatas,
    index_unique="-",
    join="outer",
    fill_value=0,
)
adata.obs_names = adata.obs_names.str.split("-").str[0]

# %% [markdown]
# ## Get available metadata

# %%
patient_meta = pd.read_csv(f"{d10_1038_ng_3818}/metadata/TableS2.csv")
ID_match = pd.read_csv(
    f"{d10_1038_ng_3818}/GSE81861/GSE81861_GEO_EGA_ID_match.csv.gz", index_col=False,
).drop(columns=['Unnamed: 0'])
geofetch = pd.read_csv(
    f"{d10_1038_ng_3818}/GSE81861/GSE81861_PEP/GSE81861_PEP_raw.csv"
).dropna(axis=1, how="all")

sample_meta = pd.merge(
    ID_match,
    geofetch,
    how="left",
    left_on=["Sample"],
    right_on=["sample_geo_accession"],
    validate="m:1",
)

meta = pd.merge(
    sample_meta,
    patient_meta,
    how="left",
    left_on=["patient_id"],
    right_on=["ID"],
    validate="m:1",
)

# Remove cell line metadata -> 590 cells after qc (375 tumor + 215 matched normal)
meta = meta[meta["sample_type"] == "Primary CRC patient tissue sample"].copy()

# %%
# Merge accession numbers
adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    meta,
    how="left",
    on=["Title"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# -> For some reason raw data has 641 cells, I guess the difference did not pass qc. Unfortunately no metadata available for these cells
# Please note that [1] only the QC-passed samples are included in the records
adata = adata[adata.obs["Title"].isin(meta["Title"])].copy()

# %%
# Get original sample/patient meta and remove cell level meta
sample_meta = (
    adata.obs[
        adata.obs.columns[adata.obs.nunique() <= adata.obs["Title"].nunique()]
    ]
    .groupby("Title")
    .first()
    .reset_index()
)

# %%
sample_meta.to_csv(
    f"{artifact_dir}/Li_2017-original_sample_meta.csv",
    index=False,
)

# %% [markdown]
# ## 2. Gene annotations

# %%
adata.var["Geneid"] = adata.var_names.str.rsplit("_", n=1).str[1]
adata.var["ensembl"] = adata.var["Geneid"].apply(ah.pp.remove_gene_version)
adata.var["study_var_names"] = adata.var_names
adata.var_names = adata.var["ensembl"]
adata.var_names.name = None

# %%
gtf = pd.read_csv(gtf)
gtf["ensembl"] = gtf["Geneid"].apply(ah.pp.remove_gene_version)

gene_symbols = gtf.set_index("Geneid")["GeneSymbol"].to_dict()
adata.var["GeneSymbol"] = (
    adata.var["Geneid"].map(gene_symbols).fillna(value=adata.var["Geneid"])
)

# %%
unmapped_genes = ah.pp.find_unmapped_genes(adata)
len(unmapped_genes)

# %%
# Reorder by gtf (i.e. chromosome position)
gene_index = gtf[gtf["ensembl"].isin(adata.var_names)]["ensembl"].values
missing_genes = adata[:, ~adata.var_names.isin(gtf["ensembl"])].var_names.values
gene_index = np.append(gene_index, missing_genes)
adata = adata[:, gene_index]

# %%
# SMARTer (C1) is a full‐length protocol -> take into account gene length
# For some reason some of the provided start and stop positions are the same between genes??!
# Will use mapped gene length from gencode gtf later on!
#adata.var['start'] = adata.var['study_var_names'].str.extract(r'(\d+)-(\d+)')[0].astype(int)
#adata.var['end'] = adata.var['study_var_names'].str.extract(r'(\d+)-(\d+)')[1].astype(int)
#adata.var['length'] = adata.var['end'] - adata.var['start'] + 1
gene_symbols = gtf.set_index("Geneid")["Length"].to_dict()
adata.var["Length"] = (
    adata.var["Geneid"].map(gene_symbols)
)
adata.var = adata.var[["Geneid", "GeneSymbol", "Length", "ensembl", "study_var_names"]]

# %%
#length = np.ravel(adata.var["length"])
#sparsefuncs.inplace_row_scale(df.X.T.astype(float), 1 / length)
#df.X.data = np.rint(df.X.data)

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="Li_2017_Nat_Genet",
    study_doi="10.1038/ng.3818",
    study_pmid="28319088",
    tissue_processing_lab="Shyam Prabhakar lab",
    dataset="Li_2017",
    medical_condition="colorectal cancer",
    NCBI_BioProject_accession="PRJNA323703",
    matrix_type="raw counts",
    reference_genome="gencode.v19",
    platform="SMARTer (C1)",
    hospital_location="National Cancer Center of Singapore, Singapore General Hospital",
    country="Singapore",
    # Paper methods: Fresh resected tumor and normal mucosa samples were processed
    tissue_cell_state="fresh",
    enrichment_cell_types="naive",
    sample_tissue="colon",
)

# %%
adata.obs["sample_id"] = adata.obs["Title"]
adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]
adata.obs["cell_type_study"] = adata.obs["cell_type"] # -> epithelial files have fine annotation for epi cell types
adata.obs["sex"] = adata.obs["Gender"]

# define the dictionary mapping
mapping = {
    "Caecum": "cecum",
    "Sigmoid Colon": "sigmoid colon",
    "Rectosigmoid": "rectosigmoid junction",
    "Lower Rectum": "rectum",
    "Mid-sigmoid": "sigmoid colon",
    "Rectum": "rectum",
    "Ascending Colon": "ascending colon",
    "Upper Rectum": "rectum",
}
adata.obs["anatomic_location"] = adata.obs["Site of Tumor"].map(mapping)

# define the dictionary mapping
mapping = {
    "cecum": "proximal colon",
    "sigmoid colon": "distal colon",
    "rectosigmoid junction": "distal colon",
    "rectum": "distal colon",
    "ascending colon": "proximal colon",
}
adata.obs["anatomic_region"] = adata.obs["anatomic_location"].map(mapping)

# define the dictionary mapping
mapping = {
    "GSE81861_CRC_NM_all_cells_COUNT": "normal", # NM: normal matched
    "GSE81861_CRC_tumor_all_cells_COUNT": "tumor",
}

adata.obs["sample_type"] = adata.obs["file"].map(mapping)

adata.obs["sample_id"] = adata.obs["patient_id"] + "_" + adata.obs["sample_type"]

adata.obs["tumor_grade"] = "G" + adata.obs["Grading"].astype(str)

# define the dictionary mapping
mapping = {
    np.nan: "nan",
    "uncomplicated DD proximal transverse colon": "nan",
    "Colitis at Sigmoid": "nan",
    "DD ascending colon- hepatic flexure": "nan",
    "Post neoadjuvant chemoradiotherapy": "neoadjuvant chemoradiotherapy",
    "uncomplicated colonic DD": "nan",
}

adata.obs["treatment_drug"] = adata.obs["Bowel pathology"].map(mapping)

mapping = {
    "nan": "naive",
    "neoadjuvant chemoradiotherapy": "treated",
}
adata.obs["treatment_status_before_resection"] = adata.obs["treatment_drug"].map(mapping)

# %%
adata.obs["tumor_stage_TNM_T"] = adata.obs["TNM"].apply(
    lambda x: re.search(r"(T[0-4][a-c]?)", x).group(1)
    if re.search(r"(T[0-4][a-c]?)", x)
    else None
)
adata.obs["tumor_stage_TNM_N"] = adata.obs["TNM"].apply(
    lambda x: re.search(r"(N[0-4][a-c]?)", x).group(1)
    if re.search(r"(N[0-4][a-c]?)", x)
    else None
)
adata.obs["tumor_stage_TNM_M"] = adata.obs["TNM"].apply(
    lambda x: re.search(r"((M[0-1]|Mx)[a-c]?)", x).group(1)
    if re.search(r"((M[0-1]|Mx)[a-c]?)", x)
    else None
)
adata.obs["tumor_stage_TNM"] = ah.pp.tnm_tumor_stage(adata.obs, "patient_id")

# slight differences in Stage AJCC??! -> similar enough
adata.obs[["patient_id", "TNM", "Stage AJCC", "tumor_stage_TNM", "tumor_stage_TNM_T", "tumor_stage_TNM_N", "tumor_stage_TNM_M"]].groupby("patient_id").first()

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["GEO_sample_accession"]
    + "-"
    + adata.obs["sample_id"]
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
adata.obs["patient_id"].value_counts()

# %%
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
