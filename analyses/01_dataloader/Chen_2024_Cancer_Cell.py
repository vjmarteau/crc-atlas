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
# # Dataloader: Chen_2024_Cancer_Cell

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
d10_1016_j_ccell_2024_06_009 = f"{databases_path}/d10_1016_j_ccell_2024_06_009"
gtf = f"{annotation}/ensembl.v98_gene_annotation_table.csv"

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
    X=fmm.mmread(f"{d10_1016_j_ccell_2024_06_009}/GSE236581/GSE236581_counts.mtx.gz").T,
    obs=pd.read_csv(
        f"{d10_1016_j_ccell_2024_06_009}/GSE236581/GSE236581_barcodes.tsv.gz",
        delimiter="\t",
        header=None,
        index_col=0,
    ),
    var=pd.read_csv(
        f"{d10_1016_j_ccell_2024_06_009}/GSE236581/GSE236581_features.tsv.gz",
        delimiter="\t",
        header=None,
        index_col=0,
    ),
)
adata.X = csr_matrix(adata.X.astype(int))
adata.var_names.name = None
adata.obs_names.name = None

# %%
cell_meta = pd.read_csv(f"{d10_1016_j_ccell_2024_06_009}/GSE236581/GSE236581_CRC-ICB_metadata.txt.gz", sep=' ').reset_index(names="obs_names")

adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    cell_meta,
    how="left",
    on=["obs_names"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
sample_meta = pd.read_csv(f"{d10_1016_j_ccell_2024_06_009}/metadata/TableS1_sample_metadata.csv")
sample_meta["Sample ID"] = sample_meta["Sample ID"].str.replace("P", "CRC")

adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    sample_meta,
    how="left",
    left_on=["Ident"],
    right_on=["Sample ID"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
patient_meta = pd.read_csv(f"{d10_1016_j_ccell_2024_06_009}/metadata/TableS1_patient_metadata.csv")

adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    patient_meta,
    how="left",
    left_on=["Patient"],
    right_on=["Patient ID"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
geofetch = pd.read_csv(
    f"{d10_1016_j_ccell_2024_06_009}/GSE236581/GSE236581_PEP/GSE236581_PEP_raw.csv"
).dropna(axis=1, how="all")

geofetch["Ident"] = geofetch["sample_title"].str.replace("P", "CRC")

# %%
adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    geofetch,
    how="left",
    on=["Ident"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %% [markdown] tags=[]
# ## 2. Gene annotations

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
adata.var = adata.var[["symbol", "ensembl"]]
adata.var_names.name = None

# %%
unmapped_genes = ah.pp.find_unmapped_genes(adata)
len(unmapped_genes)

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="Chen_2024_Cancer_Cell",
    study_doi="10.1016/j.ccell.2024.06.009",
    study_pmid="38981439",
    tissue_processing_lab="Zemin Zhang lab",
    dataset="Chen_2024",
    medical_condition="colorectal cancer",
    # Paper methods: histopathological diagnosis of locally advanced or metastatic colon, rectal, or duodenum adenocarcinoma;
    histological_type="adenocarcinoma",
    matrix_type="raw counts",
    reference_genome="ensembl.v98",
    # Paper methods (Key resources table): ChromiumTM Next GEM Single Cell 5′ Library and Gel Bead Kit v1.1
    platform="10x 5' v1",
    cellranger_version="6.1.2",
    hospital_location="Beijing Cancer Hospital",
    country="China",
    tissue_cell_state="fresh",
    # Paper methods: FACS 7AAD− CD235a− cells (live cell enrichment; erythrocyte depletion) -> only blood samples?
    enrichment_cell_types="naive",
    NCBI_BioProject_accession="PRJNA991601",
)

# %%
adata.obs["sample_id"] = adata.obs["Ident"].str.replace("-", "_")
adata.obs["patient_id"] = adata.obs["Patient"]

adata.obs["original_obs_names"] = adata.obs_names
adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]

adata.obs["cell_type_coarse_study"] = adata.obs["MajorCellType"]
adata.obs["cell_type_study"] = adata.obs["SubCellType"]

adata.obs["mismatch_repair_deficiency_status"] = adata.obs["dMMR/pMMR"]
adata.obs["microsatellite_status"] = adata.obs["MSI/MSS"]

adata.obs["tumor_mutational_burden"] = adata.obs["TMB (Muts/Mb)"]

# %%
mapping = {
    'Tumor': "tumor",
    'Adjacent normal tissue': "normal",
    'Peripheral blood': "blood",
    'TN': "lymph node", # tumor node = lymph node?
    'LN': "lymph node",
}
adata.obs["sample_type"] = adata.obs["sample_source_name_ch1"].map(mapping)

# %%
adata.obs["age"] = adata.obs["Age"].fillna(np.nan).astype("Int64")
# define the dictionary mapping
mapping = {
    "Female": "female",
    "Male": "male",
}
adata.obs["sex"] = adata.obs["Gender"].map(mapping)

# %%
mapping = {
    "Yes": "mut",
    "No": "wt",
}
adata.obs["POLE_status_driver_mut"] = adata.obs["POLE Mutation"].map(mapping)

# %%
mapping = {
     'Descending colon': "descending colon",
     'Ascending colon': "ascending colon",
     'Low rectum': "rectum",
     'Duodenum': "duodenum",
     'Sigmoid': "sigmoid colon",
     'Sigmoid colon': "sigmoid colon",
     'Transverse colon': "transverse colon",
}
adata.obs["anatomic_location"] = adata.obs["Tumor Location"].map(mapping)

mapping = {
     'descending colon': "distal colon",
     'ascending colon': "proximal colon",
     'rectum': "distal colon",
     'duodenum': "proximal small intestine",
     'sigmoid colon': "distal colon",
     'transverse colon': "proximal colon",
}
adata.obs["anatomic_region"] = adata.obs["anatomic_location"].map(mapping)

# %%
mapping = {
    'normal': "colon",
    'tumor': "colon",
    'blood': "blood",
    'lymph node': "lymph node",
}
adata.obs["sample_tissue"] = adata.obs["sample_type"].map(mapping)

adata.obs.loc[
    (adata.obs["anatomic_region"] == "proximal small intestine"),
    "sample_tissue"
] = "small intestine"

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

# %%
# Replace None values in tumor_stage_TNM_N/M column with 'Nx'/'Mx'
adata.obs["tumor_stage_TNM_N"] = (
    adata.obs["tumor_stage_TNM_N"].astype(str).replace("None", "Nx")
)
adata.obs["tumor_stage_TNM_M"] = (
    adata.obs["tumor_stage_TNM_M"].astype(str).replace("None", "Mx")
)

adata.obs["tumor_stage_TNM"] = ah.pp.tnm_tumor_stage(adata.obs, "patient_id")

adata.obs["tumor_stage_TNM"] = adata.obs["tumor_stage_TNM"].replace("nan", np.nan).fillna(adata.obs["Tumor stage"])

# %%
mapping = {
    'Pre': "naive",
    'On': "treated",
    'Post': "treated",
}
adata.obs["treatment_status_before_resection"] = adata.obs["Treatment Stage"].map(mapping)

mapping = {
    'CR': "CR: complete response",
    'SD': "SD: stable disease",
    'PR': "PR: partial response",
}
adata.obs["RECIST"] = adata.obs["Response"].map(mapping)

mapping = {
    'CR': "responder",
    'SD': "non-responder",
    'PR': "responder",
}
adata.obs["treatment_response"] = adata.obs["Response"].map(mapping)

adata.obs["treatment_drug"] = adata.obs["Treatment Regimen"]

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
# Add back potentially interesting metadata that does not fit the reference meta
ref_meta_cols += [
    "POLE_status_driver_mut",
    "Sampling Stage",
    "Treatment Stage",
    "Tumor Regression Ratio",
    "TRG status",
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
    + adata.obs_names.str.rsplit("_", n=1).str[1]
)

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
adata

# %%
# Set display options to show all rows and columns
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# %%
adata.obs["sample_id"].value_counts()

# %%
pd.reset_option("display.max_rows")
pd.reset_option("display.max_columns")

# %%
# Save adata
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
