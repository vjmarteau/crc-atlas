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
# # Dataloader: Qin_2023_Cell_Rep_Med

# %%
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

# %%
artifact_dir = nxfvars.get("artifact_dir", "/data/scratch/marteau/downloads/crc-atlas/")
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../data/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")
d10_1016_j_xcrm_2023_101231 = f"{databases_path}/d10_1016_j_xcrm_2023_101231"
gtf = f"{annotation}/ensembl.v98_gene_annotation_table.csv"

# %%
# Read the YAML file and load it into a dictionary
with open(reference_meta, "r") as f:
    ref_meta_dict = yaml.load(f, Loader=yaml.Loader)


# %% [markdown]
# ## 1. Load raw counts

# %%
adata = sc.AnnData(
    X=fmm.mmread(
        f"{d10_1016_j_xcrm_2023_101231}/CSE0000154/MM_coordinate_matrix.mtx"
    ).T,
    obs=pd.read_csv(
        f"{d10_1016_j_xcrm_2023_101231}/CSE0000154/MM_coordinate_matrix_barcodes.csv",
        header=None,
        index_col=0,
    ),
    var=pd.read_csv(
        f"{d10_1016_j_xcrm_2023_101231}/CSE0000154/MM_coordinate_matrix_genes.csv",
        header=None,
        index_col=0,
    ),
)
adata.X = csr_matrix(adata.X)  # convert coo_matrix back to csr_matrix
adata.var_names.name = None
adata.obs_names.name = None

# %%
adata.obs["patient_id"] = (
    adata.obs_names.str.split("-", n=2).str[0]
    + adata.obs_names.str.split("-", n=2).str[1]
)
# Not sure about the rest of the obs_names string, but this gives the number of samples + sample types indicated in the paper
adata.obs["sample_id"] = (
    adata.obs_names.str.split("-", n=2).str[0]
    + adata.obs_names.str.split("-", n=2).str[1]
    + "_"
    + adata.obs_names.str.split("-", n=3).str[2]
)

adata.obs["type"] = adata.obs["sample_id"].str.split("_").str[1]

# %%
patient_meta = pd.read_csv(f"{d10_1016_j_xcrm_2023_101231}/metadata/TableS1.csv")

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

# %% [markdown]
# ## 2. Gene annotations

# %%
gtf = pd.read_csv(gtf)
gene_ids = gtf.set_index("gene_name")["gene_id"].to_dict()

# %%
adata.var["symbol"] = adata.var_names
adata.var["symbol"] = adata.var["symbol"].replace(
    {
        "Metazoa-SRP": "Metazoa_SRP",
        "Y-RNA": "Y_RNA",
        "5-8S-rRNA": "5_8S_rRNA",
        "5S-rRNA": "5S_rRNA",
    }
)

# %%
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
gene_index = gtf[gtf["gene_id"].isin(adata.var["ensembl"])]["gene_id"].apply(
    ah.pp.remove_gene_version
)
adata = adata[:, gene_index]

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="Qin_2023_Cell_Rep_Med",
    study_doi="10.1016/j.xcrm.2023.101231",
    study_pmid="37852187",
    tissue_processing_lab="Hongcheng Lin lab",
    dataset="Qin_2023",
    medical_condition="colorectal cancer",
    matrix_type="raw counts",
    reference_genome="ensembl.v98",
    platform="DNBelab C4",
    hospital_location="Sixth Affiliated Hospital of Sun Yat-sen University, Guangzhou",
    country="China",
    # Paper methods: Fresh tumor and adjacent normal tissue samples
    sample_tissue="colon",
    tissue_cell_state="fresh",
    enrichment_cell_types="naive",
)

# %%
# define the dictionary mapping
mapping = {
    "PRET": "tumor",
    "POSTT": "tumor",
    "POSTN": "normal",
}
adata.obs["sample_type"] = adata.obs["type"].map(mapping)

adata.obs["age"] = adata.obs["Age"].fillna(np.nan).astype("Int64")

# define the dictionary mapping
mapping = {
    "F": "female",
    "M": "male",
}
adata.obs["sex"] = adata.obs["Sex"].map(mapping)

# %%
# define the dictionary mapping
mapping = {
    "Rectal adenocarcinoma": "distal colon",
}
adata.obs["anatomic_region"] = adata.obs["Diagnosis"].map(mapping)

# define the dictionary mapping
mapping = {
    "Rectal adenocarcinoma": "rectum",
}
adata.obs["anatomic_location"] = adata.obs["Diagnosis"].map(mapping)

# define the dictionary mapping
mapping = {
    "Rectal adenocarcinoma": "adenocarcinoma",
}
adata.obs["histological_type"] = adata.obs["Diagnosis"].map(mapping)

# define the dictionary mapping
mapping = {
    "MSS/ pMMR": "MSS",
}
adata.obs["microsatellite_status"] = adata.obs["Molecular subtype"].map(mapping)

# define the dictionary mapping
mapping = {
    "MSS/ pMMR": "pMMR",
}
adata.obs["mismatch_repair_deficiency_status"] = adata.obs["Molecular subtype"].map(mapping)

adata.obs["KRAS_status_driver_mut"] = adata.obs["KRAS"].replace({"Mutant": "mut", "Wildtype": "wt"})
adata.obs["NRAS_status_driver_mut"] = adata.obs["NRAS"].replace({"Mutant": "mut", "Wildtype": "wt"})
adata.obs["BRAF_status_driver_mut"] = adata.obs["BRAF"].replace({"Mutant": "mut", "Wildtype": "wt"})
adata.obs["PI3KCA_status_driver_mut"] = adata.obs["PI3KCA"].replace({"Mutant": "mut", "Wildtype": "wt"})

# define the dictionary mapping
mapping = {
    "PRET": "naive",
    "POSTT": "treated",
    "POSTN": "treated",
}
adata.obs["treatment_status_before_resection"] = adata.obs["type"].map(mapping)

adata.obs["treatment_drug"] = adata.obs["Cycles of NAC"].astype(str) + "x " + adata.obs["NAC"].astype(str)

# define the dictionary mapping
mapping = {
    "Moderately differentiated": "G2",
    "Moderately-poorly differentiated": "G2-G3",
    "nan": "nan",
}
adata.obs["tumor_grade"] = adata.obs["Grade after NAC and surgery"].astype(str).map(mapping)

# define the dictionary mapping
mapping = {
    "PR": "PR: partial response",
    "NR": "NR",
    "CR": "CR: complete response",
    "nan": "nan",
}
adata.obs["RECIST"] = adata.obs["Tumor response group"].map(mapping)

# define the dictionary mapping
mapping = {
    "PR": "responder",
    "NR": "non-responder",
    "CR": "responder",
    "nan": "nan",
}
adata.obs["treatment_response"] = adata.obs["Tumor response group"].map(mapping)

# Not sure about stuff like T2-3a -> now it picks the lower stage (i.e. T2 in this example)
adata.obs["tumor_stage_TNM_T"] = adata.obs["Clinical stage (cTNM) before NAC and surgery"].apply(
    lambda x: re.search(r"(T[0-4][a-c]?)", x).group(1)
    if re.search(r"(T[0-4][a-c]?)", x)
    else None
)
# define the dictionary mapping
mapping = {
    "T3a": "T3",
    "T3c": "T3",
    "T3b": "T3",
}
adata.obs["tumor_stage_TNM_T"] = adata.obs["tumor_stage_TNM_T"].replace(mapping)

adata.obs["tumor_stage_TNM_N"] = adata.obs["Clinical stage (cTNM) before NAC and surgery"].apply(
    lambda x: re.search(r"(N[0-4][a-c]?)", x).group(1)
    if re.search(r"(N[0-4][a-c]?)", x)
    else None
)
adata.obs["tumor_stage_TNM_M"] = adata.obs["Clinical stage (cTNM) before NAC and surgery"].apply(
    lambda x: re.search(r"((M[0-1]|Mx)[a-c]?)", x).group(1)
    if re.search(r"((M[0-1]|Mx)[a-c]?)", x)
    else None
)
adata.obs["tumor_stage_TNM_M"] = adata.obs["tumor_stage_TNM_M"].fillna("Mx")

adata.obs["tumor_stage_TNM"] = ah.pp.tnm_tumor_stage(adata.obs, "patient_id")

# %%
adata.obs["original_obs_names"] = adata.obs_names
adata.obs_names = adata.obs["dataset"] + "-" + adata.obs_names.str.replace("-", "_")

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

ref_meta_cols += ["Tumor regression grade", "Surgical procedure", "Clinical stage (cTNM) before NAC and surgery", "Pathologic stage(ypTN M) after NAC and surgery"]

# Add driver_mutation columns
ref_meta_cols += [col for col in adata.obs.columns if "_status_driver_mut" in col]

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
adata.obs["patient_id"].value_counts()

# %%
adata.obs["sample_id"].value_counts()

# %%
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
