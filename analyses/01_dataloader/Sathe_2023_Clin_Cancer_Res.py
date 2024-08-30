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
# # Dataloader: Sathe_2023_Clin_Cancer_Res

# %%
import os
import re
from pathlib import Path

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
d10_1158_1078_0432_CCR_22_2041 = f"{databases_path}/d10_1158_1078_0432_CCR_22_2041"
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
# ## 1. Load raw adata counts

# %%
def _load_mtx(sample):
    mat = fmm.mmread(
        f"{d10_1158_1078_0432_CCR_22_2041}/mCRC_scRNA_raw/{sample}/matrix.mtx.gz"
    )
    features = pd.read_csv(
        f"{d10_1158_1078_0432_CCR_22_2041}/mCRC_scRNA_raw/{sample}/features.tsv.gz",
        delimiter="\t",
        header=None,
    ).rename(columns={0: "gene_id", 1: "symbol"})
    barcodes = pd.read_csv(
        f"{d10_1158_1078_0432_CCR_22_2041}/mCRC_scRNA_raw/{sample}/barcodes.tsv.gz",
        delimiter="\t",
        header=None,
    ).rename(columns={0: "barcodes"})

    adata = sc.AnnData(X=mat.T)
    adata.X = csr_matrix(adata.X)

    adata.var = features
    adata.var_names = adata.var["symbol"]
    adata.var_names.name = None
    adata.var_names_make_unique()

    adata.obs = pd.DataFrame(index=barcodes["barcodes"])
    adata.obs_names.name = None
    adata.obs["sample_id"] = sample

    return adata


# %%
samples = [
    "5784_PBMC",
    "5915_mCRC",
    "5915_normal_liver",
    "6198_mCRC",
    "6198_normal_liver",
    "6335_PBMC",
    "6335_mCRC",
    "6335_normal_liver",
    "6593_mCRC",
    "6648_mCRC",
    "6648_normal_liver",
    "8640_mCRC",
]

# %%
adatas = process_map(_load_mtx, samples, max_workers=cpus)
adata = anndata.concat(adatas, index_unique="-", join="outer", fill_value=0)
adata.obs_names = adata.obs_names.str.rsplit("-", n=1).str[0]
adata.obs["patient_id"] = "P" + adata.obs["sample_id"].str.split("_").str[0]

# %% [markdown]
# ### 1a. Gene annotations

# %%
# load gtf for gene mapping
gtf = pd.read_csv(gtf)
gtf = ah.pp.append_duplicate_suffix(df=gtf, column="gene_name", sep="-")
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
# ### 1b. Compile sample/patient metadata

# %% [markdown]
# All tumors had adenocarcinoma histopathology, The only exception was P6198’s tumor, which had mixed neuroendocrine adenocarcinoma (MANEC) histology. All tumors were MSS

# %%
mCRC_cell_labels = pd.read_csv(f"{d10_1158_1078_0432_CCR_22_2041}/mCRC_cell_labels.csv")
patient_meta = pd.read_csv(f"{d10_1158_1078_0432_CCR_22_2041}/Table1.csv")
patient_meta["Patient ID"] = "P" + patient_meta["Patient ID"].astype(str)

# %%
# Merge mCRC_cell_labels
mapping = {
    "mCRC": "tumor",
    "normal": "normal",
    "PBMC": "PBMC",
}
adata.obs["orig.ident"] = adata.obs["patient_id"] + "_" + adata.obs["sample_id"].str.split("_").str[1].map(mapping)
adata.obs["cell_barcode"] = adata.obs["orig.ident"] + "_" + adata.obs_names

adata.obs = adata.obs.reset_index(names="obs_names")
adata.obs = pd.merge(
    adata.obs,
    mCRC_cell_labels,
    how="left",
    on=["cell_barcode"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

adata.obs = adata.obs.rename(columns={'orig.ident_x': 'orig.ident'}).drop('orig.ident_y', axis=1)

# %%
# Merge patient meta
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

# %%
# From mCRC_scRNAseq_README.txt
mapping = {
    "mCRC": "metastatic colorectal cancer",
    "normal": "paired normal liver",
    "PBMC": "peripheral blood mononuclear cells",
}
adata.obs["condition"] = adata.obs["sample_id"].str.split("_").str[1].map(mapping)

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
sample_meta.to_csv(f"{artifact_dir}/Sathe_2023-original_sample_meta.csv", index=False)

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
mapping = {
    "mCRC": "metastasis",
    "normal": "normal",
    "PBMC": "blood",
}
adata.obs["sample_type"] = adata.obs["sample_id"].str.split("_").str[1].map(mapping)

mapping = {
    "mCRC": "liver",
    "normal": "liver",
    "PBMC": "blood",
}
adata.obs["sample_tissue"] = adata.obs["sample_id"].str.split("_").str[1].map(mapping)

# Paper Methods: All tumors had adenocarcinoma histopathology, The only exception was P6198’s tumor, which had mixed neuroendocrine adenocarcinoma (MANEC) histology. All tumors were MSS
mapping = {
    "P5784": "adenocarcinoma",
    "P5915": "adenocarcinoma",
    "P6198": "neuroendocrine carcinoma",
    "P6335": "adenocarcinoma",
    "P6593": "adenocarcinoma",
    "P6648": "adenocarcinoma",
    "P8640": "adenocarcinoma",
}
adata.obs["histological_type"] = adata.obs["patient_id"].map(mapping)

mapping = {
    "Sigmoid colon": "sigmoid colon",
    "Rectosigmoid colon": "rectosigmoid junction",
    "Transverse colon": "transverse colon",
    "Descending colon": "descending colon",
    "Rectum": "rectum",
}
adata.obs["anatomic_location"] = adata.obs["Primary tumor site"].map(mapping)


mapping = {
    "sigmoid colon": "distal colon",
    "rectosigmoid junction": "distal colon",
    "transverse colon": "proximal colon",
    "descending colon": "distal colon",
    "rectum": "distal colon",
}
adata.obs["anatomic_region"] = adata.obs["anatomic_location"].map(mapping)

adata.obs["microsatellite_status"] = adata.obs["Microsatellite status"]

mapping = {
    "Positive": "pos",
    "Negative": "neg",
    "Equivocal": "nan",
}
adata.obs["HER2_status_driver_mut"] = adata.obs["HER2 expression"].map(mapping)

# Paper: Table S2 - Sequencing metrics for scRNAseq
mapping = {
    "5784_PBMC": "fresh",
    "5915_mCRC": "fresh",
    "5915_normal_liver": "fresh",
    "6198_mCRC": "frozen",
    "6198_normal_liver": "frozen",
    "6335_PBMC": "fresh",
    "6335_mCRC": "fresh",
    "6335_normal_liver": "fresh",
    "6593_mCRC": "frozen",
    "6648_mCRC": "fresh",
    "6648_normal_liver": "fresh",
    "8640_mCRC": "frozen",
}
adata.obs["tissue_cell_state"] = adata.obs["sample_id"].map(mapping)

adata.obs["cell_type_study"] = adata.obs["final_celltype"]

# %%
adata.obs = adata.obs.assign(
    study_id="Sathe_2023_Clin_Cancer_Res",
    study_doi="10.1158/1078-0432.CCR-22-2041",
    study_pmid="36239989",
    tissue_processing_lab="Hanlee P Ji lab",
    dataset="Sathe_2023",
    medical_condition="colorectal cancer",
    tumor_stage_TNM_M="M1",
    tumor_stage_TNM="IV",
    # Paper: For the single-cell studies, the tumors were analyzed directly without flow sorting
    enrichment_cell_types="naive",
    platform="10x 3' v2",
    cellranger_version="3.1.0",
    reference_genome="ensembl.v93",
    matrix_type="raw counts",
    hospital_location="Department of Surgery, Stanford University, Stanford, California.",
    country="US",
)

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["orig.ident"]
    + "-"
    + adata.obs_names
)
adata.obs_names_make_unique()

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)
    
ref_meta_cols += ["HER2_status_driver_mut"]

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
adata

# %%
adata.obs["sample_id"].value_counts() # ??? raw counts all 737280 ?!

# %%
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
