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
# # Dataloader: MUI_Innsbruck_AbSeq

# %% tags=["parameters"]
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

# %%
artifact_dir = nxfvars.get("artifact_dir", "/data/scratch/marteau/downloads/crc-atlas/")
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")
cpus = nxfvars.get("cpus", 4)

# %%
data_path = nxfvars.get("data_path", "../../data/own_datasets/")
bd_output = f"{data_path}/healthy-blood-AbSeq/10_rhapsody_pipeline_v2/Fixed/"

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
meta = pd.DataFrame({'multiplex_batch': ["CML-P1", "CML-P2", "CML-P3", "CML-P4"]})


# %%
def load_adata_from_mtx(sample_meta):
    # Load data into memory
    matrix = fmm.mmread(
        f"{bd_output}/{sample_meta['multiplex_batch']}/{sample_meta['multiplex_batch']}_RSEC_MolsPerCell_Unfiltered_MEX/matrix.mtx.gz"
    )

    barcodes = pd.read_csv(
        f"{bd_output}/{sample_meta['multiplex_batch']}/{sample_meta['multiplex_batch']}_RSEC_MolsPerCell_Unfiltered_MEX/barcodes.tsv.gz",
        delimiter="\t",
        header=None,
        names=["Cell_Index"],
    )
    barcodes["obs_names"] = "Cell_Index_" + barcodes["Cell_Index"].astype(str)
    barcodes.set_index("obs_names", inplace=True)
    barcodes.index.name = None

    features = pd.read_csv(
        f"{bd_output}/{sample_meta['multiplex_batch']}/{sample_meta['multiplex_batch']}_RSEC_MolsPerCell_Unfiltered_MEX/features.tsv.gz",
        delimiter="\t",
        header=None,
        names=["var_names", "symbol", "type"],
    )

    # Assemble adata
    adata = sc.AnnData(X=matrix.T)
    adata.X = csr_matrix(adata.X)
    adata.var = features
    adata.obs = barcodes
    adata.obs = adata.obs.assign(**sample_meta)
    adata.var_names = adata.var["var_names"]
    adata.var_names.name = None

    adata_tmp = sc.read_h5ad(
        f"{bd_output}/{sample_meta['multiplex_batch']}/{sample_meta['multiplex_batch']}.h5ad"
    )
    adata_tmp.obs_names = "Cell_Index_" + adata_tmp.obs_names

    for col in ["Cell_Type_Experimental", "Sample_Tag", "Sample_Name"]:
        sample_tag_dict = adata_tmp.obs[col].to_dict()
        adata.obs[col] = adata.obs_names.map(sample_tag_dict)

    return adata


# %%
adatas = process_map(
    load_adata_from_mtx,
    [r for _, r in meta.iterrows()],
    max_workers=cpus,
)

# %%
adata = anndata.concat(adatas, index_unique="-", join="outer", fill_value=0)

# %% [markdown]
# ## 2. Gene annotations

# %%
gtf = f"{annotation}/gencode.v44_gene_annotation_table.csv"
gtf = pd.read_csv(gtf)
gtf["ensembl"] = gtf["Geneid"].apply(ah.pp.remove_gene_version)

# %%
# map back gene symbols to .var
gene_symbols = gtf.set_index("GeneSymbol")["ensembl"].to_dict()

# %%
adata.var["symbol"] = adata.var_names
adata.var["ensembl"] = (
    adata.var["symbol"].map(gene_symbols).fillna(value=adata.var["symbol"])
)

# %%
adata.var_names = adata.var["ensembl"]
adata.var_names.name = None

# %%
unmapped_genes = ah.pp.find_unmapped_genes(adata)
len(unmapped_genes)

# %%
# AbSeq
unmapped_genes

# %%
# Reorder by gtf (i.e. chromosome position)
gene_index = gtf[gtf["ensembl"].isin(adata.var_names)]["ensembl"].values
missing_genes = adata[:, ~adata.var_names.isin(gtf["ensembl"])].var_names.values
gene_index = np.append(gene_index, missing_genes)
adata = adata[:, gene_index]

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="MUI_Innsbruck_AbSeq",
    tissue_processing_lab="Andreas Pircher lab",
    dataset="MUI_Innsbruck_AbSeq",
    medical_condition="healthy",
    hospital_location="Landeskrankenhaus Innsbruck - tirol kliniken",
    country="Austria",
    enrichment_cell_types="naive",
    tissue_cell_state="fresh",
    platform="BD Rhapsody",
    reference_genome="gencode.v44",
    matrix_type="raw counts",
    sample_type="blood",
    sample_tissue="blood",
)

# %%
adata.obs["patient_id"] = adata.obs["multiplex_batch"].str.split("-").str[1]
adata.obs["sample_id"] = adata.obs["patient_id"] + "_" + adata.obs["Sample_Name"]
adata.obs["multiplex_batch"] = adata.obs["patient_id"]

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
ref_meta_cols += [
    "multiplex_batch",
    "Cell_Type_Experimental",
    "Sample_Tag",
    "Sample_Name",
]

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
adata.obs_names = (
    adata.obs["dataset"].astype(str)
    + "-"
    + adata.obs["multiplex_batch"].astype(str)
    + "-"
    + adata.obs_names.str.split("-").str[0].str.split("Cell_Index_").str[1]
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
adata.obs["patient_id"].unique()

# %%
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
