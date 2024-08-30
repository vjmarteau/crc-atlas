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
# # Dataloader: Conde_2022_Science

# %%
import itertools
import os

import anndata
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
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")

# %%
cpus = nxfvars.get("cpus", 8)
d10_1126_science_abl5197 = f"{databases_path}/d10_1126_science_abl5197"
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
# Get available metadata
ENA_meta = pd.read_csv(
    f"{d10_1126_science_abl5197}/metadata/E-MTAB-11536-sample_meta.csv"
)
ENA_meta = ENA_meta[ENA_meta["read_index"] == "read1"].reset_index(drop=True)

fetchngs = pd.read_csv(
    f"{d10_1126_science_abl5197}/fetchngs/PRJEB51634/samplesheet/samplesheet.csv"
).dropna(axis=1, how="all")

fetchngs["Assay Name"] = fetchngs["run_alias"].str.split(":").str[1]

# %%
meta = pd.merge(
    fetchngs,
    ENA_meta,
    how="left",
    on=["Assay Name"],
    validate="m:1",
)

meta = meta.loc[meta["library construction.1"] == "10x5’ v1"].reset_index(drop=True)
meta["file_name"] = meta["Assay Name"].str.split("_S1_L001").str[0]

# %%
patient_meta = pd.read_csv(f"{d10_1126_science_abl5197}/metadata/TableS1.csv")

meta = pd.merge(
    meta,
    patient_meta,
    how="left",
    left_on=["individual.1"],
    right_on=["Donor ID"],
    validate="m:1",
)

# Handle performance warning: DataFrame is highly fragmented
meta = meta.copy()

# %%
meta.to_csv(f"{artifact_dir}/Conde_2022-original_sample_meta.csv", index=False)


# %%
def load_counts(sample_meta, databases_path, mat_type="filtered"):

    file_path = f"{databases_path}/10_nfcore_scrnaseq/cellranger/count/{sample_meta['file_name']}/outs/{mat_type}_feature_bc_matrix.h5"
    adata = sc.read_10x_h5(file_path)
    adata.obs = adata.obs.assign(**sample_meta)
    adata.var_names_make_unique()
    adata.X = csr_matrix(adata.X.astype(int))

    adata.obs["sample_id"] = adata.obs["file_name"].str.split("GEX_").str[1]
    adata.obs["original_obs_names"] = adata.obs_names
    adata.obs_names = (
        adata.obs["sample_id"] + "_" + adata.obs_names.str.replace("-1", "")
    )
    assert adata.obs_names.is_unique
    return adata


# %%
adatas = process_map(
    load_counts,
    [r for _, r in meta.iterrows()],
    itertools.repeat(d10_1126_science_abl5197),
    itertools.repeat("raw"),
    max_workers=cpus,
)
adata = anndata.concat(adatas, join="outer")

# %% [markdown]
# ### Gene annotations

# %%
# Append gtf gene info
gtf = f"{annotation}/gencode.v44_gene_annotation_table.csv"
gtf = pd.read_csv(gtf)

adata.var.reset_index(names="symbol", inplace=True)
gtf["symbol"] = adata.var["symbol"]
gtf["ensembl"] = gtf["Geneid"].apply(ah.pp.remove_gene_version)
gtf.set_index("ensembl", inplace=True)
adata.var = gtf.copy()
adata.var_names.name = None

# %%
# Sum up duplicate gene ids
duplicated_ids = adata.var_names[adata.var_names.duplicated()].unique()
adata = ah.pp.aggregate_duplicate_gene_ids(adata, duplicated_ids)

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
# Get cell type annotation from cellxgene
cellxgene = sc.read_h5ad(
    f"{d10_1126_science_abl5197}/metadata/CountAdded_PIP_global_object_for_cellxgene.h5ad"
)
cellxgene.obs["sample_id"] = cellxgene.obs_names.str.rsplit("_", n=1).str[0]
cellxgene = cellxgene[
    cellxgene.obs["sample_id"].isin(adata.obs["sample_id"].unique().tolist())
].copy()

cell_types = cellxgene.obs["Manually_curated_celltype"].to_dict()
adata.obs["cell_type_study"] = adata.obs_names.map(cell_types)

# %%
adata.obs = adata.obs.assign(
    study_id="Conde_2022_Science",
    study_doi="10.1126/science.abl5197",
    study_pmid="35549406",
    tissue_processing_lab="Sarah A. Teichmann lab",
    dataset="Conde_2022",
    medical_condition="healthy",
    sample_type="normal",
    tumor_source="normal",
    matrix_type="raw counts",
    reference_genome="gencode.v44",
    cellranger_version="7.1.0",
    treatment_status_before_resection="naive",
    tissue_cell_state="fresh",
    # Methods section: Tissue processing for donors A29, A31, A35, A36, A37, A52 (Cambridge University)
    enrichment_cell_types="ficoll mononuclear cells",
    NCBI_BioProject_accession="PRJEB51634",
    hospital_location="Cambridge Biorepository of Translational Medicine",
    country="United Kingdom",
    platform="10x 5' v1",
)

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["run_accession"]
    + "-"
    + adata.obs_names.str.rsplit("_", n=1).str[1]
)

# %%
adata.obs["patient_id"] = adata.obs["individual"]
adata.obs["ENA_sample_accession"] = adata.obs["ENA_SAMPLE"]
adata.obs["SRA_sample_accession"] = adata.obs["secondary_sample_accession"]

# define the dictionary mapping
mapping = {
    "F": "female",
    "M": "male",
}
adata.obs["sex"] = adata.obs["Sex"].map(mapping)
adata.obs["age"] = adata.obs["Age"]

mapping = {
    "caecum": "cecum",
    "mesenteric lymph node": "mesenteric lymph nodes",
    "ileum": "ileum",
    "transverse colon": "transverse colon",
    "sigmoid colon": "sigmoid colon",
}
adata.obs["anatomic_location"] = adata.obs["organism part"].map(mapping)

mapping = {
    "cecum": "proximal colon",
    "mesenteric lymph nodes": "mesenteric lymph nodes",
    "ileum": "middle small intestine",
    "transverse colon": "proximal colon",
    "sigmoid colon": "distal colon",
}
adata.obs["anatomic_region"] = adata.obs["anatomic_location"].map(mapping)

mapping = {
    "caecum": "colon",
    "mesenteric lymph node": "lymph node",
    "ileum": "small intestine",
    "transverse colon": "colon",
    "sigmoid colon": "colon",
}
adata.obs["sample_tissue"] = adata.obs["organism part"].map(mapping)

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
