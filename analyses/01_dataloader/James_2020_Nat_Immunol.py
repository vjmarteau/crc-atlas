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
# # Dataloader: James_2020_Nat_Immunol

# %%
import itertools
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
d10_1038_s41590_020_0602_z = f"{databases_path}/d10_1038_s41590_020_0602_z"

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
E_MTAB_8007_sample_meta = pd.read_csv(
    f"{d10_1038_s41590_020_0602_z}/metadata/E-MTAB-8007_sample_meta.csv"
).drop(["Scan Name"], axis=1)

E_MTAB_8474_sample_meta = pd.read_csv(
    f"{d10_1038_s41590_020_0602_z}/metadata/E-MTAB-8474_sample_meta.csv"
).drop(["Scan Name"], axis=1)

ENA_meta = pd.concat([E_MTAB_8007_sample_meta, E_MTAB_8474_sample_meta], axis=0)

# %%
patient_meta = pd.read_csv(
    f"{d10_1038_s41590_020_0602_z}/metadata/TableS1_patient_meta.csv"
)

meta = (
    pd.merge(
        ENA_meta,
        patient_meta,
        how="left",
        left_on=["individual"],
        right_on=["Donor ID"],
        validate="m:1",
    )
    .drop_duplicates(subset=["ENA_SAMPLE"])
    .reset_index(drop=True)
)

# %%
fetchngs_PRJEB32912 = pd.read_csv(
    f"{d10_1038_s41590_020_0602_z}/fetchngs/PRJEB32912/samplesheet/samplesheet.csv"
).dropna(axis=1, how="all")
fetchngs_PRJEB35119 = pd.read_csv(
    f"{d10_1038_s41590_020_0602_z}/fetchngs/PRJEB35119/samplesheet/samplesheet.csv"
).dropna(axis=1, how="all")

fetchngs_meta = pd.concat(
    [fetchngs_PRJEB32912, fetchngs_PRJEB35119], axis=0
).drop_duplicates(subset=["secondary_sample_accession"])

# %%
meta = pd.merge(
    meta,
    fetchngs_meta,
    how="left",
    left_on=["ENA_SAMPLE"],
    right_on=["secondary_sample_accession"],
    validate="m:1",
)
# Handle performance warning: DataFrame is highly fragmented
meta = meta.copy()

# %%
meta.to_csv(f"{artifact_dir}/James_2020-original_sample_meta.csv", index=False)


# %%
def load_counts(sample_meta, databases_path, mat_type="filtered"):

    file_path = f"{databases_path}/10_nfcore_scrnaseq/cellranger/count/{sample_meta['ENA_SAMPLE']}/outs/{mat_type}_feature_bc_matrix.h5"
    adata = sc.read_10x_h5(file_path)
    adata.obs = adata.obs.assign(**sample_meta)
    adata.var_names_make_unique()
    adata.X = csr_matrix(adata.X.astype(int))

    adata.obs["original_obs_names"] = adata.obs_names
    adata.obs_names = (
        adata.obs["ENA_SAMPLE"] + "-" + adata.obs_names.str.replace("-1", "")
    )
    assert adata.obs_names.is_unique
    return adata


# %%
adatas = process_map(
    load_counts,
    [r for _, r in meta.iterrows()],
    itertools.repeat(d10_1038_s41590_020_0602_z),
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
# ### Merge metadata from gutcellatlas_adata

# %%
adata.obs = adata.obs.assign(
    study_id="James_2020_Nat_Immunol",
    study_doi="10.1038/s41590-020-0602-z",
    study_pmid="32066951",
    tissue_processing_lab="Sarah A. Teichmann lab",
    medical_condition="healthy",
    dataset="James_2020",
    sample_type="normal",
    tumor_source="normal",
    matrix_type="raw counts",
    cellranger_version="7.1.0",
    reference_genome="gencode.v44",
    treatment_status_before_resection="naive",
    tissue_cell_state="fresh",
    hospital_location="Cambridge Biorepository of Translational Medicine",
    country="United Kingdom",
)

# %%
adata.obs["patient_id"] = adata.obs["individual"]
adata.obs["sample_id"] = adata.obs["Source Name"]

adata.obs["NCBI_BioProject_accession"] = adata.obs["study_accession"]
adata.obs["ENA_sample_accession"] = adata.obs["ENA_SAMPLE"]
adata.obs["SRA_sample_accession"] = adata.obs["secondary_sample_accession"]

adata.obs["age"] = adata.obs["Age"]

# define the dictionary mapping
mapping = {
    "caecum": "colon",
    "mesenteric lymph node": "lymph node",
    "sigmoid colon": "colon",
    "transverse colon": "colon",
}
adata.obs["sample_tissue"] = adata.obs["organism part"].map(mapping)

# define the dictionary mapping
mapping = {
    "caecum": "cecum",
    "mesenteric lymph node": "mesenteric lymph nodes",
    "sigmoid colon": "sigmoid colon",
    "transverse colon": "transverse colon",
}
adata.obs["anatomic_location"] = adata.obs["organism part"].map(mapping)

# define the dictionary mapping
mapping = {
    "cecum": "proximal colon",
    "mesenteric lymph nodes": "mesenteric lymph nodes",
    "sigmoid colon": "distal colon",
    "transverse colon": "proximal colon",
}
adata.obs["anatomic_region"] = adata.obs["anatomic_location"].map(mapping)

# %%
# define the dictionary mapping
mapping = {
    "10x5prime": "10x 5' v2",
    "10xV2": "10x 3' v2",
}
adata.obs["platform"] = adata.obs["library construction"].map(mapping)

# define the dictionary mapping
mapping = {
    "CD45+/CD3+/CD4+": "CD45+/CD3+/CD4+",
    "CD45+/CD3-/CD4-": "CD45+/CD3-/CD4-",
    "CD45 enriched": "CD45+",
    "unsorted": "naive",
}
adata.obs["enrichment_cell_types"] = adata.obs["immunophenotype"].map(mapping)

# %%
adata.obs["dataset"] = (
    adata.obs["dataset"].astype(str)
    + "_"
    + adata.obs["platform"].astype(str)
    + "_"
    + adata.obs["enrichment_cell_types"].astype(str)
)

mapping = {
    "James_2020_10x 3' v2_CD45+/CD3+/CD4+": "James_2020_10x3p_CD45Pos_CD3Pos_CD4Pos",
    "James_2020_10x 3' v2_CD45+/CD3-/CD4-": "James_2020_10x3p_CD45Pos_CD3Neg_CD4Neg",
    "James_2020_10x 5' v2_CD45+": "James_2020_10x5p_CD45Pos",
    "James_2020_10x 5' v2_naive": "James_2020_10x5p",
}
adata.obs["dataset"] = adata.obs["dataset"].map(mapping)

# %% [markdown]
# ### Get cell type mapping

# %%
# seems to be the same as Colon_cell_atlas.obs
meta = pd.read_csv(
    f"{d10_1038_s41590_020_0602_z}/gutcellatlas.org/Colon_immune_metadata.csv",
    low_memory=False,
)

# %%
Colon_cell_atlas = sc.read_h5ad(
    f"{d10_1038_s41590_020_0602_z}/gutcellatlas.org/Colon_cell_atlas.h5ad"
)

# %%
Colon_cell_atlas.obs["sample_id"] = Colon_cell_atlas.obs_names.str.rsplit("-", n=1).str[1]
Colon_cell_atlas.obs["cell_type_ids"] = Colon_cell_atlas.obs["sample_id"] + "-" + Colon_cell_atlas.obs_names.str.rsplit("-", n=2).str[0]

# %%
adata.obs["cell_type_ids"] = (
    adata.obs["Source Name"]
    + "-"
    + adata.obs["original_obs_names"].str.split("-").str[0]
)

# %%
# get dict
cell_type_mapping = (
    Colon_cell_atlas.obs[["cell_type_ids", "cell_type"]]
    .set_index("cell_type_ids")["cell_type"]
    .drop_duplicates()
    .to_dict()
)

# %%
adata.obs["cell_type_study"] = adata.obs["cell_type_ids"].map(cell_type_mapping)
adata.obs.loc[adata.obs.duplicated("cell_type_ids"), "cell_type_study"] = np.nan

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
adata.obs_names = adata.obs["dataset"] + "-" + adata.obs_names

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
adata

# %%
adata.obs["sample_id"].value_counts()

# %%
datasets = [
    adata[adata.obs["dataset"] == dataset, :].copy()
    for dataset in adata.obs["dataset"].unique()
    if adata[adata.obs["dataset"] == dataset, :].shape[0] != 0
]

# %%
for _adata in datasets:
    _adata.obs = _adata.obs.dropna(
        axis=1, how="all"
    )  # Remove all nan columns, else write_h5ad will fail
    dataset = _adata.obs["dataset"].values[0]  # Get the dataset name from the subset
    _adata.write_h5ad(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
