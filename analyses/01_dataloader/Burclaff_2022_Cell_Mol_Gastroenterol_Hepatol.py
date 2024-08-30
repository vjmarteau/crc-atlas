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
#     display_name: Python [conda env:.conda-2024-crc-atas-scanpy]
#     language: python
#     name: conda-env-.conda-2024-crc-atas-scanpy-py
# ---

# %% [markdown]
# # Dataloader: Burclaff_2022_Cell_Mol_Gastroenterol_Hepatol

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
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")
cpus = nxfvars.get("cpus", 2)

# %%
d10_1016_j_jcmgh_2022_02_007 = f"{databases_path}/d10_1016_j_jcmgh_2022_02_007"
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
# ## 1. Load raw adata counts

# %%
def load_counts(file):

    adata = sc.read_10x_h5(
        f"{d10_1016_j_jcmgh_2022_02_007}/GSE185224/GSE185224_{file}_filtered_feature_bc_matrix.h5"
    )
    adata.X = csr_matrix(adata.X.astype(int))
    adata.var["symbol"] = adata.var_names
    adata.var_names = adata.var["gene_ids"]
    adata.var_names.name = None
    adata.obs_names.name = None
    adata.obs["patient_id"] = file
    return adata


# %%
adatas = process_map(
    load_counts,
    ["Donor1", "Donor2", "Donor3"],
    max_workers=cpus,
)
adata = anndata.concat(adatas, join="outer")

# %%
adata.obs_names = adata.obs["patient_id"] + "-" + adata.obs_names.str.split("-").str[0]
# adata.obs_names_make_unique()

gene_ids = adatas[0].var["symbol"].to_dict()
adata.var["symbol"] = adata.var_names.map(gene_ids)

# %% [markdown]
# ### sample/patient metadata

# %%
# not sure I can use this to get all labels ...
hashtags = pd.read_csv(
    f"{d10_1016_j_jcmgh_2022_02_007}/GSE185224/GSE185224_Donors_IntestinalRegions_Hashtags.txt.gz",
    delimiter="\t",
)

# %%
hashtags

# %%
meta = pd.read_csv(
    f"{d10_1016_j_jcmgh_2022_02_007}/metadata/Table2_donor_characteristics.csv"
)

# %%
geofetch = pd.read_csv(
    f"{d10_1016_j_jcmgh_2022_02_007}/GSE185224/GSE185224_PEP/GSE185224_PEP_raw.csv"
).dropna(axis=1, how="all")

geofetch["Donor"] = (
    geofetch["sample_title"].str.split(" pooled").str[0].str.replace(" ", "")
)
geofetch = geofetch.drop_duplicates(subset="Donor", keep="first")

# %%
meta = pd.merge(
    meta,
    geofetch,
    how="left",
    on=["Donor"],
    validate="m:1",
)

# %%
meta.to_csv(
    f"{artifact_dir}/Burclaff_2022-original_sample_meta.csv",
    index=False,
)

# %%
# Merge all available metadata
adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    meta,
    how="left",
    left_on=["patient_id"],
    right_on=["Donor"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
# merge cell meta from .h5ad file
annotated_adata = sc.read_h5ad(
    f"{d10_1016_j_jcmgh_2022_02_007}/GSE185224/GSE185224_clustered_annotated_adata_k10_lr0.92_v1.5.h5ad"
)
annotated_adata.obs_names = (
    annotated_adata.obs["donor"].str.replace(" ", "")
    + "-"
    + annotated_adata.obs_names.str.split("-").str[0]
)

# %%
adata.obs["obs_names"] = adata.obs_names
annotated_adata.obs["obs_names"] = annotated_adata.obs_names

for column in [col for col in annotated_adata.obs if col != "obs_names"]:
    adata.obs[column] = adata.obs["obs_names"].map(
        annotated_adata.obs.set_index("obs_names")[column]
    )

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="Burclaff_2022_Cell_Mol_Gastroenterol_Hepatol",
    study_doi="10.1016/j.jcmgh.2022.02.007",
    study_pmid="35176508",
    tissue_processing_lab="Scott T Magness lab",
    dataset="Burclaff_2022",
    medical_condition="healthy",
    # Intestinal tracts were obtained from 3 disease-free organ donors
    treatment_status_before_resection="naive",
    # Intestines were transported on ice
    tissue_cell_state="fresh",
    enrichment_cell_types="AnnexinV-",
    sample_type="normal",
    # Paper methods: Chromium Next GEM Single Cell 3’ GEM, Library and Gel Bead Kit v3.1
    platform="10x 3' v3",
    cellranger_version="4.0.0",
    reference_genome="ensembl.v98",
    matrix_type="raw counts",
    hospital_location="University of North Carolina at Chapel Hill, North Carolina",
    country="US",
    NCBI_BioProject_accession="PRJNA768275",
)

# %%
mapping = {
    "AC": "ascending colon",
    "DC": "descending colon",
    "TC": "transverse colon",
    "Ile": "ileum",
    "Jej": "jejunum",
    "Duo": "duodenum",
    "nan": "nan",
}
adata.obs["anatomic_location"] = (
    adata.obs["donor_region"].str.split("_").str[1].astype(str).map(mapping)
)

mapping = {
    "duodenum": "small intestine",
    "jejunum": "small intestine",
    "ileum": "small intestine",
    "transverse colon": "proximal colon",
    "descending colon": "distal colon",
    "ascending colon": "proximal colon",
    "nan": "nan",
}
adata.obs["anatomic_region"] = adata.obs["anatomic_location"].map(mapping)

mapping = {
    "small intestine": "small intestine",
    "distal colon": "colon",
    "proximal colon": "colon",
    "nan": "nan",
}
adata.obs["sample_tissue"] = adata.obs["anatomic_region"].map(mapping)

# %%
adata.obs["sample_id"] = adata.obs["donor_region"].str.replace(" ", "")

adata.obs["age"] = adata.obs["Age"].astype("Int64")
mapping = {
    "Female": "female",
    "Male": "male",
}
adata.obs["sex"] = adata.obs["Sex"].map(mapping)

mapping = {
    "African American": "black or african american",
    "White": "white",
}
adata.obs["ethnicity"] = adata.obs["Race"].map(mapping)

adata.obs["cell_type_study"] = adata.obs["lineage"]
adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]
adata.obs["SRA_sample_accession"] = adata.obs["big_key"]

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["GEO_sample_accession"]
    + "-"
    + adata.obs_names.str.split("-").str[1]
)

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
ref_meta_cols += [
    "hash_label",
]

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]]

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
