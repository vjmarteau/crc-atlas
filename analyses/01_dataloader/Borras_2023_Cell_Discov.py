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
# # Dataloader: Borras_2023_Cell_Discov

# %%
import itertools
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
cpus = nxfvars.get("cpus", 6)
d10_1038_s41421_023_00605_4 = f"{databases_path}/d10_1038_s41421_023_00605_4"

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

# %% [markdown]
# ### New SMC5 cohort

# %%
adata_SMC5 = anndata.AnnData(
    X=fmm.mmread(f"{d10_1038_s41421_023_00605_4}/syn41687327/SMC5_matrix.mtx.gz"),
    obs=pd.read_csv(
        f"{d10_1038_s41421_023_00605_4}/syn41687327/SMC5_barcodes.tsv.gz",
        delimiter="\t",
        header=None,
        index_col=0,
    ),
    var=pd.read_csv(
        f"{d10_1038_s41421_023_00605_4}/syn41687327/SMC5_features.tsv.gz",
        delimiter="\t",
        header=None,
        index_col=0,
    ),
)
adata_SMC5.X = csr_matrix(adata_SMC5.X.astype(int))
adata_SMC5.var_names.name = None
adata_SMC5.obs_names.name = None

# %%
# load gtf for gene mapping -> will use same as for SMC cohort (dataset: Lee_2020_SMC)
gtf = f"{annotation}/ensembl.v84_gene_annotation_table.csv"
gtf = pd.read_csv(gtf)
gtf = ah.pp.append_duplicate_suffix(df=gtf, column="gene_name", sep=".")
gene_ids = gtf.set_index("gene_name")["gene_id"].to_dict()

# %%
adata_SMC5.var = adata_SMC5.var.rename_axis("symbol").reset_index()
adata_SMC5.var["symbol"] = adata_SMC5.var["symbol"].str.replace(
    "XXyac-YX65C7-A.2", "XXyac-YX65C7_A.2"
)
adata_SMC5.var["symbol"] = adata_SMC5.var["symbol"].str.replace(
    "XXyac-YX65C7-A.3", "XXyac-YX65C7_A.3"
)
adata_SMC5.var["ensembl"] = (
    adata_SMC5.var["symbol"]
    .str.replace("--", "__")
    .map(gene_ids)
    .fillna(value=adata_SMC5.var["symbol"])
)
adata_SMC5.var_names = adata_SMC5.var["ensembl"].apply(ah.pp.remove_gene_version)
adata_SMC5.var_names.name = None

# %%
# Did not figure out what annotation was used ...
unmapped_genes = ah.pp.find_unmapped_genes(adata_SMC5)
len(unmapped_genes)

# %%
cell_meta = pd.read_csv(
    f"{d10_1038_s41421_023_00605_4}/syn41687327/raw_obs_SMC5.tsv",
    sep="\t",
).rename(columns={"Unnamed: 0": "obs_names"})

# Merge accession numbers
adata_SMC5.obs["obs_names"] = adata_SMC5.obs_names
adata_SMC5.obs = pd.merge(
    adata_SMC5.obs,
    cell_meta,
    how="left",
    on=["obs_names"],
    validate="m:1",
).set_index("obs_names")
adata_SMC5.obs_names.name = None

# %%
patient_meta = pd.read_csv(f"{d10_1038_s41421_023_00605_4}/metadata/TableS1.csv")

# Merge accession numbers
adata_SMC5.obs["obs_names"] = adata_SMC5.obs_names
adata_SMC5.obs = pd.merge(
    adata_SMC5.obs,
    patient_meta,
    how="left",
    on=["patient"],
    validate="m:1",
).set_index("obs_names")
adata_SMC5.obs_names.name = None

# %%
adata_SMC5.obs = adata_SMC5.obs.assign(
    study_id="Borras_2023_Cell_Discov",
    study_doi="10.1038/s41421-023-00605-4",
    study_pmid="37968259",
    # According to Figure 1 this is also a South Korea cohort
    tissue_processing_lab="Woong-Yang Park lab",
    dataset="Borras_2023_SMC5",
    medical_condition="colorectal cancer",
    matrix_type="raw counts",
    # reference_genome="",
    # Paper methods: sequencing libraries were generated using Chromium Next GEM Single Cell 5′ v1.1 Reagent Kits
    platform="10x 5' v1",
    cellranger_version="3.1.0",
    hospital_location="Samsung Genome Institute, Samsung Medical Center, Seoul",
    country="South Korea",
    # Paper methods: cryopreserved in CELLBANKER 1 (Zenoaq Pharma) before scRNA-seq
    tissue_cell_state="frozen",
    # Paper methods: although only CD8 T cells are made available !
    enrichment_cell_types="naive",
    sample_type="tumor",
    sample_tissue="colon",
    # treatment_status_before_resection="", ? naive -> same as previous SMC cohort?
)

# %%
# No sex or age available?
adata_SMC5.obs["patient_id"] = adata_SMC5.obs["patient"]
adata_SMC5.obs["sample_id"] = adata_SMC5.obs["sample"]
adata_SMC5.obs["original_obs_names"] = adata_SMC5.obs["Unnamed: 0.1"]
adata_SMC5.obs["microsatellite_status"] = adata_SMC5.obs["MSI"]
adata_SMC5.obs["cell_type_study"] = adata_SMC5.obs["Tmajor"]

# %%
# define the dictionary mapping
mapping = {
    "ascending": "ascending colon",
    "sigmoid": "sigmoid colon",
    "Ascending colon": "ascending colon",
    "Hepatic flexure colon": "hepatic flexure",
    "Proximal transverse colon": "transverse colon",
    "Splenic flexure colon": "splenic flexure",
}
adata_SMC5.obs["anatomic_location"] = adata_SMC5.obs["Unnamed: 7"].map(mapping)

# define the dictionary mapping
mapping = {
    "ascending colon": "proximal colon",
    "hepatic flexure": "proximal colon",
    "transverse colon": "proximal colon",
    "splenic flexure": "distal colon",
    "sigmoid colon": "distal colon",
}
adata_SMC5.obs["anatomic_region"] = adata_SMC5.obs["anatomic_location"].map(mapping)

# %%
# define the dictionary mapping
mapping = {
    "Adenocarcinoma_moderately_differentiated": "adenocarcinoma",
    "Adenocarcinoma_well_differentiated": "adenocarcinoma",
    "AdenocarcinomaMD": "adenocarcinoma",
    "Mucinouscarcinoma": "mucinous adenocarcinoma",
    "AdenocarcinomaWD": "adenocarcinoma",
    "AdenocarcinomaPD": "adenocarcinoma",
}
adata_SMC5.obs["histological_type"] = adata_SMC5.obs["Unnamed: 10"].map(mapping)

# define the dictionary mapping
mapping = {
    "Adenocarcinoma_moderately_differentiated": "G2",
    "Adenocarcinoma_well_differentiated": "G1",
    "AdenocarcinomaMD": "G2",  # MD: Moderately differentiated
    "Mucinouscarcinoma": "nan",
    "AdenocarcinomaWD": "G1",  # WD: Well differentiated
    "AdenocarcinomaPD": "G3",  # PD: Poorly differentiated
}
adata_SMC5.obs["tumor_grade"] = adata_SMC5.obs["Unnamed: 10"].map(mapping)

# %%
adata_SMC5.obs["tumor_stage_TNM_T"] = adata_SMC5.obs["Unnamed: 11"].str.replace(
    "t", "T"
)
adata_SMC5.obs["tumor_stage_TNM_N"] = adata_SMC5.obs["Unnamed: 12"].str.replace(
    "n", "N"
)
adata_SMC5.obs["tumor_stage_TNM_M"] = adata_SMC5.obs["Unnamed: 13"].str.replace(
    "m", "M"
)
adata_SMC5.obs["tumor_stage_TNM"] = adata_SMC5.obs["Unnamed: 14"]

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
# Subset adata for columns in the reference meta
adata_SMC5.obs = adata_SMC5.obs[
    [col for col in ref_meta_cols if col in adata_SMC5.obs.columns]
].copy()

# %%
adata_SMC5.obs_names = (
    adata_SMC5.obs["dataset"]
    + "-"
    + adata_SMC5.obs["sample_id"]
    + "-"
    + adata_SMC5.obs_names.str.split("_", n=2).str[2]
)

# %%
assert np.all(np.modf(adata_SMC5.X.data)[0] == 0), "X does not contain all integers"
assert adata_SMC5.var_names.is_unique
assert adata_SMC5.obs_names.is_unique

# %%
adata_SMC5

# %%
adata_SMC5.obs["sample_id"].value_counts()

# %%
# Save unfiltered raw counts
dataset = adata_SMC5.obs["dataset"].unique()[0]
adata_SMC5.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")

# %% [markdown]
# ### New enriched KUL5 samples

# %%
adata_KUL5 = anndata.AnnData(
    X=fmm.mmread(f"{d10_1038_s41421_023_00605_4}/syn41687327/KUL5_matrix.mtx.gz"),
    obs=pd.read_csv(
        f"{d10_1038_s41421_023_00605_4}/syn41687327/KUL5_barcodes.tsv.gz",
        delimiter="\t",
        header=None,
        index_col=0,
    ),
    var=pd.read_csv(
        f"{d10_1038_s41421_023_00605_4}/syn41687327/KUL5_features.tsv.gz",
        delimiter="\t",
        header=None,
        index_col=0,
    ),
)
adata_KUL5.X = csr_matrix(adata_KUL5.X.astype(int))
adata_KUL5.var_names.name = None
adata_KUL5.obs_names.name = None

# %%
# load gtf for gene mapping -> will use same as for original KUL5 cohort (dataset: Joanito_2022_KUL5)
gtf = f"{annotation}/ensembl.v93_gene_annotation_table.csv"
gtf = pd.read_csv(gtf)
gtf = ah.pp.append_duplicate_suffix(df=gtf, column="gene_name", sep=".")
gene_ids = gtf.set_index("gene_name")["gene_id"].to_dict()

# %%
adata_KUL5.var = adata_KUL5.var.rename_axis("symbol").reset_index()
adata_KUL5.var["symbol"] = adata_KUL5.var["symbol"].str.replace(
    "XXyac-YX65C7-A.2", "XXyac-YX65C7_A.2"
)
adata_KUL5.var["symbol"] = adata_KUL5.var["symbol"].str.replace(
    "XXyac-YX65C7-A.3", "XXyac-YX65C7_A.3"
)
adata_KUL5.var["ensembl"] = (
    adata_KUL5.var["symbol"]
    .str.replace("--", "__")
    .map(gene_ids)
    .fillna(value=adata_KUL5.var["symbol"])
)
adata_KUL5.var_names = adata_KUL5.var["ensembl"].apply(ah.pp.remove_gene_version)
adata_KUL5.var_names.name = None

# %%
# Did not figure out what annotation was used ...
unmapped_genes = ah.pp.find_unmapped_genes(adata_KUL5)
len(unmapped_genes)

# %%
cell_meta = pd.read_csv(
    f"{d10_1038_s41421_023_00605_4}/syn41687327/raw_obs_KUL5.tsv",
    sep="\t",
).rename(columns={"Unnamed: 0": "obs_names"})

# Merge accession numbers
adata_KUL5.obs["obs_names"] = adata_KUL5.obs_names
adata_KUL5.obs = pd.merge(
    adata_KUL5.obs,
    cell_meta,
    how="left",
    on=["obs_names"],
    validate="m:1",
).set_index("obs_names")
adata_KUL5.obs_names.name = None

# subset for new enriched samples only
# Paper methods: An additional enrichment of T cells was performed for two patients (SC040 and SC044) of the KUL5 dataset using and the REAlease® CD4/CD8 (TIL) MicroBead Kit (Miltenyi Biotec), prior to scRNA-seq
adata_KUL5 = adata_KUL5[adata_KUL5.obs["enriched"] == "CD45"].copy()

# %%
# Merge accession numbers
adata_KUL5.obs["obs_names"] = adata_KUL5.obs_names
adata_KUL5.obs = pd.merge(
    adata_KUL5.obs,
    patient_meta,
    how="left",
    on=["patient"],
    validate="m:1",
).set_index("obs_names")
adata_KUL5.obs_names.name = None

# %%
adata_KUL5.obs = adata_KUL5.obs.assign(
    study_id="Borras_2023_Cell_Discov",
    study_doi="10.1038/s41421-023-00605-4",
    study_pmid="37968259",
    tissue_processing_lab="Sabine Tejpar lab",
    dataset="Borras_2023_KUL5_CD45Pos",
    medical_condition="colorectal cancer",
    matrix_type="raw counts",
    # reference_genome="",
    platform="10x 5' v3",
    cellranger_version="3.1.0",
    hospital_location="Department of Oncology, Katholieke Universiteit Leuven",
    country="Belgium",
    tissue_cell_state="fresh",
    # Paper methods: although only CD8 T cells are made available !
    enrichment_cell_types="CD45+",
    sample_tissue="colon",
)

# %%
adata_KUL5.obs["patient_id"] = adata_KUL5.obs["patient"]
adata_KUL5.obs["sample_id"] = adata_KUL5.obs["sample"].str.upper()
adata_KUL5.obs["original_obs_names"] = adata_KUL5.obs["Unnamed: 0.1"]
adata_KUL5.obs["microsatellite_status"] = adata_KUL5.obs["MSI"].map({"MSI": "MSI-H", "MSS": "MSS"}) # Same as in Joanito_2022_KUL5
adata_KUL5.obs["cell_type_study"] = adata_KUL5.obs["Tmajor"]

# %%
# define the dictionary mapping
mapping = {
    "Core": "core",
    "Border": "border",
    "Normal": "normal",
    "Invasive": "border",
}
adata_KUL5.obs["tumor_source"] = adata_KUL5.obs["tissue"].map(mapping)

# define the dictionary mapping
mapping = {
    "Core": "tumor",
    "Border": "tumor",
    "Normal": "normal",
    "Invasive": "tumor",
}
adata_KUL5.obs["sample_type"] = adata_KUL5.obs["tissue"].map(mapping)

# define the dictionary mapping
mapping = {
    "ascending": "ascending colon",
    "caecum": "cecum",
}
adata_KUL5.obs["anatomic_location"] = adata_KUL5.obs["Unnamed: 7"].map(mapping)

# define the dictionary mapping
mapping = {
    "ascending colon": "proximal colon",
    "cecum": "proximal colon",
}
adata_KUL5.obs["anatomic_region"] = adata_KUL5.obs["anatomic_location"].map(mapping)

# %%
# define the dictionary mapping
mapping = {
    "Moderately_differentiated_adenocarcinoma_NST": "adenocarcinoma",
    "Low_differentiated_adenocarcinoma": "adenocarcinoma",
}
adata_KUL5.obs["histological_type"] = adata_KUL5.obs["Unnamed: 10"].map(mapping)

# define the dictionary mapping
mapping = {
    "Moderately_differentiated_adenocarcinoma_NST": "G2",
    "Low_differentiated_adenocarcinoma": "G1",
}
adata_KUL5.obs["tumor_grade"] = adata_KUL5.obs["Unnamed: 10"].map(mapping)

# %%
adata_KUL5.obs["tumor_stage_TNM_T"] = adata_KUL5.obs["Unnamed: 11"].str.replace(
    "t", "T"
)
adata_KUL5.obs["tumor_stage_TNM_N"] = adata_KUL5.obs["Unnamed: 12"].str.replace(
    "n", "N"
)
adata_KUL5.obs["tumor_stage_TNM_M"] = adata_KUL5.obs["Unnamed: 13"].str.replace(
    "m", "M"
)
adata_KUL5.obs["tumor_stage_TNM"] = adata_KUL5.obs["Unnamed: 14"].map({"II": "IIA", "III": "IIIB"}) # Same as in Joanito_2022_KUL5

# %%
# Meta from dataset Joanito_2022_KUL5

adata_KUL5.obs["sex"] = "female"

# define the dictionary mapping
mapping = {
    "SC040": 68,
    "SC044": 80,
}
adata_KUL5.obs["age"] = adata_KUL5.obs["patient_id"].map(mapping)

# define the dictionary mapping
mapping = {
    "SC040": "mut",
    "SC044": "wt",
}
adata_KUL5.obs["KRAS_status_driver_mut"] = adata_KUL5.obs["patient_id"].map(mapping)

mapping = {
    "SC040": "mut",
    "SC044": "wt",
}
adata_KUL5.obs["APC_status_driver_mut"] = adata_KUL5.obs["patient_id"].map(mapping)

adata_KUL5.obs["BRAF_status_driver_mut"] = "wt"
adata_KUL5.obs["TP53_status_driver_mut"] = "wt"
adata_KUL5.obs["sex"] = "female"
adata_KUL5.obs["treatment_status_before_resection"] = "naive"

# %%
# Subset adata for columns in the reference meta
adata_KUL5.obs = adata_KUL5.obs[
    [col for col in ref_meta_cols if col in adata_KUL5.obs.columns]
].copy()

# %%
adata_KUL5.obs_names = (
    adata_KUL5.obs["dataset"]
    + "-"
    + adata_KUL5.obs["sample_id"]
    + "-"
    + adata_KUL5.obs_names.str.split("_", n=2).str[2]
)

# %%
assert np.all(np.modf(adata_KUL5.X.data)[0] == 0), "X does not contain all integers"
assert adata_KUL5.var_names.is_unique
assert adata_KUL5.obs_names.is_unique

# %%
adata_KUL5

# %%
adata_KUL5.obs["sample_id"].value_counts()

# %%
# Save unfiltered raw counts
dataset = adata_KUL5.obs["dataset"].unique()[0]
adata_KUL5.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
