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
# # Dataloader: Liu_2024_Cancer_Res

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
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")

# %%
cpus = nxfvars.get("cpus", 8)
d10_1158_0008_5472_CAN_23_2123 = f"{databases_path}/d10_1158_0008_5472_CAN_23_2123"
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
def load_counts(sample_meta):

    adata = sc.AnnData(
        X=fmm.mmread(
            f"{d10_1158_0008_5472_CAN_23_2123}/GSE245552/GSE245552_RAW/{sample_meta['file']}_matrix.mtx.gz"
        ).T,
        obs=pd.read_csv(
            f"{d10_1158_0008_5472_CAN_23_2123}/GSE245552/GSE245552_RAW/{sample_meta['file']}_barcodes.tsv.gz",
            sep="\t",
            header=None,
            index_col=0,
        ),
        var=pd.read_csv(
            f"{d10_1158_0008_5472_CAN_23_2123}/GSE245552/GSE245552_RAW/{sample_meta['file']}_features.tsv.gz",
            delimiter="\t",
            header=None,
            index_col=0,
        ),
    )
    adata.X = csr_matrix(adata.X)
    adata.var_names.name = None
    adata.obs_names.name = None
    adata.obs["original_obs_names"] = adata.obs_names
    adata.obs = adata.obs.assign(**sample_meta)
    return adata


# %%
geofetch = pd.read_csv(
    f"{d10_1158_0008_5472_CAN_23_2123}/GSE245552/GSE245552_PEP/GSE245552_PEP_raw.csv"
).dropna(axis=1, how="all")

geofetch["file"] = (
    geofetch["sample_supplementary_file_1"]
    .str.rsplit("/", n=1)
    .str[1]
    .str.rsplit("_", n=1)
    .str[0]
)
geofetch["Patient number"] = geofetch["sample_title"].str.split("_").str[0]

# %%
patient_meta = pd.read_csv(f"{d10_1158_0008_5472_CAN_23_2123}/metadata/TableS1.csv")

meta = pd.merge(
    geofetch,
    patient_meta,
    how="left",
    on=["Patient number"],
    validate="m:1",
)

# %%
adatas = process_map(
    load_counts,
    [r for _, r in meta.iterrows()],
    max_workers=cpus,
)

adata = anndata.concat(adatas, join="outer")

# %% [markdown]
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
adata.var_names.name = None

# %%
unmapped_genes = ah.pp.find_unmapped_genes(adata)
len(unmapped_genes)

# %%
# Get original sample/patient meta and remove cell level meta
sample_meta = (
    adata.obs[
        adata.obs.columns[
            adata.obs.nunique() <= adata.obs["sample_geo_accession"].nunique()
        ]
    ]
    .groupby("sample_geo_accession")
    .first()
    .reset_index()
)

# %%
sample_meta.to_csv(
    f"{artifact_dir}/Liu_2024-original_sample_meta.csv",
    index=False,
)

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="Liu_2024_Cancer_Res",
    study_doi="10.1158/0008-5472.CAN-23-2123",
    study_pmid="38335276",
    tissue_processing_lab="Lixia Xu lab",
    dataset="Liu_2024_mixCD45PosCD45Neg",
    medical_condition="colorectal cancer",
    matrix_type="raw counts",
    reference_genome="ensembl.v98",
    cellranger_version="5.0.1",
    platform="10x 5' v2",
    hospital_location="The First Affiliated Hospital of Sun Yat-sen University, Guangzhou",
    country="China",
    # Paper methods: Paired tumor samples of primary and metastatic lesions were freshly obtained from resections or biopsies for scRNA-seq
    tissue_cell_state="fresh",
    # Paper methods: None of them received prior chemotherapy, radiation or any other anti-tumor therapy.
    treatment_status_before_resection="naive",
    # Paper methods: To obtain sufficient cells for immune components investigation, CD45+ and CD45- cells were sorted via magnetic-activated cell sorting and mixed at a ratio of 2:1 for each sample
    enrichment_cell_types="mixCD45+CD45-",
    NCBI_BioProject_accession="PRJNA1028966",
    # Paper methods: All patients were pathologically diagnosed as colorectal adenocarcinoma
    histological_type="adenocarcinoma",
)

# %%
adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]
adata.obs["sample_id"] = adata.obs["file"].str.split("_", n=1).str[1]
adata.obs["patient_id"] = adata.obs["Patient number"]

# define the dictionary mapping
mapping = {
    "Female": "female",
    "Male": "male",
}
adata.obs["sex"] = adata.obs["Gender"].map(mapping)

adata.obs["age"] = adata.obs["Age"].fillna(np.nan).astype("Int64")

mapping = {
    "primary tumor": "tumor",
    "colon adjacent tissue": "normal",
    "liver metastasis": "metastasis",
    "liver adjacent tissue": "normal",
}
adata.obs["sample_type"] = adata.obs["sample_source_name_ch1"].map(mapping)

mapping = {
    "primary tumor": "colon",
    "colon adjacent tissue": "colon",
    "liver metastasis": "liver",
    "liver adjacent tissue": "liver",
}
adata.obs["sample_tissue"] = adata.obs["sample_source_name_ch1"].map(mapping)

mapping = {
    "Transverse colon": "transverse colon",
    "Rectum": "rectum",
    "Ascending colon": "ascending colon",
    "Descending colon": "descending colon",
    "Sigmoid colon": "sigmoid colon",
}
adata.obs["anatomic_location"] = adata.obs["Location"].map(mapping)

mapping = {
    "Transverse colon": "proximal colon",
    "Rectum": "distal colon",
    "Ascending colon": "proximal colon",
    "Descending colon": "distal colon",
    "Sigmoid colon": "distal colon",
}
adata.obs["anatomic_region"] = adata.obs["Location"].map(mapping)

adata.obs["microsatellite_status"] = adata.obs["MSI status"]

mapping = {
    "Moderate-differentiated": "G2",
    "Low or moderate differentiated": "G1-G2",
}
adata.obs["tumor_grade"] = adata.obs["Grade"].map(mapping)

adata.obs["tumor_stage_TNM_T"] = "T" + adata.obs["pTNM: T"].astype(str)
adata.obs["tumor_stage_TNM_N"] = "N" + adata.obs["pTNM: N"].astype(str)
adata.obs["tumor_stage_TNM_M"] = "M" + adata.obs["pTNM: M"].astype(str)

adata.obs["tumor_stage_TNM"] = adata.obs["Stage"]

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
    + adata.obs["GEO_sample_accession"]
    + "-"
    + adata.obs_names.str.split("_").str[1].str.split("-").str[0]
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
# Save unfiltered raw counts
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
