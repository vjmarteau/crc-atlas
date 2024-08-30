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
# # Dataloader: Han_2020_Nature

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
d10_1038_s41586_020_2157_4 = f"{databases_path}/d10_1038_s41586_020_2157_4"
gtf = f"{annotation}/ensembl.v86_gene_annotation_table.csv"

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
# -> fastq files available (downloaded)

# %%
def _load_mtx(file):
    counts = pd.read_csv(
        f"{d10_1038_s41586_020_2157_4}/GSE134355/GSE134355_RAW/{file}", sep="\t"
    )
    mat = counts.T
    mat.columns = mat.iloc[0]
    mat = mat[1:]
    mat = mat.rename_axis(None, axis=1)
    mat = mat.astype(int)

    adata = sc.AnnData(mat)
    adata.X = csr_matrix(adata.X)
    adata.obs["file"] = file

    return adata


# %%
files = [
    "GSM3943046_Adult-Adipose1_dge.txt.gz",
    "GSM3980125_Adult-Ascending-Colon1_dge.txt.gz",
    "GSM3980131_Adult-Duodenum1_dge.txt.gz",
    "GSM3980140_Adult-Ileum2_dge.txt.gz",
    "GSM3980141_Adult-Jejunum2_dge.txt.gz",
    "GSM4008638_Adult-Peripheral-Blood1_dge.txt.gz",
    "GSM4008639_Adult-Peripheral-Blood2_dge.txt.gz",
    "GSM4008640_Adult-Peripheral-Blood3-1_dge.txt.gz",
    "GSM4008641_Adult-Peripheral-Blood3-2_dge.txt.gz",
    "GSM4008642_Adult-Peripheral-Blood4-1_dge.txt.gz",
    "GSM4008643_Adult-Peripheral-Blood4-2_dge.txt.gz",
    "GSM4008644_Adult-Peripheral-Blood4-3_dge.txt.gz",
    "GSM4008647_Adult-Rectum1_dge.txt.gz",
    "GSM4008648_Adult-Sigmoid-Colon1_dge.txt.gz",
    "GSM4008662_Adult-Transverse-Colon1_dge.txt.gz",
    "GSM4008663_Adult-Transverse-Colon2-1_dge.txt.gz",
    "GSM4008664_Adult-Transverse-Colon2-2_dge.txt.gz",
]

# %%
adatas = process_map(_load_mtx, files, max_workers=cpus)
adata = anndata.concat(adatas, index_unique="-", join="outer", fill_value=0)
adata.obs_names = adata.obs_names.str.rsplit("-", n=1).str[0]
adata.obs["original_obs_names"] = adata.obs_names

# %%
geofetch = pd.read_csv(
    f"{d10_1038_s41586_020_2157_4}/GSE134355/GSE134355_PEP/GSE134355_PEP_raw.csv"
).dropna(axis=1, how="all")

fetchngs = pd.read_csv(
    f"{d10_1038_s41586_020_2157_4}/fetchngs/PRJNA554845/samplesheet/samplesheet.csv"
).dropna(axis=1, how="all")

meta = pd.merge(
    fetchngs,
    geofetch,
    how="left",
    left_on=["sample"],
    right_on=["srx"],
    validate="m:1",
)

# %%
adata.obs["sample_geo_accession"] = adata.obs["file"].str.split("_").str[0]

# Merge all available metadata
adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    meta,
    how="left",
    on=["sample_geo_accession"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
patient_meta = pd.read_csv(f"{d10_1038_s41586_020_2157_4}/metadata/tableS1.csv")

patient_meta = patient_meta.loc[
    patient_meta["Tissues/Cell lines"].isin(
        [
            "Adult Adipose1",
            "Adult Ascending Colon1",
            "Adult Duodenum1",
            "Adult Ileum2",
            "Adult Jejunum2",
            "Adult Periphera Blood1",
            "Adult Periphera Blood2",
            "Adult Periphera Blood3",
            "Adult Periphera Blood4",
            "Adult Rectum1",
            "Adult Sigmoid Colon1",
            "Adult Transverse Colon1",
            "Adult Transverse Colon2",
        ]
    )
]

# %%
adata.obs["patient_id"] = adata.obs["file"].str.split("_").str[1]
adata.obs["patient_id"] = adata.obs["patient_id"].str.replace("-1", "")
adata.obs["patient_id"] = adata.obs["patient_id"].str.replace("-2", "")
adata.obs["patient_id"] = adata.obs["patient_id"].str.replace("-3", "")
adata.obs["patient_id"] = adata.obs["patient_id"].str.replace("-", " ")
adata.obs["patient_id"] = adata.obs["patient_id"].str.replace("Adult Peripheral Blood", "Adult Periphera Blood")

# %%
# Merge all available metadata
adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    patient_meta,
    how="left",
    left_on=["patient_id"],
    right_on=["Tissues/Cell lines"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
# Get original sample/patient meta and remove cell level meta
sample_meta = (
    adata.obs[
        adata.obs.columns[adata.obs.nunique() <= adata.obs["sample_name"].nunique()]
    ]
    .groupby("sample_name")
    .first()
    .reset_index()
)

sample_meta.to_csv(f"{artifact_dir}/Han_2020-original_sample_meta.csv", index=False)

# %% [markdown] tags=[]
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

# %%
adata.obs = adata.obs.assign(
    study_id="Han_2020_Nature",
    study_doi="10.1038/s41586-020-2157-4",
    study_pmid="32214235",
    tissue_processing_lab="Guoji Guo lab",
    dataset="Han_2020",
    medical_condition="healthy",
    tumor_source="normal",
    platform="Microwell-seq",
    matrix_type="raw counts",
    reference_genome="ensembl.v86",
    enrichment_cell_types="naive",
    treatment_status_before_resection="naive",
    tissue_cell_state="fresh",
    NCBI_BioProject_accession="PRJNA554845",
    hospital_location="Zhejiang University School of Medicine, Hanzhou",
    country="China",
)

# %%
adata.obs["SRA_sample_accession"] = adata.obs["run_accession"]
adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]

# %%
adata.obs["patient_id"] = adata.obs["Donor"]
adata.obs["sample_id"] = adata.obs["sample_name"].str.replace("-", "_")

# %%
# define the dictionary mapping
mapping = {
    "adult-adipose1": "normal",
    "adult-ascending-colon1": "normal",
    "adult-duodenum1": "normal",
    "adult-ileum2": "normal",
    "adult-jejunum2": "normal",
    "adult-peripheral-blood1": "blood",
    "adult-peripheral-blood2": "blood",
    "adult-peripheral-blood3-1": "blood",
    "adult-peripheral-blood3-2": "blood",
    "adult-peripheral-blood4-1": "blood",
    "adult-peripheral-blood4-2": "blood",
    "adult-peripheral-blood4-3": "blood",
    "adult-rectum1": "normal",
    "adult-sigmoid-colon1": "normal",
    "adult-transverse-colon1": "normal",
    "adult-transverse-colon2-1": "normal",
    "adult-transverse-colon2-2": "normal",
}
adata.obs["sample_type"] = adata.obs["sample_name"].map(mapping)

# %%
# define the dictionary mapping
mapping = {
    "adult-adipose1": "adipose",
    "adult-ascending-colon1": "ascending colon",
    "adult-duodenum1": "duodenum",
    "adult-ileum2": "ileum",
    "adult-jejunum2": "jejunum",
    "adult-peripheral-blood1": "blood",
    "adult-peripheral-blood2": "blood",
    "adult-peripheral-blood3-1": "blood",
    "adult-peripheral-blood3-2": "blood",
    "adult-peripheral-blood4-1": "blood",
    "adult-peripheral-blood4-2": "blood",
    "adult-peripheral-blood4-3": "blood",
    "adult-rectum1": "rectum",
    "adult-sigmoid-colon1": "sigmoid colon",
    "adult-transverse-colon1": "transverse colon",
    "adult-transverse-colon2-1": "transverse colon",
    "adult-transverse-colon2-2": "transverse colon",
}
adata.obs["anatomic_location"] = adata.obs["sample_name"].map(mapping)


# define the dictionary mapping
mapping = {
    "adipose": "nan",
    "ascending colon": "proximal colon",
    "duodenum": "small intestine",
    "ileum": "small intestine",
    "jejunum": "small intestine",
    "blood": "blood",
    "rectum": "distal colon",
    "sigmoid colon": "distal colon",
    "transverse colon": "proximal colon",
}
adata.obs["anatomic_region"] = adata.obs["anatomic_location"].map(mapping)

# define the dictionary mapping
mapping = {
    "nan": "nan",
    "proximal colon": "colon",
    "distal colon": "colon",
    "small intestine": "small intestine",
    "blood": "blood",
}
adata.obs["sample_tissue"] = adata.obs["anatomic_region"].map(mapping)

# %%
adata.obs["age"] = adata.obs["Age"].str.split("-").str[0].astype("Int64")
mapping = {
    "Female": "female",
    "Male": "male",
}
adata.obs["sex"] = adata.obs["Gender"].map(mapping)

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
    + adata.obs["SRA_sample_accession"]
    + "-"
    + adata.obs_names
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
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
