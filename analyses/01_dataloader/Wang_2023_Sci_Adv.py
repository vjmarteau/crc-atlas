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
#     display_name: Python [conda env:.conda-2022-calpro-scanpy]
#     language: python
#     name: conda-env-.conda-2022-calpro-scanpy-py
# ---

# %% [markdown]
# # Dataloader: Wang_2023_Sci_Adv

# %%
from pathlib import Path
import os

import anndata
import fast_matrix_market as fmm
import numpy as np
import pandas as pd
import sc_atlas_helpers as ah
import scanpy as sc
import scipy.sparse
import yaml
from nxfvars import nxfvars
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits
from tqdm.contrib.concurrent import process_map

# %% tags=["parameters"]
artifact_dir = nxfvars.get(
    "artifact_dir", "/data/scratch/marteau/downloads/crc-atlas/"
)
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")
cpus = nxfvars.get("cpus", 6)

# %%
d10_1126_sciadv_adf5464 = f"{databases_path}/d10_1126_sciadv_adf5464"
gtf = f"{annotation}/ensembl.v94_gene_annotation_table.csv"
GSE225857 = f"{d10_1126_sciadv_adf5464}/GSE225857"

# %%
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

# %%
# Read the YAML file and load it into a dictionary
with open(reference_meta, "r") as f:
    ref_meta_dict = yaml.load(f, Loader=yaml.Loader)

# %% [markdown]
# ## BD Rhapsody Data

# %%
# load CD45+ immune adata
immune_adata = sc.read_h5ad(f"{d10_1126_sciadv_adf5464}/GSE225857/immune_adata.h5ad")

immune_meta = (
    pd.read_csv(
        f"{GSE225857}/GSE225857_RAW/GSM7058754_immune_meta.txt.gz",
        delimiter="\t",
        index_col=0,
    )
    .reset_index()
    .rename(columns={"index": "obs_names"})
)

# Merge accession numbers
immune_adata.obs["obs_names"] = immune_adata.obs_names
immune_adata.obs = pd.merge(
    immune_adata.obs,
    immune_meta,
    how="left",
    on=["obs_names"],
    validate="m:1",
).set_index("obs_names")
immune_adata.obs_names.name = None

immune_adata.obs["patients"] = immune_adata.obs["patients"].str.replace("s", "P")
immune_adata.obs["enrichment_cell_types"] = "CD45+"
immune_adata.obs["GEO_sample_accession"] = "GSM7058754"


# Recreate sample ids used in study figures
# From paper supplements Table s3: CN: Colorectal normal tissue; CC: Colorectal cancer; LN: Liver normal tissue; LM: Liver metastasis; PB: Peripheral blood
# -> There seems to be an aditional patient "P0816" with additional samples liver metastasis, matched normal liver, and pbmcs, no primary CRC samples !

mapping = {
    "LCL": "LM",
    "CCL": "CC",
    "PBL": "PB",
    "LNL": "LN",
    "CNL": "CN",  # missing for P0920
}
immune_adata.obs["sample_id"] = immune_adata.obs["organs"].map(mapping)

mapping = {
    "LCL": "metastasis",
    "CCL": "tumor",
    "PBL": "blood",
    "LNL": "normal",
    "CNL": "normal",
}
immune_adata.obs["sample_type"] = immune_adata.obs["organs"].map(mapping)

mapping = {
    "LCL": "liver",
    "CCL": "colon",
    "PBL": "blood",
    "LNL": "liver",
    "CNL": "colon",
}
immune_adata.obs["sample_tissue"] = immune_adata.obs["organs"].map(mapping)

immune_adata.obs["sample_id"] = immune_adata.obs["sample_id"] + immune_adata.obs[
    "patients"
].str.replace("P", "")

# %%
# load CD45- non-immune adata
non_immune_adata = sc.read_h5ad(
    f"{d10_1126_sciadv_adf5464}/GSE225857/non_immune_adata.h5ad"
)
non_immune_adata.obs_names = non_immune_adata.obs_names.str.replace(
    ".", "-", regex=False
)

non_immune_meta = (
    pd.read_csv(
        f"{GSE225857}/GSE225857_RAW/GSM7058755_non_immune_meta.txt.gz",
        delimiter="\t",
        index_col=0,
    )
    .reset_index()
    .rename(columns={"index": "obs_names"})
)

# Merge accession numbers
non_immune_adata.obs["obs_names"] = non_immune_adata.obs_names
non_immune_adata.obs = pd.merge(
    non_immune_adata.obs,
    non_immune_meta,
    how="left",
    on=["obs_names"],
    validate="m:1",
).set_index("obs_names")
non_immune_adata.obs_names.name = None

non_immune_adata.obs["patients"] = non_immune_adata.obs["patients"].str.replace(
    "s", "P"
)
non_immune_adata.obs["enrichment_cell_types"] = "CD45-"
non_immune_adata.obs["GEO_sample_accession"] = "GSM7058755"


# Recreate sample ids used in study figures
# From paper supplements Table s4: CD45- available only for "CC" and "LM"
mapping = {
    "CCT": "CC",
    "LCT": "LM",
}
non_immune_adata.obs["sample_id"] = non_immune_adata.obs["organs"].map(mapping)

mapping = {
    "CCT": "tumor",
    "LCT": "metastasis",
}
non_immune_adata.obs["sample_type"] = non_immune_adata.obs["organs"].map(mapping)

mapping = {
    "CCT": "colon",
    "LCT": "liver",
}
non_immune_adata.obs["sample_tissue"] = non_immune_adata.obs["organs"].map(mapping)

non_immune_adata.obs["sample_id"] = (
    non_immune_adata.obs["sample_id"]
    + non_immune_adata.obs["patients"].str.replace("P", "")
    + "_"
    + non_immune_adata.obs["enrichment_cell_types"]
)

# %%
adata = anndata.concat(
    [immune_adata, non_immune_adata],
    index_unique=".",
    join="outer",
    fill_value=0,
)

adata.obs_names = adata.obs_names.str.split(".").str[0]
adata.obs_names_make_unique()

# something weird has happend to ".", and "-", in both .obs_names and .var_names ...
adata.var_names = adata.var_names.str.replace(".", "-", regex=False)

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
duplicated_ids = adata.var_names[adata.var_names.duplicated()].unique()
adata = ah.pp.aggregate_duplicate_gene_ids(adata, duplicated_ids)

# %%
unmapped_genes = ah.pp.find_unmapped_genes(adata)
len(unmapped_genes)

# %%
unmapped_genes

# %%
# update gene_ids dict to include missing genes "unmapped_genes"
# For some reason they exist as CH17-340M24.3 etc in the original ensembl.v94 gtf
gene_ids.update(
    {
        "CTD-3080P12-3": "ENSG00000249201.2",
        "CTD-2201I18-1": "ENSG00000249825.5",
        "CTB-178M22-2": "ENSG00000253978.1",
        "CTD-2270F17-1": "ENSG00000253647.1",
        "GS1-124K5-4": "ENSG00000237310.1",
        "GS1-594A7-3": "ENSG00000225833.1",
        "CH17-340M24-3": "ENSG00000197180.2",
        "GS1-24F4-2": "ENSG00000245857.2",
        "LL22NC03-63E9-3": "ENSG00000220891.1",
    }
)

# %%
adata.var_names = adata.var["symbol"]
adata.var_names.name = None

del adata.var["ensembl"]
del adata.var["symbol"]

# %%
adata.var = adata.var.rename_axis("symbol").reset_index()
adata.var["ensembl"] = (
    adata.var["symbol"].map(gene_ids).fillna(value=adata.var["symbol"])
)
adata.var_names = adata.var["ensembl"].apply(ah.pp.remove_gene_version)
adata.var_names.name = None

# %%
duplicated_ids = adata.var_names[adata.var_names.duplicated()].unique()
adata = ah.pp.aggregate_duplicate_gene_ids(adata, duplicated_ids)

# %%
unmapped_genes = ah.pp.find_unmapped_genes(adata)
len(unmapped_genes)

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
# patient metadata from paper supplements Table s2: Clinical and demographic details of patients
mapping = {
    "P0107": "75",
    "P0115": "80",
    "P0813": "58",
    "P0920": "64",
    "P1125": "41",
    "P1231": "58",
    #   "P0816": "",
}
adata.obs["age"] = adata.obs["patients"].map(mapping)

mapping = {
    "P0107": "male",
    "P0115": "male",
    "P0813": "female",
    "P0920": "male",
    "P1125": "female",
    "P1231": "male",
    #   "P0816": "",
}
adata.obs["sex"] = adata.obs["patients"].map(mapping)

mapping = {
    "P0107": "distal colon",  # left colon
    "P0115": "distal colon",
    "P0813": "proximal colon",  # right colon
    "P0920": "distal colon",
    "P1125": "distal colon",
    "P1231": "distal colon",
    #   "P0816": "",
}
adata.obs["anatomic_region"] = adata.obs["patients"].map(mapping)

# %%
# Paper methods: All the patients received preoperative chemotherapy and/or radiotherapy.

# patient metadata from paper supplements Table s3: Neoadjuvant regimens of the included patients
# FOLFOX: Fluorouracil+Calcium levofolinate+Oxaliplatin; FOLFOXIRI: Fluorouracil+Calcium levofolinate+Oxaliplatin+Irinotecan
mapping = {
    "P0107": "3x FOLFOX",
    "P0115": "4x FOLFOX",
    "P0813": "5x FOLFOXIRI",
    "P0920": "5x FOLFOX",
    "P1125": "8x FOLFOXIRI+Bevacizumab",
    "P1231": "3x FOLFOX",
}
adata.obs["treatment_drug"] = adata.obs["patients"].map(mapping)

# %%
# Get original sample/patient meta and remove cell level meta
sample_meta = (
    adata.obs[adata.obs.columns[adata.obs.nunique() <= adata.obs["sample_id"].nunique()]]
    .groupby("sample_id")
    .first()
    .reset_index()
)

# %%
sample_meta.to_csv(
    f"{artifact_dir}/Wang_2023-original_sample_meta.csv",
    index=False,
)

# %%
adata.obs = adata.obs.assign(
    study_id="Wang_2023_Sci_Adv",
    study_doi="10.1126/sciadv.adf5464",
    study_pmid="37327339",
    tissue_processing_lab="Zhi-Bin Zhao lab",
    dataset="Wang_2023",
    medical_condition="colorectal cancer",
    NCBI_BioProject_accession="PRJNA937723",
    hospital_location="Guang-zhou First People’s Hospital | Sixth Affiliated Hospital of Sun Yat-sen University",
    country="China",
    tumor_stage_TNM_T="nan",
    tumor_stage_TNM_N="nan",
    tumor_stage_TNM_M="M1",
    treatment_status_before_resection="treated",
    matrix_type="raw counts",
    platform="BD Rhapsody",
    reference_genome="ensembl.v94",
    tissue_cell_state="fresh"
)

adata.obs["patient_id"] = adata.obs["patients"]
adata.obs["tumor_stage_TNM"] = ah.pp.tnm_tumor_stage(adata.obs, "patient_id")
adata.obs["cell_type_study"] = adata.obs["cluster"]

adata.obs["dataset"] = adata.obs["dataset"].astype(str) + "_" + adata.obs["enrichment_cell_types"].astype(str)

mapping = {
    "Wang_2023_CD45+": "Wang_2023_CD45Pos",
    "Wang_2023_CD45-": "Wang_2023_CD45Neg",
}
adata.obs["dataset"] = adata.obs["dataset"].map(mapping)

adata.obs_names = adata.obs["dataset"] + "-" + adata.obs_names

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
ref_meta_cols += ["sampletag", "doublet.score", "predicted.doublet", "doublet"]

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
adata.obs["sample_id"] = adata.obs["sample_id"].str.replace("-", "_")
adata.obs["sample_id"] = adata.obs["sample_id"].str.rstrip("_CD45_")

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
adata

# %% [markdown]
# ## 4. Save adata by dataset

# %%
datasets = [
    adata[adata.obs["dataset"] == dataset, :].copy()
    for dataset in adata.obs["dataset"].unique()
    if adata[adata.obs["dataset"] == dataset, :].shape[0] != 0
]

# %%
datasets

# %%
for _adata in datasets:
    _adata.obs = _adata.obs.dropna(axis=1, how="all")  # Remove all nan columns, else write_h5ad will fail
    dataset = _adata.obs["dataset"].values[0]  # Get the dataset name from the subset
    _adata.write_h5ad(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")


# %% [markdown] tags=[]
# ## Spatial data

# %%
def _load_mtx(stem):
    # Load data into memory
    matrix = fmm.mmread(Path(f"{GSE225857}/GSE225857_RAW/{stem}.matrix.mtx.gz"))

    barcodes = pd.read_csv(
        Path(f"{GSE225857}/GSE225857_RAW/{stem}.barcodes.tsv.gz"),
        delimiter="\t",
        header=None,
        names=["barcodes"],
    )
    features = pd.read_csv(
        Path(f"{GSE225857}/GSE225857_RAW/{stem}.features.tsv.gz"),
        delimiter="\t",
        header=None,
        names=["var_names", "symbol", "0"],
    )

    # Assemble adata
    adata = sc.AnnData(X=matrix.T)
    adata.X = csr_matrix(adata.X)

    adata.obs = barcodes
    adata.obs_names = adata.obs["barcodes"]
    adata.obs_names.name = None

    adata.var = features
    adata.var_names = adata.var["var_names"]
    adata.var_names.name = None

    adata.obs["sample"] = stem

    return adata


# %%
files = [x.stem for x in Path(f"{GSE225857}/GSE225857_RAW/").glob("*.tsv.gz")]
files = list({item.split(".")[0] for item in files})
files.sort()

# %%
# No need to load spatial data
#adatas = process_map(_load_mtx, files, max_workers=cpus)
#adata = anndata.concat(adatas, index_unique="-", join="outer", fill_value=0)
