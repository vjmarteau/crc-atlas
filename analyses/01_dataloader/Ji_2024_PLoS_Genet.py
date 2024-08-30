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
# # Dataloader: Ji_2024_PLoS_Genet

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
d10_1371_journal_pgen_1011176 = f"{databases_path}/d10_1371_journal_pgen_1011176"
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
def load_counts(file):

    adata = sc.AnnData(
        X=fmm.mmread(
            f"{d10_1371_journal_pgen_1011176}/GSE217774/GSE217774_RAW/{file}_matrix.mtx.gz"
        ).T,
        obs=pd.read_csv(
            f"{d10_1371_journal_pgen_1011176}/GSE217774/GSE217774_RAW/{file}_barcodes.tsv.gz",
            header=None,
            index_col=0,
        ),
        var=pd.read_csv(
            f"{d10_1371_journal_pgen_1011176}/GSE217774/GSE217774_RAW/{file}_features.tsv.gz",
            delimiter="\t",
            header=None,
            index_col=0,
            names=["ensembl", "symbol", "type"],
        ),
    )
    adata.X = csr_matrix(adata.X)
    adata.var_names.name = None
    adata.obs_names.name = None
    adata.obs["file"] = file
    adata.obs["patient_id"] = adata.obs["file"].str.split("_").str[1]
    adata.obs["sample_geo_accession"] = adata.obs["file"].str.split("_").str[0]
    return adata


# %%
adatas = process_map(
    load_counts,
    ["GSM6726591_WHU1", "GSM6726592_WHU2", "GSM6726593_WHU3"],
    max_workers=cpus,
)
adata = anndata.concat(adatas, join="outer")

adata.obs_names = adata.obs_names.str.split("-").str[0]
adata.obs_names_make_unique()

gene_ids = adatas[0].var["symbol"].to_dict()
adata.var["symbol"] = adata.var_names.map(gene_ids)

# %%
meta = pd.read_csv(f"{d10_1371_journal_pgen_1011176}/metadata/Table1.csv")

adata.obs = adata.obs.reset_index(names="obs_names")
adata.obs = pd.merge(
    adata.obs,
    meta,
    how="left",
    on=["patient_id"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
geofetch = pd.read_csv(
    f"{d10_1371_journal_pgen_1011176}/GSE217774/GSE217774_PEP/GSE217774_PEP_raw.csv"
).dropna(axis=1, how="all")

adata.obs = adata.obs.reset_index(names="obs_names")
adata.obs = pd.merge(
    adata.obs,
    geofetch,
    how="left",
    on=["sample_geo_accession"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %% [markdown]
# ## 2. Gene annotations

# %%
# load gtf for gene mapping
gtf = pd.read_csv(gtf)
gtf = ah.pp.append_duplicate_suffix(df=gtf, column="gene_name", sep=".")
gtf["ensembl"] = gtf["gene_id"].apply(ah.pp.remove_gene_version)
gene_ids = gtf.set_index("ensembl")["gene_id"].to_dict()

# %%
adata.var = adata.var.rename_axis("var_names").reset_index()
adata.var["ensembl"] = (
    adata.var["var_names"].map(gene_ids).fillna(value=adata.var["symbol"])
)
adata.var_names = adata.var["var_names"]
adata.var_names.name = None

# %%
# Reorder by gtf (i.e. chromosome position)
gene_index = gtf[gtf["ensembl"].isin(adata.var_names)]["ensembl"].values
adata = adata[:, gene_index]

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="Ji_2024_PLoS_Genet",
    study_doi="10.1371/journal.pgen.1011176",
    study_pmid="38408082",
    tissue_processing_lab="Haoyu Fu lab",
    dataset="Ji_2024_CD14Pos",
    medical_condition="colorectal cancer",
    enrichment_cell_types="CD14+",
    # Paper method: Tumor tissues were immediately cut ...
    tissue_cell_state="fresh",
    sample_type="tumor",
    sample_tissue="colon",
    # Paper method: Chromium Single cell 3′ Reagent v3 kits
    platform="10x 3' v3",
    reference_genome="ensembl.v93",
    matrix_type="raw counts",
    hospital_location="Union Hospital of Tongji Medical College, Huazhong University of Science and Technology, Wuhan, Hubei",
    country="China",
    NCBI_BioProject_accession="PRJNA900354",
)

# %%
adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]
adata.obs["SRA_sample_accession"] = adata.obs["big_key"]

adata.obs["sex"] = adata.obs["Gender"]
adata.obs["age"] = adata.obs["Age at diagnosis"].fillna(np.nan).astype("Int64")
adata.obs["treatment_status_before_resection"] = adata.obs["preoperative chemotherapy"].map({"No": "naive"})

# Paper results: we collected the tumor tissues of three CRC patients (WUH1, WUH2 and WUH3)
adata.obs["sample_id"] = adata.obs["patient_id"]

# %%
mapping = {
    "Ascending colon": "ascending colon",
    "rectum": "rectum",
    "descending colon": "descending colon",
}
adata.obs["anatomic_location"] = adata.obs["Site of tumor"].map(mapping)

mapping = {
    "ascending colon": "proximal colon",
    "rectum": "distal colon",
    "descending colon": "distal colon",
}
adata.obs["anatomic_region"] = adata.obs["anatomic_location"].map(mapping)

mapping = {
    "moderately differentiated adenocarcinoma": "adenocarcinoma",
    "moderately poorly differentiated adenocarcinoma": "adenocarcinoma",
}
adata.obs["histological_type"] = adata.obs["Pathology"].map(mapping)

mapping = {
    "moderately differentiated adenocarcinoma": "G2",
    "moderately poorly differentiated adenocarcinoma": "G2-G3",
}
adata.obs["tumor_grade"] = adata.obs["Pathology"].map(mapping)

adata.obs["tumor_stage_TNM_T"] = "T" + adata.obs["T"].astype(str)
adata.obs["tumor_stage_TNM_N"] = "N" + adata.obs["N"].astype(str)
adata.obs["tumor_stage_TNM_M"] = "M" + adata.obs["M"].astype(str)
adata.obs["tumor_stage_TNM"] = ah.pp.tnm_tumor_stage(adata.obs, "patient_id")

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
# Save unfiltered raw counts
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
