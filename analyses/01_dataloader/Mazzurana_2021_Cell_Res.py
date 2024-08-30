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
# # Dataloader: Mazzurana_2021_Cell_Res

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
cpus = nxfvars.get("cpus", 2)
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")
d10_1038_s41422_020_00445_x = f"{databases_path}/d10_1038_s41422_020_00445_x"
gtf = f"{annotation}/ensembl.v92_gene_annotation_table.csv"

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
sample_meta = pd.read_csv(
    f"{d10_1038_s41422_020_00445_x}/GSE150050/GSE150050_RAW/GSE150050_metadata.csv",
)
sample_meta = sample_meta.loc[sample_meta["Tissue"] == "COLON"].reset_index(drop=True)
sample_meta = sample_meta[["Dataset", "no_cells", "Sex"]]

# %%
def _load_mtx(meta):
    adata = sc.AnnData(
        X=fmm.mmread(
            f"{d10_1038_s41422_020_00445_x}/GSE150050/GSE150050_RAW/{meta['Dataset']}/matrix.mtx"
        ).T,
        obs=pd.read_csv(
            f"{d10_1038_s41422_020_00445_x}/GSE150050/GSE150050_RAW/{meta['Dataset']}/barcodes.tsv",
            index_col=0,
            header=None,
        ),
        var=pd.read_csv(
            f"{d10_1038_s41422_020_00445_x}/GSE150050/GSE150050_RAW/{meta['Dataset']}/features.tsv",
            index_col=0,
            header=None,
        ),
    )
    adata.X = csr_matrix(adata.X)
    adata.var_names.name = None
    adata.obs_names.name = None
    adata.obs = adata.obs.assign(**meta)

    cell_meta = pd.read_csv(
        f"{d10_1038_s41422_020_00445_x}/GSE150050/GSE150050_RAW/{meta['Dataset']}/metadata_{meta['Dataset']}.csv",
        index_col=0,
        sep=";",
    ).reset_index(names=["obs_names"])

    adata.obs = adata.obs.reset_index(names="obs_names")
    adata.obs = pd.merge(
        adata.obs,
        cell_meta,
        how="left",
        on=["obs_names"],
        validate="m:1",
    ).set_index("obs_names")
    adata.obs_names.name = None
    return adata

# %%
adatas = process_map(
    _load_mtx, [r for _, r in sample_meta.iterrows()], max_workers=cpus
)
adata = anndata.concat(adatas, index_unique=".", join="outer", fill_value=0)

# %%
geofetch = pd.read_csv(
    f"{d10_1038_s41422_020_00445_x}/GSE150050/GSE150050_PEP/GSE150050_PEP_raw.csv"
).dropna(axis=1, how="all")

# %%
adata.obs = adata.obs.reset_index(names="obs_names")
adata.obs = pd.merge(
    adata.obs,
    geofetch,
    how="left",
    left_on=["Dataset"],
    right_on=["sample_title"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %% [markdown]
# ## 2. Gene annotations

# %%
# There seem to be citeseq data + also duplicate ENSG ids with and without version number
#adata[:, ~adata.var_names.str.contains('ENSG', regex=True)].var
#adata[:, adata.var_names.str.contains('\.', regex=True)].var_names
adata.var["ensembl"] = adata.var_names
adata.var["ensembl"] = adata.var["ensembl"].apply(ah.pp.remove_gene_version)
adata.var_names = adata.var["ensembl"]

# %%
adata.var[adata.var['ensembl'].str.contains('\+')]

# %%
# ENSG00000263436: MIR3158-1
# ENSG00000283558: MIR3158-2
# ENSG00000207688: MIR548AA2
# ENSG00000263690: MIR548D2
# ...
# Not sure what happened here -> will just drop these!
adata = adata[:, ~adata.var['ensembl'].str.contains('\+')]

# %%
duplicated_ids = adata.var_names[adata.var_names.duplicated()].unique()
adata = ah.pp.aggregate_duplicate_gene_ids(adata, duplicated_ids)

# %%
# Not sure what this is either -> will just drop!
adata[:, ~adata.var_names.str.contains('ENSG', regex=True)].var

# %%
adata = adata[:, adata.var_names.str.contains('ENSG', regex=True)].copy()
adata.var["ensembl"] = adata.var_names

# %%
gtf = pd.read_csv(gtf)
gtf["ensembl"] = gtf["gene_id"].apply(ah.pp.remove_gene_version)
gene_ids = gtf.set_index("ensembl")["gene_name"].to_dict()

# %%
for column in [col for col in gtf.columns if col != "ensembl"]:
    adata.var[column] = adata.var_names.map(gtf.set_index("ensembl")[column])

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="Mazzurana_2021_Cell_Res",
    study_doi="10.1038/s41422-020-00445-x",
    study_pmid="33420427",
    tissue_processing_lab="Jenny Mjösberg lab",
    dataset="Mazzurana_2021_CD127Pos",
    # Paper methods: Biopsies from the ascending colon were obtained from three tumor-screening patients (age 57–71 year old, two males, one female)
    # with known genetic predisposition for colon cancer (hereditary nonpolyposis colorectal cancer (HPNCC) or Lynch Syndrome) but without any polyps
    # or tumors at the time of sampling.
    medical_condition="healthy",
    anatomic_location="ascending colon",
    anatomic_region="proximal colon",
    age="57–71 years",
    matrix_type="raw counts",
    platform="Smart-seq2",
    # Paper methods: Counts per gene were calculated for each transcript in Ensembl release 92
    reference_genome="ensembl.v92",
    NCBI_BioProject_accession="PRJNA630996",
    sample_tissue="colon",
    sample_type="normal",
    treatment_status_before_resection="naive",
    # Paper methods: Samples were processed within 1 h of biopsy retrieval
    tissue_cell_state="fresh",
    # Paper: flow cytometric sorting of human CD127+ ILCs
    enrichment_cell_types="CD127+",
    hospital_location="Karolinska University Hospital or GHP Stockholm Gastro Center",
    country="Sweden",
)

# %%
adata.obs["patient_id"] = adata.obs["Donor"]
adata.obs["sample_id"] = adata.obs["Dataset"]
adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]
adata.obs["cell_type_study"] = adata.obs["Celltype"]

# define the dictionary mapping
mapping = {
    "F": "female",
    "M": "male",
}
adata.obs["sex"] = adata.obs["Sex"].map(mapping)

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["GEO_sample_accession"]
    + "-"
    + adata.obs["Well"]
)

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
# Include also FACS data
ref_meta_cols += [
    "FSC_A",
    "FSC_W",
    "FSC_H",
    "SSC_A",
    "SSC_W",
    "SSC_H",
    "LIN_B530",
    "CD45_V525",
    "CD161_V610",
    "CD56_V710",
    "CD3_V780",
    "CD4_R780",
    "CD103_YG582",
    "CRTH2_YG610",
    "NKP44_YG670",
    "CD117_YG710",
    "CD127_YG780",
]

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
adata.obs["patient_id"].value_counts()

# %%
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
