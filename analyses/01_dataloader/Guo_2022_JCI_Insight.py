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
# # Dataloader: Guo_2022_JCI_Insight

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
cpus = nxfvars.get("cpus", 6)
d10_1172_jci_insight_152616 = f"{databases_path}/d10_1172_jci_insight_152616"

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
# Get available metadata
SraRunTable = pd.read_csv(
    f"{d10_1172_jci_insight_152616}/fetchngs/PRJNA779978/samplesheet/samplesheet.csv"
)

geofetch = pd.read_csv(
    f"{d10_1172_jci_insight_152616}/GSE188711/GSE188711_PEP/GSE188711_PEP_raw.csv"
).dropna(axis=1, how="all")

meta = pd.merge(
    SraRunTable.groupby("sample").first(),
    geofetch,
    how="left",
    left_on=["sample"],
    right_on=["srx"],
    validate="m:1",
).rename(columns={"srx": "SRA_sample_accession"})

# Match column sample names from clinical meta
meta["sample_title_x"] = meta["sample_title_x"].str.replace(
    r"(\d+)", r"Patient \1", regex=True
)

# %%
# import tabula
# tables = tabula.read_pdf(f"{d10_1172_jci_insight_152616}/metadata/page_5.pdf", pages=1)
# clinical_info_df = tables[0]
# Rename columns and drop the first row
# clinical_info_df.rename(
#    columns={
#        "Tumor": "Tumor Location",
#        "Tumor.1": "Tumor.1 Size(cm)",
#        "AJCC": "AJCC Stage",
#        "LN": "LN metastasis",
#        "Distant": "Distant metastasis",
#        "MSI": "MSI status",
#        "Metabolic": "Metabolic disorders",
#    },
#    inplace=True,
# )
# clinical_info_df = clinical_info_df.iloc[1:, :]
# clinical_info_df.to_csv(f"{d10_1172_jci_insight_152616}/metadata/clinical_info.csv", index=False)

clinical_info = pd.read_csv(f"{d10_1172_jci_insight_152616}/metadata/clinical_info.csv")

# %%
meta = pd.merge(
    meta,
    clinical_info,
    how="left",
    left_on=["sample_title_x"],
    right_on=["Patient No."],
    validate="m:1",
).rename(columns={"sample_title_x": "sample_title"})


# %%
def load_counts(sample_meta, databases_path, mat_type="filtered"):

    file_path = f"{databases_path}/10_nfcore_scrnaseq/cellranger/count/{sample_meta['SRA_sample_accession']}/outs/{mat_type}_feature_bc_matrix.h5"
    adata = sc.read_10x_h5(file_path)
    adata.obs = adata.obs.assign(**sample_meta)
    adata.var_names_make_unique()
    adata.X = csr_matrix(adata.X.astype(int))
    adata.obs_names = (
        adata.obs["SRA_sample_accession"] + "_" + adata.obs_names.str.replace("-1", "")
    )
    assert adata.obs_names.is_unique
    return adata


# %%
adatas = process_map(
    load_counts,
    [r for _, r in meta.iterrows()],
    itertools.repeat(d10_1172_jci_insight_152616),
    itertools.repeat("raw"),
    max_workers=cpus,
)

adata = anndata.concat(adatas, join="outer")

# %%
# Get original sample/patient meta and remove cell level meta
sample_meta = (
    adata.obs[
        adata.obs.columns[
            adata.obs.nunique() <= adata.obs["SRA_sample_accession"].nunique()
        ]
    ]
    .groupby("SRA_sample_accession")
    .first()
    .reset_index()
)

# %%
sample_meta.to_csv(
    f"{artifact_dir}/Guo_2022-original_sample_meta.csv",
    index=False,
)

# %% [markdown]
# ## 2. Gene annotations

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
# ## 3. Compile original meta from study supplements

# %%
adata.obs = adata.obs.assign(
    study_id="Guo_2022_JCI_Insight",
    study_doi="10.1172/jci.insight.152616",
    study_pmid="34793335",
    tissue_processing_lab="Jingxin Li lab",
    dataset="Guo_2022",
    medical_condition="colorectal cancer",
    # Paper methods: Six patients who were pathologically diagnosed with colorectal adenocarcinoma were enrolled in this study
    histological_type="adenocarcinoma",
    sample_type="tumor",
    sample_tissue="colon",
    # Paper methods: None of the patients was treated with chemotherapy, radiation or any other antitumor medicines prior to tumor resection
    treatment_status_before_resection="naive",
    tissue_cell_state="fresh",
    platform="10x 3' v3",
    matrix_type="raw counts",
    reference_genome="gencode.v44",
    cellranger_version="7.1.0",
    hospital_location="Shandong University Qilu Hospital, Jinan",
    country="China",
    enrichment_cell_types="naive",
)

adata.obs["NCBI_BioProject_accession"] = adata.obs["study_accession"]
adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]


adata.obs["sample_id"] = adata.obs["SRA_sample_accession"]
# define the dictionary mapping
mapping = {
    "Left-sided CRC Patient 1": "Left_CRC1",
    "Left-sided CRC Patient 2": "Left_CRC2",
    "Left-sided CRC Patient 3": "Left_CRC3",
    "Right-sided CRC Patient 1": "Right_CRC1",
    "Right-sided CRC Patient 2": "Right_CRC2",
    "Right-sided CRC Patient 3": "Right_CRC3",
}
adata.obs["patient_id"] = adata.obs["sample_title"].map(mapping)

# %%
adata.obs["age"] = adata.obs["Age"].fillna(np.nan).astype("Int64")

# define the dictionary mapping
mapping = {
    "Female": "female",
    "Male": "male",
}
adata.obs["sex"] = adata.obs["Sex"].map(mapping)

# define the dictionary mapping
mapping = {
    "Sigmoid": "sigmoid colon",
    "Ascending": "ascending colon",
}
adata.obs["anatomic_location"] = adata.obs["Tumor Location"].map(mapping)

# define the dictionary mapping
mapping = {
    "Sigmoid": "distal colon",
    "Ascending": "proximal colon",
}
adata.obs["anatomic_region"] = adata.obs["Tumor Location"].map(mapping)

adata.obs["microsatellite_status"] = adata.obs["MSI status"]


# define the dictionary mapping
mapping = {
    "No": "N0",
}
adata.obs["tumor_stage_TNM_N"] = adata.obs["LN metastasis"].map(mapping)

# define the dictionary mapping
mapping = {
    "No": "M0",
}
adata.obs["tumor_stage_TNM_M"] = adata.obs["Distant metastasis"].map(mapping)

adata.obs["tumor_stage_TNM"] = adata.obs["AJCC Stage"]


adata.obs["tumor_dimensions"] = adata.obs["Tumor.1 Size(cm)"].str.replace(" ", "")

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]]

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["SRA_sample_accession"]
    + "-"
    + adata.obs_names.str.split("_").str[1]
)

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
adata


# %% [markdown]
# ## Load GEO data
# For some reason the umap from the fastqs looks weird. Will keep only barcodes from study for now ...

# %%
def _load(stem, accession):
    mat = fmm.mmread(
        f"{d10_1172_jci_insight_152616}/GSE188711/GSE188711_RAW/{accession}_matrix_{stem}.mtx.gz"
    )

    barcodes = pd.read_csv(
        f"{d10_1172_jci_insight_152616}/GSE188711/GSE188711_RAW/{accession}_barcodes_{stem}.tsv.gz",
        header=None,
    )
    features = pd.read_csv(
        f"{d10_1172_jci_insight_152616}/GSE188711/GSE188711_RAW/{accession}_features_{stem}.tsv.gz",
        header=None,
        delimiter="\t",
    )

    features = features.set_index(0)
    barcodes = barcodes.set_index(0)
    features.index.name = None
    barcodes.index.name = None

    adata = sc.AnnData(X=mat.T)
    adata.X = csr_matrix(adata.X)
    adata.var = features
    adata.obs = barcodes
    adata.obs["file name"] = stem
    return adata


# %%
output_list = []
file_names = ["WGC", "JCA", "LS-CRC3", "RS-CRC1", "R_CRC3", "R_CRC4"]
accession = [
    "GSM5688706",
    "GSM5688707",
    "GSM5688708",
    "GSM5688709",
    "GSM5688710",
    "GSM5688711",
]
for stem, accession in zip(file_names, accession):
    adata_geo = _load(stem, accession)
    output_list.append(adata_geo)

# %%
adata_geo = anndata.concat(output_list, index_unique="_", join="outer", fill_value=0)
adata_geo.obs_names = adata_geo.obs_names.str.split("-").str[0]
adata_geo.obs_names_make_unique()

# %%
# define the dictionary mapping
mapping = {
    "WGC": "Left-sided CRC Patient 1",
    "JCA": "Left-sided CRC Patient 2",
    "LS-CRC3": "Left-sided CRC Patient 3",
    "RS-CRC1": "Right-sided CRC Patient 1",
    "R_CRC3": "Right-sided CRC Patient 2",
    "R_CRC4": "Right-sided CRC Patient 3",
}
adata_geo.obs["Patient No."] = adata_geo.obs["file name"].map(mapping)

# %%
# Merge all available metadata
adata_geo.obs["obs_names"] = adata_geo.obs_names
adata_geo.obs = pd.merge(
    adata_geo.obs,
    meta,
    how="left",
    on=["Patient No."],
    validate="m:1",
).set_index("obs_names")
adata_geo.obs_names.name = None

adata_geo.obs_names = (
    "Guo_2022"
    + "-"
    + adata_geo.obs["SRA_sample_accession"]
    + "-"
    + adata_geo.obs_names.str.replace("-1", "")
)

# %%
# Keep only barcodes from uploaded GEO Samples
# adata = adata[adata.obs_names.isin(adata_geo.obs_names)].copy()

# Better to filter after scAR denoising
# This dataset had a weird umap embedding after scVI integration
adata.obs["obs_names_geo"] = adata.obs_names.isin(adata_geo.obs_names)

# %%
adata_geo

# %%
adata_geo.obs["SRA_sample_accession"].value_counts()

# %% [markdown]
# ## 4. Save adata by dataset

# %%
# Save unfiltered raw counts
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
