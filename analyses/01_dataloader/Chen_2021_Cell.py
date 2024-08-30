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
# # Dataloader: Chen_2021_Cell

# %%
import os
import re
from pathlib import Path

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
artifact_dir = nxfvars.get("artifact_dir", "/data/scratch/marteau/downloads/crc-atlas/")
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")
cpus = nxfvars.get("cpus", 8)

# %%
d10_1016_j_cell_2021_11_031 = f"{databases_path}/d10_1016_j_cell_2021_11_031"
gtf = f"{annotation}/gencode.v25_gene_annotation_table.csv"
synapse = f"{d10_1016_j_cell_2021_11_031}/fetchngs/synapse"

# %%
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

# %%
# Read the YAML file and load it into a dictionary
with open(reference_meta, "r") as f:
    ref_meta_dict = yaml.load(f, Loader=yaml.Loader)

# %% [markdown]
# ## 1. Load raw adata counts from Synapse and append HTAN metadata

# %%
# load manifest to get file names
meta = pd.read_csv(
    f"{synapse}/syn23520239/HTAN_Level_3/v2/manifest_1704122426567156333.csv.gz"
)
meta = meta[
    ~meta["name"].isin(
        [
            "synapse_storage_manifest_scrna-seqlevel3.csv",  # Remove manifest
            "7133-YX-12_immune.csv",  # File is empty!
        ]
    )
]
meta = meta.rename(columns={"ID": "SynapseID", "name": "FileName"})
meta = meta.dropna(axis=1, how="all").drop("path", axis=1)


# %%
# v1 only
# For some reason downloaded file names do not match manifest names -> upper()
# meta["file"] = (
#    meta["name"]
#    .str.split(".")
#    .str[0]
#    .apply(
#        lambda x: str(x).upper()
#        if not any(pattern in str(x) for pattern in ["_epi", "_immune", "Patient"])
#        else x
#    )
# )

# %%
# When I use scanpy sc.read_csv() for some files the gene index seems to get shifted by one !!
def _load_csv(meta):
    counts = pd.read_csv(
        f"{synapse}/syn23520239/HTAN_Level_3/v2/{meta['FileName']}.gz", index_col=0
    )
    # Immune cell fraction obs_names are int in v1 only!
    # counts.index = counts.index.map(
    #    lambda x: "obs_" + str(x) if isinstance(x, int) else x
    # )
    counts.index.name = None
    counts = counts.astype(int)
    adata = sc.AnnData(counts)
    adata.X = csr_matrix(adata.X.astype(int))
    adata.obs = adata.obs.assign(**meta)
    adata.obs["original_obs_names"] = adata.obs_names.str.replace("obs_", "")
    adata.obs_names_make_unique()
    return adata


# %%
adatas = process_map(_load_csv, [r for _, r in meta.iterrows()], max_workers=cpus)
adata = anndata.concat(adatas, index_unique="-", join="outer", fill_value=0)
adata.obs_names_make_unique()

# %% [markdown] tags=[]
# ## 2. Gene annotations

# %%
# gtf_to_table.sh does not work for this old annotation, using this instead!
# #! zcat gencode.v25.primary_assembly.annotation.gtf.gz | awk 'BEGIN{FS="\t";OFS=","}$3=="gene"{split($9,a,";");split(a[1],gene_id,"\"");split(a[4],gene_name,"\""); print gene_id[2],gene_name[2]}' | sed '1i\Geneid,GeneSymbol' > gencode.v25_gene_annotation_table.csv

# %%
# load gtf for gene mapping
gtf = pd.read_csv(gtf)
gtf = ah.pp.append_duplicate_suffix(df=gtf, column="GeneSymbol", sep="-")
gene_ids = gtf.set_index("GeneSymbol")["Geneid"].to_dict()

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
# Sum up duplicate gene ids
duplicated_ids = adata.var_names[adata.var_names.duplicated()].unique()
adata = ah.pp.aggregate_duplicate_gene_ids(adata, duplicated_ids)

# %%
# Reorder by gtf (i.e. chromosome position)
gene_index = gtf[gtf["Geneid"].isin(adata.var["ensembl"])]["Geneid"].apply(
    ah.pp.remove_gene_version
)
adata = adata[:, gene_index]

# %% [markdown]
# ## 3. Compile additional sample/patient metadata

# %%
HTAN_sample_meta = pd.read_csv(
    f"{d10_1016_j_cell_2021_11_031}/metadata/syn23520239_sample_meta.csv"
)

values_to_replace = ["Not Applicable", "Not Reported", "unknown", "<NA>", np.nan]
HTAN_sample_meta.replace({value: pd.NA for value in values_to_replace}, inplace=True)

# Drop columns where all values are NaN
HTAN_sample_meta.dropna(axis=1, how="all", inplace=True)
HTAN_sample_meta.drop("HTANDataFileID", axis=1, inplace=True)
HTAN_sample_meta["FileName"] = HTAN_sample_meta["FileName"].str.lower()

# %%
adata.obs["FileName"] = adata.obs["FileName"].str.lower()

adata.obs = adata.obs.reset_index(names="obs_names")
adata.obs = pd.merge(
    adata.obs,
    HTAN_sample_meta,
    how="left",
    on=["FileName"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
# Find duplicate columns and remove + rename without extension!
sorted(set([col[:-2] for col in adata.obs.columns if col.endswith(("_x", "_y"))]))

adata.obs.drop(
    columns=[
        "CellMedianNumberGenes_y",
        "CellMedianNumberReads_y",
        "CellTotal_y",
        "HTANParentDataFileID_y",
    ],
    inplace=True,
)
adata.obs.columns = [
    col.replace("_x", "").replace("_y", "") for col in adata.obs.columns
]

# %% [markdown]
# ### 3a. Polyp metadata - Discovery and Validation cohort + CRC metadata from supplemetal tables

# %%
# Get Polyp sample metadata
file_path = f"{d10_1016_j_cell_2021_11_031}/metadata/1-s2.0-S0092867421013817-mmc1.xlsx"
data_discovery = pd.read_excel(file_path, sheet_name="Discovery Meta (NL,Precancer)")
data_validation = pd.read_excel(file_path, sheet_name="Validation Meta (NL, Precancer)")

# Add a new column with the source information to each DataFrame
data_discovery["Source"] = "discovery"
data_validation["Source"] = "validation"

sample_meta_mmc1 = pd.concat([data_discovery, data_validation], ignore_index=True)


# Get CRC sample metadata
sample_meta_mmc7 = pd.read_csv(
    f"{d10_1016_j_cell_2021_11_031}/metadata/1-s2.0-S0092867421013817-mmc7.txt",
    delimiter="\t",
)

# %%
adata.obs = adata.obs.reset_index(names="obs_names")
adata.obs = pd.merge(
    adata.obs,
    sample_meta_mmc1,
    how="left",
    on=["SPECID"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
adata.obs = adata.obs.reset_index(names="obs_names")
adata.obs = pd.merge(
    adata.obs,
    sample_meta_mmc7,
    how="left",
    left_on=["SPECID"],
    right_on=["ID"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
# convert to type str
columns_to_convert = ["PCAD", "PCNORMAL", "PCVILLOUS", "PDIM", "SIZEINVIVO"]
adata.obs[columns_to_convert] = adata.obs[columns_to_convert].astype(str)

# %%
# Get original sample/patient meta and remove cell level meta
sample_meta = (
    adata.obs[adata.obs.columns[adata.obs.nunique() <= adata.obs["FileName"].nunique()]]
    .groupby("FileName")
    .first()
    .reset_index()
)

# %%
sample_meta.to_csv(f"{artifact_dir}/Chen_2021_original_sample_meta.csv", index=False)

# %% [markdown]
# ## 4. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="Chen_2021_Cell",
    study_doi="10.1016/j.cell.2021.11.031",
    study_pmid="34910928",
    tissue_processing_lab="Ken S. Lau lab",
    dataset="Chen_2021",
    medical_condition="colorectal cancer",
    enrichment_cell_types="naive",
    Synapse_project_accession="syn23520239",
    matrix_type="raw counts",
    platform="TruDrop",
    reference_genome="gencode.v25",
    sample_tissue="colon",
    hospital_location="Vanderbilt University Medical Center, Nashville",
    country="US",
    # Paper methods: All polyps which were placed in RPMI were immediately transported to the research lab for use in scRNA-seq analysis.
    tissue_cell_state="fresh",
    # TableS7: column "Treatment History"
    treatment_status_before_resection="naive",
)

# %%
adata.obs["synapse_sample_accession"] = adata.obs["SynapseID"]
adata.obs["sample_id"] = adata.obs["HTAN Biospecimen ID"]
adata.obs["patient_id"] = adata.obs["HTAN Participant ID"]

adata.obs["microsatellite_status"] = adata.obs["Microsatellite status"]
adata.obs["sex"] = adata.obs["Gender"]

# year of recruitment is between [2019, 2020, 2021] from 'Year Of Birth' + 'DOB.age'
# Will use 2020 to determine approx age from Year Of Birth for missing age info
mapping = (
    adata.obs.dropna(subset=["patient_id", "DOB.age"])
    .groupby("patient_id")
    .first()["DOB.age"]
    .to_dict()
)
adata.obs["age"] = (
    adata.obs["patient_id"]
    .map(mapping)
    .fillna(adata.obs["Age"])
    .fillna(2020 - adata.obs["Year Of Birth"])
    .astype("Int64")
)

adata.obs["dataset"] = (
    adata.obs["Atlas Name"]
    .str.replace("HTAN Vanderbilt", "VUMC_HTAN_")
    .str.cat(adata.obs["Source"].astype(str), na_rep="", sep="_")
    .replace(
        "HTAN_Vanderbilt_nan", "HTAN_Vanderbilt_cohort3"
    )  # cohort3 name from HTAN "VUMC_HTAN_NEWCo_EPI_V2.h5ad"
)

# %%
adata.obs["dataset"] = (
    adata.obs["Atlas Name"]
    .str.replace("HTAN Vanderbilt", "VUMC_HTAN")
    .str.cat(adata.obs["Source"].astype(str), na_rep="", sep="_")
    .replace(
        "VUMC_HTAN_nan", "VUMC_HTAN_cohort3"
    )  # cohort3 name from HTAN "VUMC_HTAN_NEWCo_EPI_V2.h5ad"
)

# %%
mapping = {
    "Descending colon": "descending colon",
    "Ascending colon": "ascending colon",
    "Rectum": "rectum",
    "Transverse colon": "transverse colon",
    "Cecum": "cecum",
    "Sigmoid colon": "sigmoid colon",
    "Hepatic flexure": "hepatic flexure",
    "nan": np.nan,
}
adata.obs["anatomic_location"] = (
    adata.obs["SEGMENT"].map(mapping).fillna(adata.obs["Tumor location"].map(mapping))
)

mapping = {
    "descending colon": "distal colon",
    "ascending colon": "proximal colon",
    "cecum": "proximal colon",
    "transverse colon": "proximal colon",
    "rectum": "distal colon",
    "sigmoid colon": "distal colon",
    "hepatic flexure": "proximal colon",
    "nan": np.nan,
}
adata.obs["anatomic_region"] = adata.obs["anatomic_location"].map(mapping)

mapping = {
    "Carcinoma / Adenocarcinoma / Mucinous": "mucinous adenocarcinoma",
    "Carcinoma / Adenocarcinoma": "adenocarcinoma",
    "nan": np.nan,
}
adata.obs["histological_type"] = adata.obs["Type"].map(mapping)

mapping = {
    "G2 Moderately differentiated": "G2",
    "G3 Poorly differentiated": "G3",
    "nan": np.nan,
}
adata.obs["tumor_grade"] = adata.obs["Grade"].map(mapping)

adata.obs["tumor_stage_TNM"] = adata.obs["Staging"]

mapping = {
    "BRAF c.1799T>A": "mut",
    "No BRAF test": np.nan,
    "nan": np.nan,
}
adata.obs["BRAF_status_driver_mut"] = adata.obs["Mutations"].map(mapping)

mapping = {
    "Lymph nodes": "N1",
    "nan": np.nan,
}
adata.obs["tumor_stage_TNM_N"] = adata.obs["Metastasis"].map(mapping)

adata.obs["tumor_size"] = adata.obs["Biospecimen Dimension 1"]
mapping = {
    "6.2 cm": 6.2,
    "3.6 cm": 3.6,
    "5.8 cm": 5.8,
    "nan": np.nan,
}
adata.obs["tumor_size"] = (
    adata.obs["tumor_size"]
    .fillna(adata.obs["Tumor greatest dimension"].map(mapping))
    .astype(float)
)

# Paper: Polyps were histologically categorized by two pathologists into two subtypes: ADs consisting of tubular ADs (TAs) and tubulovillous ADs (TVAs),
# or serrated polyps (SERs) consisting of hyperplastic polyps (HPs) and SSLs (Figure 1B). While standard histological features were observed for polyps,
# HPs were further subdivided into goblet cell-rich HPs (GCHPs) and microvesicular HPs (MVHPs), with MVHPs appearing more advanced and may progress to SSLs
mapping = {
    "HP": "hyperplastic polyp",
    "TA": "tubular adenoma",
    "SSL": "sessile serrated lesion",
    "NL": "normal",
    "TV": "tubulovillous adenoma",
    "TVA": "tubulovillous adenoma",
    "UNK": "unknown",
    "nan": np.nan,
}
adata.obs["histological_type"] = adata.obs["histological_type"].fillna(
    adata.obs["SAMPLE TYPE"].map(mapping)
)

# https://kankerregister.org/media/docs/downloads/pourpathologistes/SNOMED3.5VF_codebook.pdf
mapping = {
    "M82110": "tubular adenoma",
    "M82630": "tubulovillous adenoma",
    "M82130": "sessile serrated lesion",  # M82130: serrated adenoma
    "0": np.nan,  # should be normal
    "Unknown": np.nan,
    "Fresh": np.nan,
}
adata.obs["histological_type"] = adata.obs["histological_type"].fillna(
    adata.obs["Histologic Morphology Code"].map(mapping)
)

# %%
mapping = {
    "Other": "other",
    "White": "white",
    "Black or African American": "black or african american",
    "Unknown": np.nan,
    "Not stated": np.nan,
    "nan": np.nan,
}
adata.obs["ethnicity"] = adata.obs["RACE"].astype(str).map(mapping)

mapping = {
    "Other": "other",
    "white": "white",
    "black or african american": "black or african american",
    "<NA>": np.nan,
}

adata.obs["ethnicity"] = adata.obs["ethnicity"].fillna(
    adata.obs["Race"].astype(str).map(mapping)
)

mapping = {
    "not hispanic or latino": "not hispanic or latino",
    "hispanic or latino": "hispanic",
    "<NA>": np.nan,
}

adata.obs["ethnicity"] = adata.obs["ethnicity"].fillna(
    adata.obs["Ethnicity_x"].astype(str).map(mapping)
)

mapping = {
    "White": "white",
    "Black": "black or african american",
    "nan": np.nan,
}

adata.obs["ethnicity"] = adata.obs["ethnicity"].fillna(
    adata.obs["Ethnicity_y"].astype(str).map(mapping)
)

# %%
mapping = {
    "Premalignant": "polyp",
    "Atypia - hyperplasia": "polyp",
    "Primary": "tumor",
    "Normal": "normal",
    "Not Otherwise Specified": "polyp",  # can be inferred from other samples from patient -> 2X normal! only leaves polyp!
    "<NA>": np.nan,
}
adata.obs["sample_type"] = adata.obs["Tumor Tissue Type"].astype(str).map(mapping)

mapping = {
    "NL": "normal",
}
adata.obs["sample_type"] = adata.obs["sample_type"].fillna(
    adata.obs["SAMPLE TYPE"].astype(str).map(mapping)
)

# Update datasets, put CRC samples seperately -> "VUMC_ABNORMALS"
adata.obs.loc[adata.obs["sample_type"] == "tumor", "dataset"] = "VUMC_HTAN_CRC"

# %% [markdown]
# ### Get metadata for 4 missing CRC patients from cellxgene as well as cell type mapping

# %%
# Synapse HTAN Level 4
DATASETS = {
    "VUMC_ABNORMALS_EPI_V2": f"{synapse}/syn25459679/HTAN_Level_4/VUMC_ABNORMALS_EPI_V2.h5ad",
    "VUMC_HTAN_NEWCo_EPI_V2": f"{synapse}/syn25459679/HTAN_Level_4/VUMC_HTAN_NEWCo_EPI_V2.h5ad",
    "VUMC_HTAN_VAL_EPI_V2": f"{synapse}/syn25459679/HTAN_Level_4/VUMC_HTAN_VAL_EPI_V2.h5ad",
    "VUMC_HTAN_All_NonEPI_V2": f"{synapse}/syn25459679/HTAN_Level_4/VUMC_HTAN_All_NonEPI_V2.h5ad",
    "VUMC_HTAN_DIS_EPI_V2": f"{synapse}/syn25459679/HTAN_Level_4/VUMC_HTAN_DIS_EPI_V2.h5ad",
    "VUMC_HTAN_VAL_DIS_NONEPI_V2": f"{synapse}/syn25459679/HTAN_Level_4/VUMC_HTAN_VAL_DIS_NONEPI_V2.h5ad",
}
#datasets = {dataset_id: sc.read_h5ad(path) for dataset_id, path in DATASETS.items()}

# %%
# if using HTAN adata need to take care of sample names -> some have additional underscores ...
# datasets["VUMC_HTAN_NEWCo_EPI_V2"].obs["SPECID.htan"] = (
#    datasets["VUMC_HTAN_NEWCo_EPI_V2"]
#    .obs["SPECID.htan"]
#    .str.rsplit("_", n=1)
#    .str.join("")
# )

# %%
DATASETS = {
    "VUMC_HTAN_DIS_EPI_cellxgene": f"{synapse}/syn25459679/HTAN_Level_4/VUMC_HTAN_DIS_EPI_cellxgene.h5ad",
    "VUMC_VAL_DIS_NOEPI_cellxgene": f"{synapse}/syn25459679/HTAN_Level_4/VUMC_VAL_DIS_NOEPI_cellxgene.h5ad",
    "VUMC_HTAN_VAL_EPLI_cellxgene": f"{synapse}/syn25459679/HTAN_Level_4/VUMC_HTAN_VAL_EPLI_cellxgene.h5ad",
}
cellxgene = {dataset_id: sc.read_h5ad(path) for dataset_id, path in DATASETS.items()}
adata_cellxgene = anndata.concat(cellxgene, index_unique="_", join="outer", fill_value=0)

# %%
adata_cellxgene.obs["cell_type_ids"] = (
    adata_cellxgene.obs["HTAN Specimen ID"].astype(str)
    + "_"
    + adata_cellxgene.obs_names.str.split("-").str[0].astype(str)
)
# get dict
cell_type_mapping = (
    adata_cellxgene.obs[["cell_type_ids", "cell_type"]]
    .set_index("cell_type_ids")["cell_type"]
    .drop_duplicates()
    .to_dict()
)

# %%
adata.obs["cell_type_ids"] = (
    adata.obs["sample_id"] + "_" + adata.obs["original_obs_names"]
)

adata.obs["cell_type_study"] = adata.obs["cell_type_ids"].map(cell_type_mapping)
adata.obs.loc[adata.obs.duplicated("cell_type_ids"), "cell_type_study"] = np.nan

# %%
# Paper: Figure 2A (bottom) for cell type abbrev.
mapping = {
    "ABS": "absortive cell",
    "STM": "stem cell",
    "TAC": "transit amplifying cell",
    "SSC": "serrated specific cell",
    "ASC": "adenoma specific cell",
    "GOB": "goblet cell",
    "EE": "enteroendocrine cell",
    "TUF": "tuft cell",
    "CT": "crypt top colonocyte",
    "20": "nan",
}

# %%
tmp_df = (
    cellxgene["VUMC_VAL_DIS_NOEPI_cellxgene"][
        cellxgene["VUMC_VAL_DIS_NOEPI_cellxgene"]
        .obs["HTAN Specimen ID"]
        .isin(
            [
                "HTA11_99999971662_82457",
                "HTA11_99999973458_83798",
                "HTA11_99999973899_84307",
                "HTA11_99999974143_84620",
            ]
        )
    ]
    .obs[
        [
            "HTAN Specimen ID",
            "donor_id",
            "sex",
            "tissue",
            "self_reported_ethnicity",
            "development_stage",
            "Sample_Classification",
        ]
    ]
    .groupby("HTAN Specimen ID", observed=False)
    .first()
)

tmp_df

# %%
adata[
    adata.obs["sample_id"].isin(
        [
            "HTA11_99999971662_82457",
            "HTA11_99999973458_83798",
            "HTA11_99999973899_84307",
            "HTA11_99999974143_84620",
        ]
    )
].obs[
    [
        "patient_id",
        "microsatellite_status",
        "sex",
        "age",
        "anatomic_location",
        "anatomic_region",
        "ethnicity",
    ]
].groupby(
    "patient_id", observed=False
).first()

# %%
adata.obs["microsatellite_status"] = adata.obs["microsatellite_status"].fillna(
    adata.obs["sample_id"].map(tmp_df["Sample_Classification"].to_dict())
)

adata.obs["anatomic_location"] = adata.obs["anatomic_location"].fillna(
    adata.obs["sample_id"].map(tmp_df["tissue"].to_dict())
)

mapping = {
    "descending colon": "distal colon",
    "ascending colon": "proximal colon",
    "sigmoid colon": "distal colon",
}
adata.obs["anatomic_region"] = adata.obs["anatomic_region"].fillna(
    adata.obs["anatomic_location"].map(mapping)
)

# based on cellxgene "ethnicity_ontology_term_id": 'HANCESTRO:0005' and 'HANCESTRO:0568'
mapping = {
    "African American": "black or african american",
    "European": "white",
}
tmp_df["self_reported_ethnicity"] = tmp_df["self_reported_ethnicity"].map(mapping)

adata.obs["ethnicity"] = adata.obs["ethnicity"].fillna(
    adata.obs["sample_id"].map(tmp_df["self_reported_ethnicity"].to_dict())
)

# %%
adata[
    adata.obs["sample_id"].isin(
        [
            "HTA11_99999971662_82457",
            "HTA11_99999973458_83798",
            "HTA11_99999973899_84307",
            "HTA11_99999974143_84620",
        ]
    )
].obs[
    [
        "patient_id",
        "microsatellite_status",
        "sex",
        "age",
        "anatomic_location",
        "anatomic_region",
        "ethnicity",
    ]
].groupby(
    "patient_id", observed=False
).first()

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
ref_meta_cols += [
    "HPTYPE",
    "ADVANCED",
    "ATYPIA",
    "PCAD",
    "PCNORMAL",
    "PCVILLOUS",
    "PDIM",
    "PSHAPE",
    "SIZEINVIVO",
    "BRAF_status_driver_mut",
]

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]]

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["sample_id"]
    + "-"
    + adata.obs["original_obs_names"]
)
adata.obs_names_make_unique()

# %% [markdown]
# ### append polyp mutations

# %%
mutations = pd.read_csv(
    f"{d10_1016_j_cell_2021_11_031}/metadata/Figure1_Oncoplot_polyp_somatic_mutations.csv"
)
mutations = mutations.add_suffix("_status_driver_mut")

# %%
mutations = mutations.set_index("PatientID_status_driver_mut")
mutations.fillna("wt", inplace=True)
mutations = mutations.applymap(lambda x: "mut" if x != "wt" else x)
mutations = mutations.reset_index().rename(
    columns={"PatientID_status_driver_mut": "patient_id"}
)

# %%
# Mutations available for 3 patients
adata[adata.obs["patient_id"].isin(mutations["patient_id"].unique().tolist())].obs[
    "patient_id"
].unique()

# %%
adata.obs = adata.obs.reset_index(names="obs_names")
adata.obs = pd.merge(
    adata.obs,
    mutations,
    how="left",
    on=["patient_id"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
adata.obs['BRAF_status_driver_mut'] = adata.obs['BRAF_status_driver_mut_x'].fillna(adata.obs['BRAF_status_driver_mut_y'])

# %%
del adata.obs['BRAF_status_driver_mut_x']
del adata.obs['BRAF_status_driver_mut_y']

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
adata

# %%
# Set display options to show all rows and columns
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# %%
adata.obs["sample_id"].value_counts()

# %%
pd.reset_option("display.max_rows")
pd.reset_option("display.max_columns")

# %% [markdown]
# ### Save raw counts from fastq

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
