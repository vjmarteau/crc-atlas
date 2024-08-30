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
# # Dataloader: Zhang_2018_Nature

# %%
from pathlib import Path
import os

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
cpus = nxfvars.get("cpus", 2)
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")
gtf = f"{annotation}/ensembl.v94_gene_annotation_table.csv"
d10_1038_s41586_018_0694_x = f"{databases_path}/d10_1038_s41586_018_0694_x"

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
counts = pd.read_csv(
    Path(
        f"{d10_1038_s41586_018_0694_x}/GSE108989/GSE108989_CRC.TCell.S11138.count.txt.gz"
    ),
    delimiter="\t",
)

# %%
mat = counts.drop(columns=["symbol"]).T
mat.columns = mat.iloc[0]
mat = mat[1:]
mat = mat.rename_axis(None, axis=1)
mat = mat.astype(int)

# %%
adata = sc.AnnData(mat)
adata.X = csr_matrix(adata.X)

# %%
# map back gene symbols to .var
counts['geneID'] = counts['geneID'].astype(str)
gene_symbols = counts.set_index("geneID")["symbol"].to_dict()

adata.var["UCSC stable ID"] = adata.var_names.astype(str)

adata.var["symbol"] = (
    adata.var["UCSC stable ID"].map(gene_symbols).fillna(value=adata.var["UCSC stable ID"])
)

adata.var_names = adata.var["symbol"]
adata.var_names.name = None
adata.var_names_make_unique()

# %% [markdown]
# ## 2. Load UCSC ensemblToGeneName table for gene mapping
# Dowloaded from [UCSC hgTables](https://genome.ucsc.edu/cgi-bin/hgTables). See also: https://www.biostars.org/p/92939/ and http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/genes/.
# Roughly 2000 genes do not have an ensembl id. I don't think it is a good idea to map these using the gene symbols to gencode/ensembl gtf files - which is possible. This UCSC GRCh37/hg19 assembly is from 2009 and I don't think that the gene coordinates match any more. Will only use the official UCSC to ensembl mapping and drop the rest!

# %%
UCSC_ensemblToGeneName = pd.read_csv(f"{d10_1038_s41586_018_0694_x}/metadata/UCSC_hg19_ensemblToGeneName.txt", sep='\t')
UCSC_ensemblToGeneName=UCSC_ensemblToGeneName.drop_duplicates(subset=['hg19.ensGene.name2'])
gene_ids = UCSC_ensemblToGeneName.set_index("hg19.ensemblToGeneName.value")["hg19.ensGene.name2"].to_dict()

# %%
adata.var["ensembl"] = (
    adata.var["symbol"].map(gene_ids).fillna(value=adata.var["symbol"])
)

adata.var_names = adata.var["ensembl"]
adata.var_names.name = None

# %%
unmapped_genes = ah.pp.find_unmapped_genes(adata)
len(unmapped_genes)

# %% [markdown]
# ## 2. Compile sample/patient metadata from original study supplemetary tables

# %%
cell_meta = pd.read_csv(f"{d10_1038_s41586_018_0694_x}/GSE108989/cell_meta.csv")
cell_meta.rename(columns={'UniqueCell_ID ': 'UniqueCell_ID'}, inplace=True)

patient_meta = pd.read_csv(f"{d10_1038_s41586_018_0694_x}/metadata/Sup_Table1.csv")
patient_meta2 = pd.read_csv(f"{d10_1038_s41586_018_0694_x}/metadata/Sup_Table2.csv")

patient_meta = pd.merge(
    patient_meta,
    patient_meta2,
    how="left",
    on=["Patient_ID"],
    validate="m:1",
)

# %%
adata.obs["UniqueCell_ID"] = adata.obs_names

# Merge cell_meta
adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    cell_meta,
    how="left",
    on=["UniqueCell_ID"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# Get sample mappings from "GEO: Overall design"
mapping = {
    "PTC": "CD8+ T cells from peripheral blood",
    "NTC": "CD8+ T cells from adjacent normal colonrectal tissues",
    "TTC": "CD8+ T cells from tumor",
    "PTH": "CD3+, CD4+ and CD25- T cells from peripheral blood",
    "NTH": "CD3+, CD4+ and CD25- T cells from adjacent normal colonrectal tissues",
    "TTH": "CD3+, CD4+ and CD25- T cells from tumor",
    "PTR": "CD3+, CD4+ and CD25high T cells from peripheral blood",
    "NTR": "CD3+, CD4+ and CD25high T cells from adjacent normal colonrectal tissues",
    "TTR": "CD3+, CD4+ and CD25high T cells from tumor",
    "PTY": "CD3+, CD4+ and CD25mediate T cells from peripheral blood",
    "NTY": "CD3+, CD4+ and CD25mediate T cells from adjacent normal colonrectal tissues",
    "TTY": "CD3+, CD4+ and CD25medate T cells from tumor",
    "PP7": "CD3+, CD4+ T cells from peripheral blood",
    "NP7": "CD3+, CD4+ T cells from adjacent normal colonrectal tissues",
    "TP7": "CD3+, CD4+ T cells from tumor",
}
adata.obs["Sample"] = adata.obs["sampleType"].map(
    mapping
)

# %%
# Merge patient_meta
adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    patient_meta,
    how="left",
    on=["Patient_ID"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
geofetch = pd.read_csv(
    f"{d10_1038_s41586_018_0694_x}/GSE108989/GSE108989_PEP/GSE108989_PEP_raw.csv"
).dropna(axis=1, how="all")

# Merge accession numbers
adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    geofetch,
    how="left",
    left_on=["Patient_ID"],
    right_on=["sample_title"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
# Get original sample/patient meta and remove cell level meta
sample_meta = (
    adata.obs[
        adata.obs.columns[adata.obs.nunique() <= adata.obs["sampleType"].nunique()]
    ]
    .groupby("sampleType")
    .first()
    .reset_index()
)

# %%
sample_meta.to_csv(f"{artifact_dir}/Zhang_2018_CD3Pos-original_sample_meta.csv", index=False)

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="Zhang_2018_Nature",
    study_doi="10.1038/s41586-018-0694-x",
    study_pmid="30479382",
    tissue_processing_lab="Zemin Zhang lab",
    dataset="Zhang_2018_CD3Pos",
    medical_condition="colorectal cancer",
    platform="Smart-seq2",
    reference_genome="UCSC GRCh37/hg19",
    matrix_type="raw counts",
    hospital_location="Peking University People’s Hospital, Beijing",
    country="China",
    # Paper methods: Fresh tumour and adjacent normal tissue samples (at least 2 cm from matched tumour tissues) were surgically resected from the above-described patients
    tissue_cell_state="fresh",
    # Paper methods: None of the patients was treated with chemotherapy or radiation before tumour resection.
    treatment_status_before_resection="naive",
    NCBI_BioProject_accession="PRJNA429424",
)

# %%
mapping = {
    "PTC": "blood",
    "NTC": "normal",
    "TTC": "tumor",
    "PTH": "blood",
    "NTH": "normal",
    "TTH": "tumor",
    "PTR": "blood",
    "NTR": "normal",
    "TTR": "tumor",
    "PTY": "blood",
    "NTY": "normal",
    "TTY": "tumor",
    "PP7": "blood",
    "NP7": "normal",
    "TP7": "tumor",
}
adata.obs["sample_type"] = adata.obs["sampleType"].map(
    mapping
)

mapping = {
    "tumor": "colon",
    "normal": "colon",
    "blood": "blood",
}
adata.obs["sample_tissue"] = adata.obs["sample_type"].map(
    mapping
)

mapping = {
    "PTC": "CD8+",
    "NTC": "CD8+",
    "TTC": "CD8+",
    "PTH": "CD3+/CD4+/CD25-",
    "NTH": "CD3+/CD4+/CD25-",
    "TTH": "CD3+/CD4+/CD25-",
    "PTR": "CD3+/CD4+/CD25high",
    "NTR": "CD3+/CD4+/CD25high",
    "TTR": "CD3+/CD4+/CD25high",
    "PTY": "CD3+/CD4+/CD25mediate",
    "NTY": "CD3+/CD4+/CD25mediate",
    "TTY": "CD3+/CD4+/CD25medate",
    "PP7": "CD3+/CD4+",
    "NP7": "CD3+/CD4+",
    "TP7": "CD3+/CD4+",
}
adata.obs["enrichment_cell_types"] = adata.obs["sampleType"].map(
    mapping
)

# %%
adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]
adata.obs["sample_id"] = adata.obs["sampleType"]
adata.obs["patient_id"] = adata.obs["Patient_ID"]

adata.obs["age"] = adata.obs["Age"].fillna(np.nan).astype("Int64")

# define the dictionary mapping
mapping = {
    "Female": "female",
    "Male": "male",
}
adata.obs["sex"] = adata.obs["Gender"].map(mapping)

adata.obs["cell_type_study"] = adata.obs["majorCluster"]

mapping = {
    "Colon": "nan",
    "Rectum": "rectum",
    "Rectum ": "rectum",
}
adata.obs["anatomic_location"] = adata.obs["Histological typea"].map(
    mapping
)

mapping = {
    "Colon": "nan",
    "Rectum": "distal colon",
    "Rectum ": "distal colon",
}
adata.obs["anatomic_region"] = adata.obs["Histological typea"].map(
    mapping
)

mapping = {
    "ADC": "adenocarcinoma",
}
adata.obs["histological_type"] = adata.obs["Unnamed: 4"].map(
    mapping
)

adata.obs["microsatellite_status"] = adata.obs["MSI statusb"]

# %%
adata.obs["tumor_stage_TNM_T"] = "T" + adata.obs["pTNM: T"].astype(str)
adata.obs["tumor_stage_TNM_N"] = "N" + adata.obs["pTNM: N"].astype(str)
adata.obs["tumor_stage_TNM_M"] = "M" + adata.obs["pTNM: M"].astype(str)

adata.obs["tumor_stage_TNM"] = ah.pp.tnm_tumor_stage(adata.obs, "patient_id")
# Change "tumor_stage_TNM" value for patient "P0309" -> Only one that does not match completely
adata.obs.loc[adata.obs['patient_id'] == 'P0309', 'tumor_stage_TNM'] = "ⅢC"

adata.obs["tumor_dimensions"] = [item.replace(' ', '') for item in adata.obs["Tumour size"]]

mapping = {
    '11.5×7cm': "11.5",
    '6.5×4cm': "6.5",
    '5×4.5cm': "5",
    '6.5×3.5cm': "6.5",
    '10×10cm': "10",
    '1×0.8cm': "1",
    '9×4cm': "9",
    '6×4cm': "6",
    '7×6cm': "7",
    '6×6cm': "6",
    '4.5×4cm': "4.5",
}
adata.obs["tumor_size"] = adata.obs["tumor_dimensions"].map(mapping)

mapping = {
    "Moderate-differentiated": "G2",
    "Low-differentiated": "G1",
    "Well-differentiated": "G1",
    "Low or moderate-differentiated": "G1-G2",
}
adata.obs["tumor_grade"] = adata.obs["Grade"].map(
    mapping
)

# %%
# Split and explode the "Key Driver Mutations" column to get unique mutations per patient
mutation_list = (
    adata.obs["Mutations"]
    .str.split("/")
    .explode()
    .dropna()
    .str.strip()
    .unique()
)

# List comprehension to create new columns for each unique mutation
for mutation in mutation_list:
    adata.obs[f"{mutation}_status_driver_mut"] = adata.obs["Mutations"].apply(
        lambda x: "mut" if mutation in str(x) else "wt" if x != "None" else "nan"
    )

# %%
# Get the column names that contain "_status_driver_mut"
driver_mutation_columns = [
    col for col in adata.obs.columns if "_status_driver_mut" in col
]
# Set the values to NaN for "normal" sample_type in the driver mutation columns
adata.obs.loc[adata.obs["sample_type"] == "normal", driver_mutation_columns] = "nan"

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)
    
# Add newly created driver_mutation columns
ref_meta_cols += [f"{mutation}_status_driver_mut" for mutation in mutation_list]

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]]

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["GEO_sample_accession"]
    + "-"
    + adata.obs_names
)

# %%
adata

# %%
adata.obs["patient_id"].value_counts()

# %%
adata.obs["sample_id"].value_counts()

# %%
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
