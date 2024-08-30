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
# # Dataloader: Zhang_2020_Cell

# %%
import os
from pathlib import Path

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
gtf = f"{annotation}/ensembl.v84_gene_annotation_table.csv"
d10_1016_j_cell_2020_03_048 = f"{databases_path}/d10_1016_j_cell_2020_03_048"
figshare = f"{d10_1016_j_cell_2020_03_048}/figshare/CRC_leukocyte_cellranger_outs/"
cpus = nxfvars.get("cpus", 6)

# %%
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

# %%
# Read the YAML file and load it into a dictionary
with open(reference_meta, "r") as f:
    ref_meta_dict = yaml.load(f, Loader=yaml.Loader)

# %% [markdown]
# ## 1. Load raw counts - 10X data

# %%
adata_10x = anndata.AnnData(
    X=fmm.mmread(f"{d10_1016_j_cell_2020_03_048}/figshare/CLX_matrix.mtx.gz"),
    obs=pd.read_csv(
        f"{d10_1016_j_cell_2020_03_048}/figshare/CLX_barcodes.tsv",
        delimiter="\t",
        header=None,
        names=["obs_names"],
        index_col=0,
    ),
    var=pd.read_csv(
        f"{d10_1016_j_cell_2020_03_048}/figshare/CLX_features.tsv",
        delimiter="\t",
        header=None,
        index_col=0,
    ),
)
adata_10x.X = csr_matrix(adata_10x.X.astype(int))

adata_10x.obs_names.name = None
adata_10x.var_names.name = None

cell_meta = pd.read_csv(f"{d10_1016_j_cell_2020_03_048}/figshare/CLX_metadata.csv")

# Merge accession numbers
adata_10x.obs["CellName"] = adata_10x.obs_names
adata_10x.obs = pd.merge(
    adata_10x.obs,
    cell_meta,
    how="left",
    on=["CellName"],
    validate="m:1",
).set_index("Sample_BarcodeID")
adata_10x.obs_names.name = None

# %%
patient_meta = pd.read_csv(f"{d10_1016_j_cell_2020_03_048}/metadata/Table_S1.csv")

# Merge accession numbers
adata_10x.obs["obs_names"] = adata_10x.obs_names
adata_10x.obs = pd.merge(
    adata_10x.obs,
    patient_meta[patient_meta["Platform"] == "10x Genomics"],
    how="left",
    left_on=["Sample"],
    right_on=["Patient"],
    validate="m:1",
).set_index("obs_names")
adata_10x.obs_names.name = None

# %%
geofetch = pd.read_csv(
    f"{d10_1016_j_cell_2020_03_048}/GSE146771/GSE146771_PEP/GSE146771_PEP_raw.csv"
).dropna(axis=1, how="all")

geofetch["platform"] = (
    geofetch["sample_title"].str.split(" ").str[1].str.replace(r"\(|\)", "", regex=True)
)

# Merge accession numbers
adata_10x.obs["obs_names"] = adata_10x.obs_names
adata_10x.obs = pd.merge(
    adata_10x.obs,
    geofetch[geofetch["platform"] == "10x"],
    how="left",
    left_on=["Patient"],
    right_on=["patient"],
    validate="m:1",
).set_index("obs_names")
adata_10x.obs_names.name = None

# %%
adata_10x.obs = adata_10x.obs.assign(
    study_id="Zhang_2020_Cell",
    study_doi="10.1016/j.cell.2020.03.048",
    study_pmid="32302573",
    tissue_processing_lab="Ziyi Li lab",
    dataset="Zhang_2020_10X_CD45Pos",
    medical_condition="colorectal cancer",
    matrix_type="raw counts",
    # reference_genome="",
    cellranger_version="2.1.0",
    platform="10x 3' v2",
    hospital_location="Peking University People’s Hospital, Beijing",
    country="China",
    tissue_cell_state="fresh",
    enrichment_cell_types="CD45+",
    # Paper methods: None of them were treated with chemotherapy or radiation prior to tumor resection.
    treatment_status_before_resection="naive",
    NCBI_BioProject_accession="PRJNA611928",
)

# %%
adata_10x.obs["sample_id"] = adata_10x.obs["CellName"].str.rsplit("_", n=1).str[0]
adata_10x.obs["sample_id"] = adata_10x.obs["sample_id"].str.split("_", n=1).str[1]

adata_10x.obs["patient_id"] = adata_10x.obs["Patient"]
adata_10x.obs["GEO_sample_accession"] = adata_10x.obs["sample_geo_accession"]

adata_10x.obs["age"] = adata_10x.obs["Age"].fillna(np.nan).astype("Int64")

# define the dictionary mapping
mapping = {
    "Female": "female",
    "Male": "male",
}
adata_10x.obs["sex"] = adata_10x.obs["Gender"].map(mapping)

adata_10x.obs["histological_type"] = adata_10x.obs["type"].map(
    {"ADC": "adenocarcinoma"}
)

mapping = {
    "T": "tumor",
    "N": "normal",
    "P": "blood",
}
adata_10x.obs["sample_type"] = adata_10x.obs["Tissue"].map(mapping)

mapping = {
    "T": "colon",
    "N": "colon",
    "P": "blood",
}
adata_10x.obs["sample_tissue"] = adata_10x.obs["Tissue"].map(mapping)

mapping = {
    "ReC": "rectum",  # ReC: Rectum cancer
    "RCC": "nan",  # RCC: right-sided colon cancer
}
adata_10x.obs["anatomic_location"] = adata_10x.obs["Locationb"].map(mapping)

mapping = {
    "ReC": "distal colon",
    "RCC": "proximal colon",
}
adata_10x.obs["anatomic_region"] = adata_10x.obs["Locationb"].map(mapping)

adata_10x.obs["microsatellite_status"] = adata_10x.obs["MSI statusc"]

mapping = {
    "High-differentiated": "G3",
    "Moderate-differentiated": "G2",
    "Low or moderate differentiated": "G1-G2",
}
adata_10x.obs["tumor_grade"] = adata_10x.obs["Grade"].map(mapping)

adata_10x.obs["tumor_stage_TNM_T"] = "T" + adata_10x.obs["pTNM: T"].astype(str)
adata_10x.obs["tumor_stage_TNM_N"] = "N" + adata_10x.obs["pTNM: N"].astype(str)
adata_10x.obs["tumor_stage_TNM_M"] = "M" + adata_10x.obs["pTNM: M"].astype(str)
adata_10x.obs["tumor_stage_TNM"] = adata_10x.obs["Stage"]

mapping = {
    "5×3.9 cm": 5,
    "3.5x2cm": 3.5,
    "3.2×3.2 cm": 3.2,
    "5×4 cm": 5,
    "11.5×7 cm": 11.5,
    "6×5 cm": 6,
    "7×6×4 cm": 7,
    "3.5×2.8 cm": 3.5,
    "6.5×6 cm": 6.5,
    "4.5×2.5 cm": 4.5,
}
adata_10x.obs["tumor_size"] = adata_10x.obs["Tumour size"].map(mapping)
adata_10x.obs["tumor_dimensions"] = adata_10x.obs["Tumour size"]

adata_10x.obs["original_obs_names"] = adata_10x.obs["Unnamed: 0"]
adata_10x.obs["cell_type_coarse_study"] = adata_10x.obs["Global_Cluster"]
adata_10x.obs["cell_type_study"] = adata_10x.obs["Sub_Cluster"]

# %%
adata_10x.obsm["X_umap"] = np.array(adata_10x.obs[["Global_tSNE_1", "Global_tSNE_2"]])

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
# Subset adata for columns in the reference meta
adata_10x.obs = adata_10x.obs[
    [col for col in ref_meta_cols if col in adata_10x.obs.columns]
].copy()

# %%
adata_10x.obs_names = (
    adata_10x.obs["dataset"]
    + "-"
    + adata_10x.obs["GEO_sample_accession"]
    + "_"
    + adata_10x.obs_names.str.split("_").str[1]
    + "_"
    + adata_10x.obs_names.str.split("_").str[2]
    + "-"
    + adata_10x.obs_names.str.rsplit("_", n=1).str[1]
)

# %% [markdown]
# ### Gene annotations

# %%
# ensembl gene ids from figshare "CRC_leukocyte_cellranger_outs.tar.gz", I guess this is from the 10X data only ?!
# number of ensembl ids is the same for all samples! taking a random one!
# see: https://figshare.com/articles/dataset/CRC_CD45_rda/14820318
gene_ids = (
    pd.read_csv(
        f"{figshare}/P0123-T/genes.tsv",
        delimiter="\t",
        header=None,
        names=["ensembl", "symbol"],
    )
    .set_index("symbol")["ensembl"]
    .to_dict()
)

# %%
adata_10x.var = adata_10x.var.rename_axis("symbol").reset_index()
adata_10x.var["ensembl"] = (
    adata_10x.var["symbol"].map(gene_ids).fillna(value=adata_10x.var["symbol"])
)
adata_10x.var_names = adata_10x.var["ensembl"].apply(ah.pp.remove_gene_version)
adata_10x.var_names.name = None

# %%
# for some reason this maps not all genes ...
unmapped_genes = ah.pp.find_unmapped_genes(adata_10x)
len(unmapped_genes)

# %%
assert np.all(np.modf(adata_10x.X.data)[0] == 0), "X does not contain all integers"
assert adata_10x.var_names.is_unique
assert adata_10x.obs_names.is_unique

# %%
adata_10x

# %%
adata_10x.obs["sample_id"].value_counts()

# %%
dataset = adata_10x.obs["dataset"].unique()[0]
adata_10x.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")

# %% [markdown]
# ## 2. Load raw counts - Smart-seq2

# %%
adata = anndata.AnnData(
    X=fmm.mmread(f"{d10_1016_j_cell_2020_03_048}/figshare/CLS_matrix.mtx.gz"),
    obs=pd.read_csv(
        f"{d10_1016_j_cell_2020_03_048}/figshare/CLS_barcodes.tsv",
        delimiter="\t",
        header=None,
        names=["obs_names"],
        index_col=0,
    ),
    var=pd.read_csv(
        f"{d10_1016_j_cell_2020_03_048}/figshare/CLS_features.tsv",
        delimiter="\t",
        header=None,
        # names=["ensembl", "symbol"],
        index_col=0,
    ),
)
adata.X = csr_matrix(adata.X.astype(int))
adata.obs_names.name = None
adata.var_names.name = None

cell_meta = pd.read_csv(f"{d10_1016_j_cell_2020_03_048}/figshare/CLS_metadata.csv")

# Merge accession numbers
adata.obs["CellName"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    cell_meta,
    how="left",
    on=["CellName"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
# Merge patient_meta
adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    patient_meta[patient_meta["Platform"] == "Smart-seq2"],
    how="left",
    left_on=["Sample"],
    right_on=["Patient"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# Merge geofetch
adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    geofetch[geofetch["platform"] == "Smart-seq2"],
    how="left",
    left_on=["Patient"],
    right_on=["patient"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
adata.obs = adata.obs.assign(
    study_id="Zhang_2020_Cell",
    study_doi="10.1016/j.cell.2020.03.048",
    study_pmid="32302573",
    tissue_processing_lab="Ziyi Li lab",
    dataset="Zhang_2020_CD45Pos_CD45Neg",
    medical_condition="colorectal cancer",
    matrix_type="raw counts",
    reference_genome="UCSC GRCh38/hg38",
    platform="Smart-seq2",
    hospital_location="Peking University People’s Hospital, Beijing",
    country="China",
    tissue_cell_state="fresh",
    enrichment_cell_types="CD45+/CD45-",
    # Paper: 18 treatment-naive CRC patients
    treatment_status_before_resection="naive",
    NCBI_BioProject_accession="PRJNA611928",
)

# %%
adata.obs["sample_id"] = adata.obs["Patient"] + "_" + adata.obs["Tissue"]
# adata.obs["sample_id"] = adata.obs["CellName"].str.rsplit("_", n=1).str[0]

adata.obs["patient_id"] = adata.obs["Patient"]
adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]

adata.obs["age"] = adata.obs["Age"].fillna(np.nan).astype("Int64")

# define the dictionary mapping
mapping = {
    "Female": "female",
    "Male": "male",
}
adata.obs["sex"] = adata.obs["Gender"].map(mapping)

adata.obs["histological_type"] = adata.obs["type"].map({"ADC": "adenocarcinoma"})

mapping = {
    "T": "tumor",
    "N": "normal",
    "P": "blood",
}
adata.obs["sample_type"] = adata.obs["Tissue"].map(mapping)

mapping = {
    "T": "colon",
    "N": "colon",
    "P": "blood",
}
adata.obs["sample_tissue"] = adata.obs["Tissue"].map(mapping)

mapping = {
    "ReC": "rectum",  # ReC: Rectum cancer
    "RCC": "nan",  # RCC: right-sided colon cancer
}
adata.obs["anatomic_location"] = adata.obs["Locationb"].map(mapping)

mapping = {
    "ReC": "distal colon",
    "RCC": "proximal colon",
}
adata.obs["anatomic_region"] = adata.obs["Locationb"].map(mapping)

adata.obs["microsatellite_status"] = adata.obs["MSI statusc"]

mapping = {
    "High-differentiated": "G3",
    "Moderate-differentiated": "G2",
    "Low or moderate differentiated": "G1-G2",
    "Low-differentiated": "G1",
}
adata.obs["tumor_grade"] = adata.obs["Grade"].map(mapping)

adata.obs["tumor_stage_TNM_T"] = "T" + adata.obs["pTNM: T"].astype(str)
adata.obs["tumor_stage_TNM_N"] = "N" + adata.obs["pTNM: N"].astype(str)
adata.obs["tumor_stage_TNM_M"] = "M" + adata.obs["pTNM: M"].astype(str)
adata.obs["tumor_stage_TNM"] = adata.obs["Stage"]

mapping = {
    "5×4 cm": 5,
    "6.5×6 cm": 6.5,
    "5×4.5 cm": 5,
    "6.5×3.5 cm": 6.5,
    "10×10 cm": 10,
    "6x3cm": 6,
    "3.5x3.4cm": 3.5,
    "9×4 cm": 9,
    "6×4 cm": 6,
    "4.5×4 cm": 4.5,
}
adata.obs["tumor_size"] = adata.obs["Tumour size"].map(mapping)
adata.obs["tumor_dimensions"] = adata.obs["Tumour size"]

adata.obs["original_obs_names"] = adata.obs["Unnamed: 0"]
adata.obs["cell_type_coarse_study"] = adata.obs["Global_Cluster"]
adata.obs["cell_type_study"] = adata.obs["Sub_Cluster"]

# %%
adata.obsm["X_umap"] = np.array(
    adata.obs[["Global_tSNE_1", "Global_tSNE_2"]]
)

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["GEO_sample_accession"]
    + "-"
    + adata.obs_names.str.replace("n-", "N")
    .str.replace("-", "_")
    .str.replace("N", "_N")
    .str.replace("T_", "T")
)

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %% [markdown]
# ### Gene annotations

# %%
# ### 2. Load UCSC ensemblToGeneName table for gene mapping
# Dowloaded from [UCSC hgTables](https://genome.ucsc.edu/cgi-bin/hgTables). See also: https://www.biostars.org/p/92939/.
# Roughly 3000 genes do not have an ensembl id. I don't think it is a good idea to map these using the gene symbols to gencode/ensembl gtf files - which is possible. This UCSC GRCh38/hg38 assembly is from Dec 2013 and I don't think that the gene coordinates match any more. Will only use the official UCSC to ensembl mapping and drop the rest!
# Will have to correct fo gene length later on when I have gene length from the ensembl ids, probably not ideal ...

# %%
UCSC_ensemblToGeneName = pd.read_csv(f"{d10_1016_j_cell_2020_03_048}/metadata/UCSC_hg38_ensemblToGeneName.txt", sep='\t')
UCSC_ensemblToGeneName=UCSC_ensemblToGeneName.drop_duplicates(subset=['hg38.wgEncodeGencodeAttrsV20.geneId'])
gene_ids = UCSC_ensemblToGeneName.set_index("hg38.wgEncodeGencodeBasicV20.name2")["hg38.wgEncodeGencodeAttrsV20.geneId"].to_dict()

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
adata

# %%
adata.obs["sample_id"].value_counts()

# %%
adata_10x.obs["dataset"].unique()[0]

# %%
adata

# %%
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")


# %% [markdown]
# ### additional cellranger output available

# %%
def _load_mtx(file):
    file_path = f"{zenodo}/{file}"
    adata = anndata.AnnData(
        X=fmm.mmread(f"{file_path}/matrix.mtx").T,
        obs=pd.read_csv(
            f"{file_path}/barcodes.tsv",
            delimiter="\t",
            header=None,
            names=["obs_names"],
            index_col=0,
        ),
        var=pd.read_csv(
            f"{file_path}/genes.tsv",
            delimiter="\t",
            header=None,
            names=["ensembl", "symbol"],
            index_col=0,
        ),
    )
    adata.obs["file"] = file
    adata.obs_names.name = None
    adata.X = csr_matrix(adata.X.astype(int))
    return adata


# %%
# For now will not look at cellranger output from this study
files = [str(x) for x in Path(figshare).glob("**/")]
files = [Path(file).stem.rsplit("/")[0] for file in files]
files.remove("CRC_leukocyte_cellranger_outs")
files.sort()

# %%
files

# %%
#adatas = process_map(_load_mtx, files, max_workers=cpus)
#adata = anndata.concat(adatas, index_unique="-", join="outer", fill_value=0)xw
