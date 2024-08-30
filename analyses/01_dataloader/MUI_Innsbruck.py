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
# # Dataloader: MUI_Innsbruck (Tyrolean cohort)

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
import yaml
from nxfvars import nxfvars
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits
from tqdm.contrib.concurrent import process_map

# %% tags=["parameters"]
artifact_dir = nxfvars.get("artifact_dir", "/data/scratch/marteau/downloads/crc-atlas/")
data_path = nxfvars.get(
    "data_path", "../../data/own_datasets/"
)
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")
cpus = nxfvars.get("cpus", 2)
# gtf = f"{annotation}/ensembl.v108_gene_annotation_table.csv"

# %%
# sevenbridges_output = f"{data_path}/01_sevenbridges_output_files/"
bd_output = f"{data_path}/trajanoski-scrnaseq-colon/10_rhapsody_pipeline_v2/Fixed/"

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
patient_metadata = pd.read_csv(
    f"{data_path}/trajanoski-scrnaseq-colon/metadata/Clinical_Details_CRC_scRNAseq_project_BD_Rhapsody_2023.csv"
)
patient_metadata["multiplex_batch"] = patient_metadata["BD-Rhapsody File ID"].str.replace(
    "_", "-"
)


# %%
def load_adata_from_mtx(sample_meta):
    # Load data into memory
    matrix = fmm.mmread(
        f"{bd_output}/{sample_meta['multiplex_batch']}/{sample_meta['multiplex_batch']}_RSEC_MolsPerCell_Unfiltered_MEX/matrix.mtx.gz"
    )

    barcodes = pd.read_csv(
        f"{bd_output}/{sample_meta['multiplex_batch']}/{sample_meta['multiplex_batch']}_RSEC_MolsPerCell_Unfiltered_MEX/barcodes.tsv.gz",
        delimiter="\t",
        header=None,
        names=["Cell_Index"],
    )
    barcodes["obs_names"] = "Cell_Index_" + barcodes["Cell_Index"].astype(str)
    barcodes.set_index("obs_names", inplace=True)
    barcodes.index.name = None

    features = pd.read_csv(
        f"{bd_output}/{sample_meta['multiplex_batch']}/{sample_meta['multiplex_batch']}_RSEC_MolsPerCell_Unfiltered_MEX/features.tsv.gz",
        delimiter="\t",
        header=None,
        names=["var_names", "symbol", "type"],
    )

    # Assemble adata
    adata = sc.AnnData(X=matrix.T)
    adata.X = csr_matrix(adata.X)
    adata.var = features
    adata.obs = barcodes
    adata.obs = adata.obs.assign(**sample_meta)
    adata.var_names = adata.var["var_names"]
    adata.var_names.name = None

    adata_tmp = sc.read_h5ad(
        f"{bd_output}/{sample_meta['multiplex_batch']}/{sample_meta['multiplex_batch']}.h5ad"
    )
    adata_tmp.obs_names = "Cell_Index_" + adata_tmp.obs_names

    for col in ["Cell_Type_Experimental", "Sample_Tag", "Sample_Name"]:
        sample_tag_dict = adata_tmp.obs[col].to_dict()
        adata.obs[col] = adata.obs_names.map(sample_tag_dict)

    adata.obs["multiplex_batch"] = adata.obs["multiplex_batch"].str.replace("-", "_")
    return adata


# %%
adatas = process_map(
    load_adata_from_mtx,
    [r for _, r in patient_metadata.iterrows()],
    max_workers=cpus,
)

# %%
adata = anndata.concat(adatas, index_unique="-", join="outer", fill_value=0)

# %% [markdown]
# ## 2. Gene annotations

# %%
gtf = f"{annotation}/gencode.v44_gene_annotation_table.csv"
gtf = pd.read_csv(gtf)
gtf["ensembl"] = gtf["Geneid"].apply(ah.pp.remove_gene_version)

# map back gene symbols to .var
gene_symbols = gtf.set_index("GeneSymbol")["ensembl"].to_dict()

# %%
adata.var["symbol"] = adata.var_names
adata.var["ensembl"] = (
    adata.var["symbol"].map(gene_symbols).fillna(value=adata.var["symbol"])
)

# %%
adata.var_names = adata.var["ensembl"]
adata.var_names.name = None

# %%
unmapped_genes = ah.pp.find_unmapped_genes(adata)
len(unmapped_genes)

# %%
# Reorder by gtf (i.e. chromosome position)
gene_index = gtf[gtf["ensembl"].isin(adata.var_names)]["ensembl"].values
missing_genes = adata[:, ~adata.var_names.isin(gtf["ensembl"])].var_names.values
gene_index = np.append(gene_index, missing_genes)
adata = adata[:, gene_index]

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="MUI_Innsbruck",
    tissue_processing_lab="Zlatko Trajanoski lab",
    dataset="MUI_Innsbruck",
    medical_condition="colorectal cancer",
    hospital_location="Landeskrankenhaus Innsbruck - tirol kliniken",
    country="Austria",
    treatment_status_before_resection="naive",
    enrichment_cell_types="naive",
    tissue_cell_state="fresh",
    platform="BD Rhapsody",
    reference_genome="gencode.v44",
    matrix_type="raw counts",
    tumor_source="core",
)

mapping = {
    "T": "tumor",
    "Multiplet": "Multiplet",
    "B": "blood",
    "N": "normal",
    "Undetermined": "Undetermined",
}
adata.obs["sample_type"] = adata.obs["Sample_Name"].map(mapping)

adata.obs["multiplex_batch"] = adata.obs["multiplex_batch"].str.split("_").str[0]
adata.obs["patient_id"] = adata.obs["multiplex_batch"]
adata.obs["sample_id"] = adata.obs["patient_id"].astype(str) + "_" + adata.obs["sample_type"].astype(str)

mapping = {
    "T": "colon",
    "Multiplet": "Multiplet",
    "B": "blood",
    "N": "colon",
    "Undetermined": "Undetermined",
}
adata.obs["sample_tissue"] = adata.obs["Sample_Name"].map(mapping)

adata.obs["age"] = adata.obs["Age (at surgery)"].fillna(np.nan).astype("Int64")

adata.obs["sex"] = adata.obs["Gender"]

# define the dictionary mapping
mapping = {
    "rechte Flexur": "hepatic flexure",
    "Sigma": "sigmoid colon",
    "Rektum (unteres Drittel)": "rectum",
    "Coecum": "cecum",
    "Colon transversum": "transverse colon",
    "Colon descendens": "descending colon",
    "rektosigmoidaler Übergang": "rectosigmoid junction",
    "Rektum (oberes Drittel)": "rectum",
    "Kolon linke Flexur": "splenic flexure",
}
adata.obs["anatomic_location"] = adata.obs["Location of tumor"].map(mapping)

mapping = {
    "rechte Flexur": "proximal colon",
    "Sigma": "distal colon",
    "Rektum (unteres Drittel)": "distal colon",
    "Coecum": "proximal colon",
    "Colon transversum": "proximal colon",
    "Colon descendens": "distal colon",
    "rektosigmoidaler Übergang": "distal colon",
    "Rektum (oberes Drittel)": "distal colon",
    "Kolon linke Flexur": "distal colon",
}
adata.obs["anatomic_region"] = adata.obs["Location of tumor"].map(mapping)

mapping = {
    "mäßig differenziertes Adenokarzinom": "adenocarcinoma",
    "tubulopapillär strukturierten Adenokarzinoms": "adenocarcinoma",
    "mäßig differenziertes tubulär und kribriform strukturiertes Adenokarzinom": "adenocarcinoma",
    "exulzeriertes solid und kribriform gebautes gering differenziertes Adenokarzinom": "adenocarcinoma",
    "Adenokarzinom": "adenocarcinoma",
    "mäßig bis gering differenziertes tubulopapillär und kribriform strukturiertes muzinöses Adenokarzinom": "mucinous adenocarcinoma",
    "mäßig differenziertes tubulopapillär und kribriform strukturiertes Adenokarzinom": "adenocarcinoma",
    "mittelgradig differenzierten invasiven Adenokarzinom": "adenocarcinoma",
    "mäßiggradig differenziertes Adenokarzinom": "adenocarcinoma",
}
adata.obs["histological_type"] = adata.obs["Histology"].map(mapping)

mapping = {
    "mäßig differenziertes Adenokarzinom": "G2",  # Moderately differentiated?!
    "tubulopapillär strukturierten Adenokarzinoms": "nan",
    "mäßig differenziertes tubulär und kribriform strukturiertes Adenokarzinom": "G2",
    "exulzeriertes solid und kribriform gebautes gering differenziertes Adenokarzinom": "G3",  # Poorly differentiated?!
    "Adenokarzinom": "nan",
    "Mäßig bis gering differenziertes tubulopapillär und kribriform strukturiertes muzinöses Adenokarzinom": "G2-G3",
    "mäßig differenziertes tubulopapillär und kribriform strukturiertes Adenokarzinom": "G2",
    "mittelgradig differenzierten invasiven Adenokarzinom": "G2",
    "mäßiggradig differenziertes Adenokarzinom": "G2",
}
adata.obs["tumor_grade"] = adata.obs["Histology"].map(mapping)


adata.obs["tumor_stage_TNM_T"] = adata.obs["Tumor stadium (TNM)"].apply(
    lambda x: re.search(r"(T[0-4][a-c]?)", x).group(1)
    if re.search(r"(T[0-4][a-c]?)", x)
    else None
)
adata.obs["tumor_stage_TNM_N"] = adata.obs["Tumor stadium (TNM)"].apply(
    lambda x: re.search(r"(N[0-4][a-c]?)", x).group(1)
    if re.search(r"(N[0-4][a-c]?)", x)
    else None
)
adata.obs["tumor_stage_TNM_M"] = adata.obs["Tumor stadium (TNM)"].apply(
    lambda x: re.search(r"((M[0-1]|Mx)[a-c]?)", x).group(1)
    if re.search(r"((M[0-1]|Mx)[a-c]?)", x)
    else None
)

# Replace None values in 'column_name' with 'Mx'
adata.obs["tumor_stage_TNM_M"].fillna("Mx", inplace=True)

adata.obs["tumor_stage_TNM"] = ah.pp.tnm_tumor_stage(adata.obs, "patient_id")


mapping = {
    "MSI-H (MLH1-; PMS2-; MSH2+; MSH6+)": "MSI-H",
    "MSS": "MSS",
}
adata.obs["microsatellite_status"] = adata.obs["Genomic status"].map(mapping)

adata.obs["tumor_grade"] = adata.obs["Grading"]

column_rename_mapping = {
    "KRAS": "KRAS_status_driver_mut",
    "NRAS": "NRAS_status_driver_mut",
    "BRAF": "BRAF_status_driver_mut",
    "HER2": "HER2_status_driver_mut",
    "panTRK": "panTRK_status_driver_mut",
    "AKT1": "AKT1_status_driver_mut",
    "TP53": "TP53_status_driver_mut",
    "CTNNB1": "CTNNB1_status_driver_mut",
    "ABL1": "ABL1_status_driver_mut",
    "RET": "RET_status_driver_mut",
}

# Rename only the specified columns
adata.obs.rename(columns=column_rename_mapping, inplace=True)

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)
# Add newly created driver_mutation columns
ref_meta_cols += [col for col in adata.obs.columns if "_status_driver_mut" in col]

# %%
ref_meta_cols += [
    "multiplex_batch",
    "Tumor budding",
    "Cell_Type_Experimental",
    "Sample_Tag",
    "Sample_Name",
]

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
adata.obs_names = (
    adata.obs["dataset"].astype(str)
    + "-"
    + adata.obs["multiplex_batch"].str.split("_").str[0].astype(str)
    + "-"
    #+ adata.obs["Sample_Tag"].astype(str)
    #+ "_"
    + adata.obs_names.str.split("-").str[0].str.split("Cell_Index_").str[1]
)

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
# Get original sample/patient meta and remove cell level meta
sample_meta = (
    adata.obs[
        adata.obs.columns[adata.obs.nunique() <= adata.obs["sample_id"].nunique()]
    ]
    .groupby("sample_id")
    .first()
    .reset_index()
)

# %%
sample_meta.to_csv(
    f"{artifact_dir}/Trajanoski_2023-original_sample_meta.csv",
    index=False,
)

# %%
adata.obs["sample_id"].value_counts()

# %%
adata.obs["patient_id"].unique()

# %%
adata

# %%
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
