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
# # Dataloader: Pelka_2021_Cell

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

# %% tags=["parameters"]
artifact_dir = nxfvars.get("artifact_dir", "/data/scratch/marteau/downloads/crc-atlas/")
databases_path = nxfvars.get("databases_path", "../../data/external_datasets")
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")
cpus = nxfvars.get("cpus", 2)

# %%
d10_1016_j_cell_2021_08_003 = f"{databases_path}/d10_1016_j_cell_2021_08_003"
gtf = f"{annotation}/gencode.v28_lift37_gene_annotation_table.csv"
SCP1162 = f"{d10_1016_j_cell_2021_08_003}/single_cell_portal/SCP1162"

# %%
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

# %%
# Read the YAML file and load it into a dictionary
with open(reference_meta, "r") as f:
    ref_meta_dict = yaml.load(f, Loader=yaml.Loader)

# %% [markdown]
# ## 1. Load raw adata counts from GEO

# %%
# (A) raw counts from GSE178341
adata = sc.read_10x_h5(
    f"{d10_1016_j_cell_2021_08_003}/GSE178341/GSE178341_crc10x_full_c295v4_submit.h5"
)
adata.X = csr_matrix(adata.X)

metatable = pd.read_csv(
    f"{d10_1016_j_cell_2021_08_003}/GSE178341/GSE178341_crc10x_full_c295v4_submit_metatables.csv.gz"
)
cell_annot = pd.read_csv(
    f"{d10_1016_j_cell_2021_08_003}/GSE178341/GSE178341_crc10x_full_c295v4_submit_cluster.csv.gz"
)

adata.var_names = [gene_id.strip() for gene_id in adata.var_names]
adata.obs_names = [umi.strip() for umi in adata.obs_names]

# %% [markdown]
# ### 1a. Gene annotations
# Get ensembl ids without version numbers as var_names and append sex chromosome ids (_PAR_Y genes)

# %%
# gene_annotation_table.csv was generated using:
# zcat gencode.v25.primary_assembly.annotation.gtf.gz | awk 'BEGIN{FS="\t";OFS=","}$3=="gene"{split($9,a,";");split(a[1],gene_id,"\"");split(a[4],gene_name,"\""); print gene_id[2],gene_name[2]}' | sed '1i\Geneid,GeneSymbol' > gencode.v25_gene_annotation_table.csv

# %%
gtf = pd.read_csv(gtf)
gtf["ensembl"] = gtf["Geneid"].apply(ah.pp.remove_gene_version)

# Append sex chromosome info "_PAR_Y"
gtf.loc[gtf["Geneid"].str.endswith("_PAR_Y"), "ensembl"] = (
    gtf.loc[gtf["Geneid"].str.endswith("_PAR_Y"), "ensembl"] + "_PAR_Y"
)

# map back gene symbols to .var
gene_symbols = gtf.set_index("ensembl")["GeneSymbol"].to_dict()

# %%
adata.var["ensembl"] = adata.var["gene_ids"].apply(ah.pp.remove_gene_version)

# Append sex chromosome info "_PAR_Y"
adata.var.loc[adata.var["gene_ids"].str.endswith("_PAR_Y"), "ensembl"] = (
    adata.var.loc[adata.var["gene_ids"].str.endswith("_PAR_Y"), "ensembl"] + "_PAR_Y"
)

adata.var["symbol"] = adata.var_names
adata.var["GeneSymbol"] = (
    adata.var["ensembl"].map(gene_symbols).fillna(value=adata.var["ensembl"])
)
adata.var_names = adata.var["ensembl"]
adata.var_names.name = None

# %%
# Convert _PAR_Y genes to new gencode v44 ensembl ids manually where possible, otherwise remove _PAR_Y from string
adata.var["var_names"] = adata.var_names
mapping = {
    "ENSG00000182378_PAR_Y": "ENSG00000292344",
    "ENSG00000178605_PAR_Y": "ENSG00000292358",
    "ENSG00000226179_PAR_Y": "ENSG00000226179_PAR_Y",
    "ENSG00000167393_PAR_Y": "ENSG00000292327",
    "ENSG00000281849_PAR_Y": "ENSG00000281849_PAR_Y",
    "ENSG00000280767_PAR_Y": "ENSG00000280767_PAR_Y",
    "ENSG00000185960_PAR_Y": "ENSG00000292354",
    "ENSG00000237531_PAR_Y": "ENSG00000237531_PAR_Y",
    "ENSG00000205755_PAR_Y": "ENSG00000292363",
    "ENSG00000198223_PAR_Y": "ENSG00000292357",
    "ENSG00000185291_PAR_Y": "ENSG00000292332",
    "ENSG00000169100_PAR_Y": "ENSG00000292334",
    "ENSG00000236871_PAR_Y": "ENSG00000236871_PAR_Y",
    "ENSG00000236017_PAR_Y": "ENSG00000236017_PAR_Y",
    "ENSG00000169093_PAR_Y": "ENSG00000292339",
    "ENSG00000182162_PAR_Y": "ENSG00000292333",
    "ENSG00000197976_PAR_Y": "ENSG00000292343",
    "ENSG00000196433_PAR_Y": "ENSG00000292336",
    "ENSG00000223511_PAR_Y": "ENSG00000223511_PAR_Y",
    "ENSG00000234622_PAR_Y": "ENSG00000234622_PAR_Y",
    "ENSG00000169084_PAR_Y": "ENSG00000292338",
    "ENSG00000214717_PAR_Y": "ENSG00000292345",
    "ENSG00000230542_PAR_Y": "ENSG00000230542_PAR_Y",
    "ENSG00000002586_PAR_Y": "ENSG00000292348",
    "ENSG00000168939_PAR_Y": "ENSG00000168939_PAR_Y",
    "ENSG00000124333_PAR_Y": "ENSG00000292366",
    "ENSG00000124334_PAR_Y": "ENSG00000292373",
    "ENSG00000185203_PAR_Y": "ENSG00000185203_PAR_Y",
}
adata.var["var_names"] = (
    adata.var["var_names"].map(mapping).fillna(value=adata.var["var_names"])
)
adata.var["var_names"] = adata.var["var_names"].str.replace("_PAR_Y", "", regex=False)

adata.var_names = adata.var["var_names"]
adata.var_names.name = None
del adata.var["var_names"]

# %%
unmapped_genes = ah.pp.find_unmapped_genes(adata)
len(unmapped_genes)

# %%
# Sum up duplicate gene ids
duplicated_ids = adata.var_names[adata.var_names.duplicated()].unique()
adata = ah.pp.aggregate_duplicate_gene_ids(adata, duplicated_ids)

# %% [markdown]
# ### 1b. Compile sample/patient metadata

# %%
meta = pd.merge(
    metatable,
    cell_annot,
    how="left",
    left_on=["cellID"],
    right_on=["sampleID"],
    validate="m:1",
).set_index("cellID")
meta.index.name = None

adata.obs = pd.merge(adata.obs, meta, left_index=True, right_index=True)

# %% [markdown] tags=[]
# ## 2. Load processed adata counts from single cell portal
# For some reason the data from single cell portal has additional metadata plus roughly a 1000 barcodes more for a single patient "C144" - probably an additional sample. Unfortunately the scp data is quantile normalized which cannot be reversed. Will drop the counts.

# %%
# (B) processed data from single cell portal including cell annotations

# Load data into memory
matrix = fmm.mmread(
    f"{SCP1162}/expression/5fbf5a26771a5b0db8fe7a8b/matrix.mtx.gz"  # slower alternative: scipy.io.mmread()
)
features = pd.read_csv(
    f"{SCP1162}/expression/5fbf5a26771a5b0db8fe7a8b/matrix.genes.tsv",
    delimiter="\t",
    header=None,
    names=["ensembl", "symbol"],
)
features["ensembl"] = features["ensembl"].apply(ah.pp.remove_gene_version)

barcodes = pd.read_csv(
    f"{SCP1162}/expression/5fbf5a26771a5b0db8fe7a8b/matrix.barcodes.tsv",
    delimiter="\t",
    header=None,
    names=["NAME"],
)
metatable = pd.read_csv(
    f"{SCP1162}/metadata/metatable_v3_fix_v2.tsv",
    delimiter="\t",
).tail(-1)
cell_annot = pd.read_csv(
    f"{SCP1162}/all_viz/crc10x_tSNE_cl_global.tsv",
    delimiter="\t",
).tail(-1)

# Merge provided metadata to barcodes
merge_params = {
    "how": "left",
    "on": ["NAME"],
    "validate": "m:1",
}
barcodes = pd.merge(barcodes, metatable, **merge_params)
barcodes = pd.merge(barcodes, cell_annot, **merge_params).set_index("NAME")
barcodes.index.name = None

adata_p = sc.AnnData(X=matrix.T)
adata_p.X = csr_matrix(adata_p.X)
adata_p.var = features
adata_p.obs = barcodes
adata_p.obsm["X_umap"] = barcodes[["X", "Y"]].astype(float).to_numpy()

adata_p.obs_names = [umi.strip() for umi in adata_p.obs_names]

# %% [markdown]
# ### 2a. Compile sample/patient metadata from original study supplemetary tables

# %%
# Get Polyp sample metadata
meta_xls = pd.concat(
    pd.read_excel(
        f"{d10_1016_j_cell_2021_08_003}/metadata/1-s2.0-S0092867421009454-mmc1.xlsx",
        sheet_name=["A. Cohort overview"],
    ),
    ignore_index=True,
)

# Get Polyp sample metadata
meta_xls2 = pd.concat(
    pd.read_excel(
        f"{d10_1016_j_cell_2021_08_003}/metadata/1-s2.0-S0092867421009454-mmc1.xlsx",
        sheet_name=["B. 10x channels"],
    ),
    ignore_index=True,
)
meta_xls2["PatientBarcode_SpecimenType"] = (
    meta_xls2["PatientBarcode"] + "_" + meta_xls2["SpecimenType"]
)

meta = pd.merge(
    meta_xls2,
    meta_xls,
    how="left",
    on=["PatientBarcode_SpecimenType"],
    validate="m:1",
)

# %%
adata_p.obs["obs_names"] = adata_p.obs_names
adata_p.obs = pd.merge(
    adata_p.obs,
    meta,
    how="left",
    left_on=["biosample_id"],
    right_on=["batchID"],
    validate="m:1",
).set_index("obs_names")

# %%
# Subset by obs_names (single cell portal data has additional barcodes ...)
adata_p = adata_p[adata_p.obs_names.isin(adata.obs_names), :]

# %%
# Merge all available metadata
adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    adata_p.obs,
    how="left",
    on=["obs_names"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %% [markdown]
# ## Get SRA and GEO accession numbers

# %%
fetchngs = pd.read_csv(
    f"{d10_1016_j_cell_2021_08_003}/fetchngs/PRJNA723926/samplesheet/samplesheet.csv"
).dropna(axis=1, how="all")

fetchngs["batchID"] = fetchngs["sample_alias"].str.split("-").str[1]

# %%
geofetch = pd.read_csv(
    f"{d10_1016_j_cell_2021_08_003}/GSE178341/GSE178341_PEP/GSE178341_PEP_raw.csv"
).dropna(axis=1, how="all")

geofetch["batchID"] = geofetch["sample_name"].str.split("patient_").str[1]
geofetch["batchID"] = geofetch["batchID"].str.split("_", n=2, expand=True)[2]
geofetch["batchID"] = geofetch["batchID"].apply(
    lambda x: "_".join(x.rsplit("_", 1)[:-1])
)
geofetch["batchID"] = geofetch["batchID"].apply(
    lambda x: "_".join(
        [part.upper() if index < 3 else part for index, part in enumerate(x.split("_"))]
    )
)

# %%
accessions = pd.merge(
    fetchngs,
    geofetch,
    how="left",
    on=["batchID"],
    validate="m:1",
)

# %%
# Merge accession numbers
adata.obs["obs_names"] = adata.obs_names
adata.obs = pd.merge(
    adata.obs,
    accessions.groupby("batchID").first(),
    how="left",
    left_on=["batchID_x"],
    right_on=["batchID"],
    validate="m:1",
).set_index("obs_names")
adata.obs_names.name = None

# %%
# Find duplicate columns and remove + rename without extension!
sorted(set([col[:-2] for col in adata.obs.columns if col.endswith(("_x", "_y"))]))

# %%
adata.obs.drop(
    columns=[
        "Age_y",
        "HistologicTypeSimple_y",
        "MLH1Status_y",
        "MMRStatus_y",
        "MMRMLH1Tumor_y",
        "MMRMLH1Tumor",
        "PatientBarcode_y",
        "PatientTypeID_y",
        "ProcessingMethod_y",
        "Sex_y",
        "SpecimenType_y",
        "TumorStage_y",
        "batchID_y",
        "sample_description_y",
        "sample_title_x",
        "Column1",
    ],
    inplace=True,
)
adata.obs.columns = [
    col.replace("_x", "").replace("_y", "") for col in adata.obs.columns
]

# %%
adata.obs = adata.obs.assign(
    NCBI_BioProject_accession="PRJNA738517",
    Synapse_project_accession="syn26127158",
    HTAN_project_accession="HTA1_272_483110110111",
)

# %%
# Get original sample/patient meta and remove cell level meta
sample_meta = (
    adata.obs[adata.obs.columns[adata.obs.nunique() <= adata.obs["batchID"].nunique()]]
    .groupby("batchID")
    .first()
    .reset_index()
)
sample_meta = sample_meta.rename(columns={"batchID": "sample_id"})

# %%
sample_meta.to_csv(f"{artifact_dir}/Pelka_2021-original_sample_meta.csv", index=False)

# %% [markdown]
# ## 3. Harmonize metadata to ref_meta_dict

# %%
adata.obs = adata.obs.assign(
    study_id="Pelka_2021_Cell",
    study_doi="10.1016/j.cell.2021.08.003",
    study_pmid="34450029",
    medical_condition="colorectal cancer",
    sample_tissue="colon",
    # Paper: did not sample tumor down to the invasive border
    tumor_source="core",
    tissue_cell_state="fresh",
    treatment_status_before_resection="naive",
    tissue_dissociation="enzymatic",  # |Miltenyi; Human Tumor Dissociation kit"
    cellranger_version="3.1.0",
    reference_genome="gencode.v28_lift37",
    matrix_type="raw counts",
    country="US",
)

# %%
adata.obs["SRA_sample_accession"] = adata.obs["secondary_sample_accession"]
adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]

adata.obs["sample_id"] = adata.obs["batchID"]
adata.obs["patient_id"] = adata.obs["PatientTypeID"].str.split("_").str[0]

# define the dictionary mapping
mapping = {
    "N": "normal",
    "T": "tumor",
}
adata.obs["sample_type"] = adata.obs["SPECIMEN_TYPE"].map(mapping)
adata.obs.loc[adata.obs["sample_type"] == "normal", "tumor_source"] = "normal"

adata.obs["sample_matched"] = ah.pp.add_matched_samples_column(adata.obs, "patient_id")

adata.obs["age"] = adata.obs["Age"].fillna(np.nan).astype("Int64")

# define the dictionary mapping
mapping = {
    "F": "female",
    "M": "male",
}
adata.obs["sex"] = adata.obs["Sex"].map(mapping)

adata.obs["replicate"] = adata.obs_names.str.split("_").str[1].str.strip("TN")
adata.obs["replicate"] = adata.obs["replicate"].replace(
    {
        "": "no_rep",
        "A": "rep1",
        "B": "rep2",
    }
)

# define the dictionary mapping
mapping = {
    "Normal colon": "normal colon",
    "Adenocarcinoma": "adenocarcinoma",
    "Adenocarcinoma;Mucinous": "mucinous adenocarcinoma",
    "Adenocarcinoma;Medullary (with solid growth pattern)": "medullary carcinoma",
    "Adenocarcinoma;Mucinous;Neuroendocrine": "mucinous adenocarcinoma",
    "Medullary": "medullary carcinoma",
}
adata.obs["histological_type"] = adata.obs["HistologicTypeSimple"].map(mapping)

# define the dictionary mapping
mapping = {
    "left": "distal colon",
    "right": "proximal colon",
}
adata.obs["anatomic_region"] = adata.obs["TissueSiteSimple"].map(mapping)

# define the dictionary mapping
mapping = {
    "LEFT colon (sigmoid colon)": "sigmoid colon",
    "RIGHT colon (transverse)": "transverse colon",
    "RIGHT colon (ascending colon)": "ascending colon",
    "RIGHT colon (cecum)": "cecum",
    "RIGHT colon (hepatic flexure)": "hepatic flexure",
    "RIGHT colon (hepatic flexure to cecum)": "cecum",
    "RIGHT colon": "nan",
    "RIGHT colon (Ascending colon)": "ascending colon",
    "RIGHT colon (Cecum/Ascending Colon)": "ascending colon",
    "RIGHT colon (Cecum/ascending colon)": "ascending colon",
    "LEFT colon (rectosigmoid)": "rectosigmoid junction",
    "LEFT colon (Sigmoid colon)": "sigmoid colon",
    "LEFT colon": "nan",
    "RIGHT colon (Cecum;  Ileocecal valve)": "cecum",
    "LEFT (Sigmoid)": "sigmoid colon",
    "RIGHT (CECUM)": "cecum",
    "LEFT (Rectosigmoid)": "rectosigmoid junction",
    "RIGHT (ascending)": "ascending colon",
    "LEFT (sigmoid)": "sigmoid colon",
    "RIGHT (hepatic flexure)": "hepatic flexure",
    "RIGHT (cecum)": "cecum",
    "LEFT (rectosigmoid)": "rectosigmoid junction",
    "RIGHT (transverse)": "transverse colon",
    "LEFT(descending)": "descending colon",
    "Right (ascending)": "ascending colon",
}
adata.obs["anatomic_location"] = adata.obs["TissueSite_detailed"].map(mapping)


# define the dictionary mapping
mapping = {
    "Low-grade (well differentiated to moderately differentiated)": "G1-G2",
    "Low Grade (Well differentiated or Moderately differentiated)": "G1-G2",
    "nan": "nan",
    "Low Grade (Moderately differentiated)": "G2",
    "Low Grade": "G1",
    "High Grade (Poorly differentiated or Undifferentiated), medullary features": "G3-G4",
    "Low Grade (Moderately differentiated).": "G2",
    "Low Grade (Moderately differentiated), focally mucin producing": "G2",
    "Low grade": "G1",
    "High Grade (with mucinous and signet ring cell features)": "G3",
    "High Grade (poorly diff with signet ring cell features)": "G3",
    "Low-grade (well differentiated to moderately        differentiated)": "G1-G2",
    "Low Grade (Moderately differentiated) and mucinous": "G2",
    "Grade 2 (Moderately differentiated. 50-95% gland formation).": "G2",
    "Grade 2 (Moderately differntiated)": "G2",
    "Grade 3 (Poorly differentiated. < 50% gland formation)": "G3",
    "Moderately differentiated": "G2",
    "Grade 3 (Poorly differentiated. < 50% gland formation).": "G3",
    "Grade 1 (Well differentiated. > 95% gland formation).": "G1",
    "Grade 1 (Well differentiated. > 95% gland formation)": "G1",
    "Grade 2 (Moderately differentiated. 50-95% gland formation)": "G2",
    "Grade 3 (but Moderately differentiated)": "G2",
    "Grade 2 (Moderately Differentiated)": "G2",
    "Grade 2 (Moderately differentiated. 50-95% gland formation.)": "G2",
    "Grade 2 (Moderately differentiated.)": "G2",
    "Grade 3 (Poorly differentiated. < 50% gland formation.)": "G3",
    "Grade 3 (Poorly differentiated. < 50% gland formation) and Grade 2 (Moderately differentiated. 50-95% gland formation)": "G2-G3",
    "Grade 3 (poorly differentiated)": "G3",
}
adata.obs["tumor_grade"] = adata.obs["HistologicGrade_detailed"].map(mapping)

# define the dictionary mapping
mapping = {
    "pT2": "T1",
    "pT3": "T3",
    "pT4a": "T4a",
    "nan": "nan",
    "pT4b": "T4b",
    "pT1": "T1",
    " pT3": "T3",
    "pt4a": "T4a",
}
adata.obs["tumor_stage_TNM_T"] = adata.obs[
    "Tumor Stage Raw (on resection specimen path report)"
].map(mapping)

# define the dictionary mapping
adata.obs["tumor_stage_TNM_N"] = adata.obs[
    "Node Status Raw (on resection specimen path report)"
].replace(
    {
        "nan": "nan",
        "N0 (i+)* (isolated tumor cells)": "N0",
        "pN1a": "N1a",
        "pN0": "N0",
    }
)

# define the dictionary mapping
mapping = {
    "not entered (Mx)": "Mx",
    "nan": "nan",
    "pM1c (Metastases the peritoneal surface, alone or with other site or organ metastases): Sites involved: Liver and peritoneum.\n": "M1c",
    "M1a": "M1a",
    "M1c": "M1c",
}
adata.obs["tumor_stage_TNM_M"] = adata.obs[
    "Metastasis stage (on resection specimen path report)"
].map(mapping)

adata.obs["tumor_stage_TNM"] = ah.pp.tnm_tumor_stage(adata.obs, "patient_id")

adata.obs["tumor_size"] = round(
    adata.obs["Size (Tumor Largest Dimension (cm))"].astype("float"), 2
)

# define the dictionary mapping
mapping = {
    "MMRp": "pMMR",
    "MMRd": "dMMR",
    "nan": "nan",
}
adata.obs["mismatch_repair_deficiency_status"] = adata.obs["MMRStatus"].map(mapping)

# define the dictionary mapping
mapping = {
    "MMRp": "MSS",
    "MMRd": "MSI-H",
    "nan": "nan",
}
adata.obs["microsatellite_status"] = adata.obs["MMRStatus"].map(mapping)

# define the dictionary mapping
mapping = {
    "MLH1NoMeth": "no_meth",
    "MLH1Meth": "meth",
    "nan": "nan",
}
adata.obs["MLH1_promoter_methylation_status_driver_mut"] = adata.obs["MLH1Status"].map(
    mapping
)

# Not sure what the technical difference is, but from the paper methods samples are either naive or CD45 enriched
# define the dictionary mapping
mapping = {
    "unsorted": "naive",
    "Live": "CD45+",
    "CD45+": "CD45+",
    "mixUnsortCD45": "CD45+",
}
adata.obs["enrichment_cell_types"] = adata.obs["ProcessingMethod"].map(mapping)

# define the dictionary mapping
mapping = {
    "v2": "10x 3' v2",
    "v3": "10x 3' v3",
}
adata.obs["platform"] = adata.obs["10x chemistry (verType)"].map(mapping)

# define the dictionary mapping
mapping = {
    "Hacohen": "Nir Hacohen lab",
    "CCPM_Hacohen": "Nir Hacohen lab",
    "Combined": "Aviv Regev lab",
    "CCPM_Regev": "Aviv Regev lab",  # "CCPM: Center for Cancer Precision Medicine"
}
adata.obs["tissue_processing_lab"] = adata.obs["TISSUE_PROCESSING_TEAM"].map(mapping)

# define the dictionary mapping
mapping = {
    "MGH": "Massachusetts General Hospital",
    "DFCI": "Dana-Farber Cancer Institute, Brigham and Women’s Hospital",
}
adata.obs["hospital_location"] = adata.obs["SOURCE_HOSPITAL"].map(mapping)

# define the dictionary mapping
mapping = {
    "Asian": "asian",
    "White": "white",
    "white": "white",
    "Black or African American": "black or african american",
    "Other": "other",
    "ND": "nan",  # ND: not determined
}
adata.obs["ethnicity"] = adata.obs["Race"].map(mapping)

adata.obs["cell_type_coarse_study"] = adata.obs["ClusterTop"]
adata.obs["cell_type_middle_study"] = adata.obs["ClusterMidway"]
adata.obs["cell_type_study"] = adata.obs["ClusterFull"]

# %%
# Paper methods: Additionally, a CD45-enriched sample was run for each specimen.
# I will use sample_ids that subset for both naive and enriched samples.
# I could not find out what "c1" and "c2" means in the provided column ["batchID"] for patients ["C111", "C125", "C126", "C133", "C134", "C135", "C136", "C137", "C138", "C139", "C140", "C162"]
# how is this different from the used "TA" and "TB" in patient "C130" for multiple tumor samples??
# In addition, for patients ["C107", "C115", "C116", "C118"] they make a difference between CD45pMACS:_1_, mixUnsortCD45MACS:_2_, and LiveMACS:_3_

#adata.obs["sample_id2"] = (
#    adata.obs["sample_id"].str.rsplit("_", n=5).str[0]
#    + "_"
#    + adata.obs["sample_id"].str.rsplit("_", n=5).str[3]
#    + "_"
#    + adata.obs["sample_id"].str.rsplit("_", n=5).str[4]
#)
#adata.obs["sample_id"] = (
#    adata.obs["sample_id2"]
#    + adata.obs["enrichment_cell_types"]
#    + adata.obs["platform"]
#    + adata.obs["PROCESSING_TYPE"]
#)

# Will just stick to the original ids!

# %%
# Dataset: Probably major technical variabilty comes from sequencing platform 10X v2 vs v3, as well as CD45 enrichment.
# For now I will disregard potential batch effects by hospital or lab!
adata.obs["dataset"] = (
    "Pelka_2021_10X"
    + adata.obs["platform"].str.split(" ").str[2]
    + "_"
    # regular expression with a lambda function to replace "+" to "pos" and "-" to "neg"
    + adata.obs["enrichment_cell_types"].str.replace(
        r"(\w+)([+-])",
        lambda x: f"{x.group(1)}_pos" if x.group(2) == "+" else f"{x.group(1)}_neg",
        regex=True,
    )
)

mapping = {
    "Pelka_2021_10Xv2_naive": "Pelka_2021_10Xv2",
    "Pelka_2021_10Xv3_naive": "Pelka_2021_10Xv3",
    "Pelka_2021_10Xv2_CD45_pos": "Pelka_2021_10Xv2_CD45Pos",
    "Pelka_2021_10Xv3_CD45_pos": "Pelka_2021_10Xv3_CD45Pos",
}
adata.obs["dataset"] = adata.obs["dataset"].map(mapping)

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)
# Add newly created driver_mutation columns
ref_meta_cols += [col for col in adata.obs.columns if "_status_driver_mut" in col]

# %%
# Add back potentially interesting metadata that does not fit the reference meta
ref_meta_cols += [
    "LymphNodeStatus",
    "MMR_IHC",
    "MMRMLH1Tumor",
    "Size Quantile",
    "DaysDiagnosisToLastAccess",
    "SurvivalStatus",
    "Time to Death (days)",
]

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
adata.obs_names = (
    adata.obs["dataset"]
    + "-"
    + adata.obs["SRA_sample_accession"]
    + "-"
    + adata.obs_names.str.split("_id-").str[1]
)

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

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
    _adata.obs = _adata.obs.dropna(
        axis=1, how="all"
    )  # Remove all nan columns, else write_h5ad will fail
    dataset = _adata.obs["dataset"].values[0]  # Get the dataset name from the subset
    _adata.write_h5ad(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
