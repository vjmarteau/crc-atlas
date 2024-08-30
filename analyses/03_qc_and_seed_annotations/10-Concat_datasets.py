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
#     display_name: Python [conda env:.conda-2024-crc-atlas-scanpy]
#     language: python
#     name: conda-env-.conda-2024-crc-atlas-scanpy-py
# ---

# %% [markdown]
# # After scAR denoising concat all available datasets

# %%
# ensure reproducibility -> set numba multithreaded mode
import os
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from nxfvars import nxfvars
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits
from tqdm.contrib.concurrent import process_map

# %% tags=["parameters"]
artifact_dir = nxfvars.get("artifact_dir", "/data/scratch/marteau/downloads/crc-atlas/")
datasets_path = nxfvars.get(
    "datasets_path", "../../results/v1/artifacts/build_atlas/load_datasets/01-merge/"
)
adata_var_gtf = nxfvars.get(
    "adata_var_gtf",
    "../../results/v1/artifacts/build_atlas/load_datasets/harmonize_datasets/artifacts/adata_var_gtf.csv",
)
cpus = nxfvars.get("cpus", 12)

# %%
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

# %%
files = [str(x) for x in Path(datasets_path).glob("*.h5ad")]
files = [x for x in files if not isinstance(x, float)]
files.sort()

dataset = [Path(file).stem.split("-adata")[0] for file in files]
dataset = [Path(file).stem.split("-denoised_adata")[0] for file in dataset]

# %%
len(dataset)

# %%
dataset

# %%
# Concat adatas from specified directory
adatas = process_map(sc.read_h5ad, files, max_workers=cpus)

for adata in adatas:
    # module scAR sets denoised counts to .X -> need to set layer "counts" for non denoised datasets
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X
    else:
        # Apply same basic filter threshold on denoised counts
        sc.pp.filter_cells(adata, min_counts=10)
        sc.pp.filter_cells(adata, min_genes=10)

# Convert adatas list to named adatas dict
adatas = {dataset: adata for dataset, adata in zip(dataset, adatas)}

# %%
# After denosing keep only healthy blood patients - us: unsorted; s: sorted
adatas["MUI_Innsbruck_AbSeq"] = adatas["MUI_Innsbruck_AbSeq"][
    adatas["MUI_Innsbruck_AbSeq"].obs["Sample_Name"].isin(["healthy_us", "healthy_s"])
].copy()

# %%
adatas["UZH_Zurich_healthy_blood"].obs["sample_type"] = (
    adatas["UZH_Zurich_healthy_blood"].obs["sample_type"].astype(str)
)
adatas["UZH_Zurich_healthy_blood"].obs.loc[
    adatas["UZH_Zurich_healthy_blood"].obs["sample_type"] == "Undetermined",
    "sample_type",
] = "Undetermined_blood"

# %%
# Normalize Smart-seq2 matrix by gene length (from ensembl)
# see: https://docs.scvi-tools.org/en/stable/tutorials/notebooks/scrna/tabula_muris.html#normalize-smartseq2-matrix-by-gene-length

# Remove not known gene length from dataset
adatas["Zhang_2018_CD3Pos"] = adatas["Zhang_2018_CD3Pos"][:, ~adatas["Zhang_2018_CD3Pos"].var['Length'].isna()].copy()

adatas["Zhang_2018_CD3Pos"].X = adatas["Zhang_2018_CD3Pos"].X / adatas["Zhang_2018_CD3Pos"].var["Length"].astype(float).values * np.median(adatas["Zhang_2018_CD3Pos"].var["Length"].astype(float).values)
adatas["Zhang_2018_CD3Pos"].X = np.rint(adatas["Zhang_2018_CD3Pos"].X)
adatas["Zhang_2018_CD3Pos"].X = csr_matrix(adatas["Zhang_2018_CD3Pos"].X)

# %%
# Remove not known gene length from dataset
adatas["Zhang_2020_CD45Pos_CD45Neg"] = adatas["Zhang_2020_CD45Pos_CD45Neg"][:, ~adatas["Zhang_2020_CD45Pos_CD45Neg"].var['Length'].isna()].copy()

adatas["Zhang_2020_CD45Pos_CD45Neg"].X = adatas["Zhang_2020_CD45Pos_CD45Neg"].X / adatas["Zhang_2020_CD45Pos_CD45Neg"].var["Length"].astype(float).values * np.median(adatas["Zhang_2020_CD45Pos_CD45Neg"].var["Length"].astype(float).values)
adatas["Zhang_2020_CD45Pos_CD45Neg"].X = np.rint(adatas["Zhang_2020_CD45Pos_CD45Neg"].X)
adatas["Zhang_2020_CD45Pos_CD45Neg"].X = csr_matrix(adatas["Zhang_2020_CD45Pos_CD45Neg"].X)

# %%
# "Microwell-seq" (full-length protocol)
# Remove not known gene length from dataset
adatas["Han_2020"] = adatas["Han_2020"][:, ~adatas["Han_2020"].var['Length'].isna()].copy()

adatas["Han_2020"].X = adatas["Han_2020"].X / adatas["Han_2020"].var["Length"].astype(float).values * np.median(adatas["Han_2020"].var["Length"].astype(float).values)
adatas["Han_2020"].X = np.rint(adatas["Han_2020"].X)
adatas["Han_2020"].X = csr_matrix(adatas["Han_2020"].X)

# %%
# Remove normal liver samples -> not matched correctly in dataloader
adatas["Wu_2022_CD45Pos"] = adatas["Wu_2022_CD45Pos"][
    ~adatas["Wu_2022_CD45Pos"].obs["sample_id"].str.contains("Liver_P")
].copy()

# %%
adata = anndata.concat(adatas, index_unique=".", join="outer", fill_value=0)
adata.obs_names = adata.obs_names.str.rsplit(".", n=1).str[0]

# %%
# Not sure if this works to free the mem, else might need to look at python garbage collector
adatas = "del"

# %%
# Replace values for medical_condition samples with "healthy" in "treatment_status_before_resection"
adata.obs["treatment_status_before_resection"] = adata.obs["treatment_status_before_resection"].astype(str)
adata.obs.loc[
    adata.obs["medical_condition"] == "healthy", "treatment_status_before_resection"
] = "naive"

# %%
# Replace "normal" with "normal colon" in polyp datasets
adata.obs.loc[adata.obs["histological_type"] == "normal", "histological_type"] = "normal colon"

# %%
adata.obs["anatomic_location"] = adata.obs["anatomic_location"].astype(str)
adata.obs["anatomic_location"] = adata.obs["anatomic_location"].str.replace("lymph node", "mesenteric lymph nodes")

adata.obs["anatomic_location"] = pd.Categorical(
    adata.obs["anatomic_location"],
    categories=[
        "cecum",
        "ascending colon",
        "hepatic flexure",
        "transverse colon",
        "splenic flexure",
        "descending colon",
        "sigmoid colon",
        "rectosigmoid junction",
        "rectum",
        "mesenteric lymph nodes",
        "blood",
    ],
)

# %%
adata.obs["sample_tissue"] = pd.Categorical(
    adata.obs["sample_tissue"],
    categories=[
        "colon",
        "liver",
        "blood",
        "mesenteric lymph nodes",
        "Multiplet",
        "Undetermined",
    ],
)

# %%
adata.obs["sample_type"] = pd.Categorical(
    adata.obs["sample_type"],
    categories=[
        "tumor",
        "normal",
        "polyp",
        "metastasis",
        "blood",
        "lymph node",
        "Multiplet",
        "Undetermined",
        "Undetermined_blood",
    ],
)

# %%
# Change platform "bd rhapsody" to capital
adata.obs["platform"] = adata.obs["platform"].astype(str)
adata.obs.loc[adata.obs["platform"] == "bd rhapsody", "platform"] = "BD Rhapsody"

adata.obs["platform"] = pd.Categorical(
    adata.obs["platform"],
    categories=[
        "BD Rhapsody",
        "10x",
        "10x 3' v1",
        "10x 3' v2",
        "10x 3' v3",
        "10x 5' v1",
        "10x 5' v2",
        "10x 5' v3",
        "TruDrop",
        "GEXSCOPE Singleron",
        "DNBelab C4",
        "Microwell-seq",
        "SMARTer (C1)",
        "Smart-seq2",
        "scTrio-seq2_Dong_protocol",
        "scTrio-seq2_Tang_protocol",
    ],
)

adata.obs["tumor_stage_TNM"] = pd.Categorical(
    adata.obs["tumor_stage_TNM"],
    categories=[
        "I",
        "II",
        "IIA",
        "IIB",
        "IIC",
        "III",
        "IIIA",
        "IIIB",
        "IIIC",
        "IV",
        "IVA",
        "IVB",
        "IVC",
    ],
)

# %%
mapping = {
    "CR": "CR: complete response",
    "CR: complete response": "CR: complete response",
    "SD: stable disease": "SD: stable disease",
    "PR: partial response": "PR: partial response",
    "NR": "SD: stable disease",
    "PD: progressive disease": "PD: progressive disease",
    "NE: inevaluable": "NE: inevaluable",
    "PR": "PR: partial response",
    "nan": np.nan,
}
adata.obs["RECIST"] = adata.obs["RECIST"].map(mapping)

adata.obs["RECIST"] = pd.Categorical(
    adata.obs["RECIST"],
    categories=[
        "CR: complete response",
        "PR: partial response",
        "PD: progressive disease",
        "SD: stable disease",
        "NE: inevaluable",
    ],
)

# %%
# Add column "cancer_type" with COAD/READ/normal/polyp -> only for primary tumor samples
adata.obs["anatomic_region"] = adata.obs["anatomic_region"].astype(str)
adata.obs["anatomic_region"] = adata.obs["anatomic_region"].str.replace("lymph node", "mesenteric lymph nodes")

adata.obs["cancer_type"] = adata.obs.apply(lambda row: np.nan if pd.isna(row['anatomic_location']) else ("READ" if row['anatomic_location'] == 'rectum' else "COAD"), axis=1)
# proximal colon -> Not rectum
adata.obs["cancer_type"] = adata.obs["cancer_type"].fillna(value=adata.obs.apply(lambda row: "COAD" if row['anatomic_region'] == 'proximal colon' else row['cancer_type'], axis=1))

adata.obs.loc[adata.obs["medical_condition"] == "colorectal polyp", "cancer_type"] = adata.obs["cancer_type"] + " polyp"
adata.obs.loc[adata.obs["medical_condition"] == "healthy", "cancer_type"] = "normal"
adata.obs.loc[adata.obs["sample_type"] == "normal", "cancer_type"] = "normal"
adata.obs.loc[adata.obs["sample_type"].isin(["lymph node", "metastasis", "blood"]), "cancer_type"] = np.nan

# %%
adata.obs["sample_id"] = (
    adata.obs["dataset"].astype(str) + "." + adata.obs["sample_id"].astype(str)
)

# %%
# switch batch column to dataset/patient_id for cell_cycle scoring
adata.obs["batch"] = adata.obs["dataset"].astype(str)
adata.obs.loc[adata.obs["dataset"] == "Chen_2024", "batch"] = (
    adata.obs["dataset"].astype(str) + "_" + adata.obs["patient_id"].astype(str)
)

# %%
# Reorder columns
adata.obs = adata.obs[
    [
        "batch",
        "study_id",
        "dataset",
        "medical_condition",
        "cancer_type",
        "sample_id",
        "sample_type",
        "tumor_source",
        "replicate",
        "sample_tissue",
        "anatomic_region",
        "anatomic_location",
        "tumor_stage_TNM",
        "tumor_stage_TNM_T",
        "tumor_stage_TNM_N",
        "tumor_stage_TNM_M",
        "tumor_size",
        "tumor_dimensions",
        "tumor_grade",
        "histological_type",
        "microsatellite_status",
        "mismatch_repair_deficiency_status",
        "MLH1_promoter_methylation_status",
        "MLH1_status",
        "KRAS_status",
        "BRAF_status",
        "APC_status",
        "TP53_status",
        "PIK3CA_status",
        "SMAD4_status",
        "NRAS_status",
        "MSH6_status",
        "FBXW7_status",
        "NOTCH1_status",
        "MSH2_status",
        "PMS2_status",
        "POLE_status",
        "ERBB2_status",
        "STK11_status",
        "HER2_status",
        "CTNNB1_status",
        "BRAS_status",
        "patient_id",
        "sex",
        "age",
        "ethnicity",
        "treatment_status_before_resection",
        "treatment_drug",
        "treatment_response",
        "RECIST",
        "platform",
        "cellranger_version",
        "reference_genome",
        "matrix_type",
        "enrichment_cell_types",
        "tissue_cell_state",
        "tissue_dissociation",
        "tissue_processing_lab",
        "hospital_location",
        "country",
        "NCBI_BioProject_accession",
        "SRA_sample_accession",
        "GEO_sample_accession",
        "ENA_sample_accession",
        "synapse_sample_accession",
        "study_doi",
        "study_pmid",
        "original_obs_names",
        "cell_type_coarse_study",
        "cell_type_middle_study",
        "cell_type_study",
    ]
]

# %%
# Save harmonized sample meta
meta = adata.obs.groupby("sample_id", observed=False).first().sort_values(by=["dataset", "sample_id"])
meta = meta.loc[meta["sample_type"].isin(['tumor', 'normal', 'blood', 'polyp', 'metastasis', 'lymph node'])]
meta.to_csv(f"{artifact_dir}/atlas_sample_metadata.csv")

# %%
# map back gtf info to .var
adata.var["ensembl"] = adata.var_names
annotation = pd.read_csv(adata_var_gtf)

for column in [col for col in annotation.columns if col != "ensembl"]:
    adata.var[column] = adata.var["ensembl"].map(
        annotation.set_index("ensembl")[column]
    )

adata.var["Dataset_25pct_Overlap"] = adata.var["Dataset_25pct_Overlap"].astype(str)
adata.var['Dataset_25pct_Overlap'] = adata.var['Dataset_25pct_Overlap'].map({'True': True, 'False': False}).astype(bool)

# %%
# Save annotation gtf for gene order in next process
annotation.to_csv(f"{artifact_dir}/adata_var_gtf.csv", index=False)

# %%
# Switch back var_names to symbols
adata.var_names = adata.var["var_names"]
adata.var_names.name = None

# %%
gene_index = annotation[annotation["var_names"].isin(adata.var_names)]["var_names"].values
adata = adata[:, gene_index]

# %%
adata.var = adata.var[
    [
        "var_names",
        "ensembl",
        "Geneid",
        "GeneSymbol",
        "Chromosome",
        "Start",
        "End",
        "Class",
        "Strand",
        "Length",
        "Version",
        "Dataset_25pct_Overlap",
        "n_cells",
        "n_counts",
    ]
]

# %%
adata.write_h5ad(f"{artifact_dir}/merged-adata.h5ad", compression="lzf")
