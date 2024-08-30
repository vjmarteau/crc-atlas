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
# # Dataloader: Che_2021_Cell_Discov

# %%
import itertools
import os
import re

import anndata
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
d10_1038_s41421_021_00312_y = f"{databases_path}/d10_1038_s41421_021_00312_y"

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
SraRunTable = pd.read_csv(f"{d10_1038_s41421_021_00312_y}/PRJNA725335/SraRunTable.txt")

geofetch = pd.read_csv(
    f"{d10_1038_s41421_021_00312_y}/GSE178318/GSE178318_PEP/GSE178318_PEP_raw.csv"
).dropna(axis=1, how="all")

# Originaly xls sup. file with clinical annotation
patient_meta = pd.read_excel(
    f"{d10_1038_s41421_021_00312_y}/metadata/41421_2021_312_MOESM2_ESM.xlsx", header=2
)

# %%
geofetch["Library Name"] = geofetch["sample_title"].str.split(":").str[1]

geofetch["Library Name"] = geofetch["Library Name"].replace(
    {" primary CRC": "_CRC", " liver metastases": "_LM", " PBMC": "_PBMC"}, regex=True
)


meta = pd.merge(
    SraRunTable,
    geofetch,
    how="left",
    on=["Library Name"],
    validate="m:1",
).rename(columns={"Run": "SRA_sample_accession"})

meta["Patient ID"] = meta["Library Name"].str.split("_").str[0]

meta = pd.merge(
    meta,
    patient_meta,
    how="left",
    on=["Patient ID"],
    validate="m:1",
)


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
    itertools.repeat(d10_1038_s41421_021_00312_y),
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
    f"{artifact_dir}/Che_2021-original_sample_meta.csv",
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

# %% [markdown] tags=[]
# ## 3. Compile original meta from study supplements

# %%
adata.obs = adata.obs.assign(
    study_id="Che_2021_Cell_Discov",
    study_doi="10.1038/s41421-021-00312-y",
    study_pmid="34489408",
    tissue_processing_lab="Jian-Ming Li lab",
    dataset="Che_2021",
    medical_condition="colorectal cancer",
    NCBI_BioProject_accession="PRJNA725335",
    # Paper methods: Single cells were isolated from fresh tumor tissues without surface marker pre-selection.
    enrichment_cell_types="naive",
    # Paper methods: For each lesion, we collected the tissue in the core of the tumor after the surgery
    tumor_source="core",
    matrix_type="raw counts",
    platform="10x 3' v2",
    cellranger_version="7.1.0",
    reference_genome="gencode.v44",
    # Paper methods:  All patients were classified as MSS with invasive adenocarcinomas and late-stage (IV) disease
    microsatellite_status="MSS",
    hospital_location="Sun Yat-Sen Memorial Hospital, Sun Yat-Sen University, Guangzhou, Guangdong",
    country="China",
    tissue_cell_state="fresh",
    tissue_dissociation="enzymatic"# |Miltenyi; MACS® Tumor Dissociation kit",
)

adata.obs["GEO_sample_accession"] = adata.obs["sample_geo_accession"]

adata.obs["patient_id"] = adata.obs["Patient ID"]
adata.obs["sample_id"] = adata.obs["Library Name"]

mapping = {
    "CRC": "tumor",
    "LM": "metastasis",
    "PBMC": "blood",
}
adata.obs["sample_type"] = adata.obs["Library Name"].str.split("_").str[1].map(mapping)

mapping = {
    "CRC": "colon",
    "LM": "liver",
    "PBMC": "blood",
}
adata.obs["sample_tissue"] = (
    adata.obs["Library Name"].str.split("_").str[1].map(mapping)
)

mapping = {
    "Female": "female",
    "Male": "male",
}
adata.obs["sex"] = adata.obs["Gender"].map(mapping)
adata.obs["age"] = adata.obs["Age (years)"].fillna(np.nan).astype("Int64")


mapping = {
    "Ascending colon": "ascending colon",
    "upper rectum": "rectum",
    "Sigmoid colon": "sigmoid colon",
    "Transverse colon": "transverse colon",
}
adata.obs["anatomic_location"] = adata.obs["Site of Primary Tumor"].map(mapping)

mapping = {
    "Ascending colon": "proximal colon",
    "upper rectum": "distal colon",
    "Sigmoid colon": "distal colon",
    "Transverse colon": "proximal colon",
}
adata.obs["anatomic_region"] = adata.obs["Site of Primary Tumor"].map(
    mapping
)

mapping = {
    "Adenocarcinoma": "adenocarcinoma",
}
adata.obs["histological_type"] = adata.obs["Histology"].map(mapping)

mapping = {
    "Moderately differentiated": "G2",
    "Moderately to poorly differentiated": "G2-G3",
}
adata.obs["tumor_grade"] = adata.obs["Tumor Grade"].map(mapping)

adata.obs["mismatch_repair_deficiency_status"] = adata.obs["pMMR/dMMR"]

# Paper methods: CAPEOX: capecitabine plus oxaliplatin; FOLFOX-Bev: 5FU, oxaliplatin, leucovorin with bevacizumab
# Surgery was performed ~1 month after the last chemotherapy treatment in all three patients.
mapping = {
    "4 cycles of CAPEOX": "treated",
    " -": "naive",
    "3 cycles of CAPEOX": "treated",
    "8 cycles of FOLFOX-Bev": "treated",
}
adata.obs["treatment_status_before_resection"] = adata.obs["Preoperative Chemotherapy"].map(mapping)

adata.obs["treatment_drug"] = adata.obs["Preoperative Chemotherapy"].replace(
    " -", "nan"
)

# Paper methods: All three patients in this study responded well to preoperative chemotherapy (PC) with a signiﬁcant tumor shrinkage
mapping = {
    "4 cycles of CAPEOX": "responder",
    " -": "nan",
    "3 cycles of CAPEOX": "responder",
    "8 cycles of FOLFOX-Bev": "responder",
}
adata.obs["treatment_response"] = adata.obs["Preoperative Chemotherapy"].map(mapping)

mapping = {
    "4 cycles of CAPEOX": "CR",
    " -": "nan",
    "3 cycles of CAPEOX": "CR",
    "8 cycles of FOLFOX-Bev": "CR",
}
adata.obs["RECIST"] = adata.obs["Preoperative Chemotherapy"].map(mapping)

adata.obs["Primary tumor invaded to adjacent organ"] = adata.obs[
    "Primary tumor invaded to adjacent organ"
].astype(str)

# %%
adata.obs["tumor_stage_TNM_T"] = (
    adata.obs["TNM"]
    .astype(str)
    .apply(
        lambda x: re.search(r"(T[0-4][a-c]?)", x).group(1)
        if re.search(r"(T[0-4][a-c]?)", x)
        else None
    )
)
adata.obs["tumor_stage_TNM_N"] = (
    adata.obs["TNM"]
    .astype(str)
    .apply(
        lambda x: re.search(r"(N[0-4][a-c]?)", x).group(1)
        if re.search(r"(N[0-4][a-c]?)", x)
        else None
    )
)
adata.obs["tumor_stage_TNM_M"] = (
    adata.obs["TNM"]
    .astype(str)
    .apply(
        lambda x: re.search(r"((M[0-1]|Mx)[a-c]?)", x).group(1)
        if re.search(r"((M[0-1]|Mx)[a-c]?)", x)
        else None
    )
)

# Matches "Stage AJCC" column from original meta
adata.obs["tumor_stage_TNM"] = ah.pp.tnm_tumor_stage(adata.obs, "patient_id")

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
# Add back potentially interesting metadata that does not fit the reference meta
ref_meta_cols += [
    "Primary tumor invaded to adjacent organ",
    "Number of LM",
    "Maximum Diameter of LM",
    "surgical margin of CRC",
    "surgical margin of LM",
]

# %%
# Subset adata for columns in the reference meta
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

# %%
# Save unfiltered raw counts
dataset = adata.obs["dataset"].unique()[0]
adata.write(f"{artifact_dir}/{dataset}-adata.h5ad", compression="lzf")
