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
#     display_name: Python [conda env:.conda-2024-scanpy]
#     language: python
#     name: conda-env-.conda-2024-scanpy-py
# ---

# %% [markdown]
# # Tidy core-atlas metadata

# %%
# ensure reproducibility -> set numba multithreaded mode
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sc_atlas_helpers as ah
import scanpy as sc
from nxfvars import nxfvars
from threadpoolctl import threadpool_limits

cpus = nxfvars.get("cpus", 2)
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

sc.settings.set_figure_params(
    dpi=200,
    facecolor="white",
    frameon=False,
)

# %% tags=["parameters"]
adata_path = nxfvars.get(
    "adata_path",
    "../../results/v1/artifacts/build_atlas/integrate_datasets/neighbors_leiden_umap_scanvi/leiden_paga_umap/merged-umap.h5ad",
)
artifact_dir = nxfvars.get(
    "artifact_dir",
    "/data/scratch/marteau/downloads/seed/",
)
marker_genes_path = nxfvars.get(
    "marker_genes_path",
    "/data/projects/2022/CRCA/tables/cell_type_markers.csv",
)

# %%
adata = sc.read_h5ad(adata_path)
marker_genes = pd.read_csv(marker_genes_path)

# %% [markdown]
# ## Tidy metadata

# %%
# Update missing MUI patient metadata (mostly driver mutation status)
update_meta = {
    "MUI_Innsbruck.P13": {"microsatellite_status": "MSS", "HER2_status": "neg"},
    "MUI_Innsbruck.P14": {"KRAS_status": "wt", "NRAS_status": "mut", "BRAF_status": "wt", "TP53_status": "mut", "APC_status": "mut"},
    "MUI_Innsbruck.P15": {"TP53_status": "mut", "APC_status": "mut"},
}

for patient, metadata in update_meta.items():
    for col in ["microsatellite_status", "HER2_status", "KRAS_status", "NRAS_status", "BRAF_status", "APC_status", "TP53_status"]:
        adata.obs[col] = adata.obs[col].astype(str)
    adata.obs.loc[adata.obs["patient_id"] == patient, metadata.keys()] = list(metadata.values())

for col in ["HER2_status", "KRAS_status", "NRAS_status", "BRAF_status", "APC_status", "TP53_status"]:
    adata.obs.loc[adata.obs["sample_type"] == "normal", col] = pd.NA

# %%
# Full missing metadata for MUI dataset only
#update_meta = {
#    "P13": {"microsatellite_status": "MSS", "HER2_status_driver_mut": "neg", "panTRK_status_driver_mut": "neg"},
#    "P14": {"KRAS_status_driver_mut": "wt", "NRAS_status_driver_mut": "mut", "BRAF_status_driver_mut": "wt", "panTRK_status_driver_mut": "neg", "TP53_status_driver_mut": "mut", "APC_status_driver_mut": "mut"},
#    "P15": {"panTRK_status_driver_mut": "neg", "TP53_status_driver_mut": "mut", "APC_status_driver_mut": "mut"},
#    "P16": {"panTRK_status_driver_mut": "pos"},
#}

#for patient, metadata in update_meta.items():
#    for col in ["microsatellite_status", "panTRK_status_driver_mut", "HER2_status_driver_mut", "KRAS_status_driver_mut", "NRAS_status_driver_mut", "BRAF_status_driver_mut", "APC_status_driver_mut", "TP53_status_driver_mut"]:
#        adata.obs[col] = adata.obs[col].astype(str)
#    adata.obs.loc[adata.obs["patient_id"] == patient, metadata.keys()] = list(metadata.values())

#for col in ["HER2_status_driver_mut", "KRAS_status_driver_mut", "NRAS_status_driver_mut", "BRAF_status_driver_mut", "APC_status_driver_mut", "TP53_status_driver_mut"]:
#    adata.obs.loc[adata.obs["sample_type"] == "normal", col] = pd.NA

# %%
adata.obs["tumor_grade"] = adata.obs["tumor_grade"].str.replace("G2~G3", "G2-G3").str.replace("G2.0", "G2").str.replace("G1.0", "G1")
adata.obs["tumor_dimensions"] = adata.obs["tumor_dimensions"].str.replace(" cm", "").str.replace("cmcm", "cm")
adata.obs["age"] = adata.obs["age"].str.replace("years", "").str.strip().str.replace(".0", "")

# %%
adata.obs["tumor_stage"] = adata.obs["tumor_stage_TNM"].str.replace('A|B|C', '', regex=True)
adata.obs["tumor_stage"] = pd.Categorical(adata.obs["tumor_stage"], categories=["I", "II", "III", "IV"])

# %%
adata.obs.loc[
    (adata.obs["sample_type"] == "blood"),
    "sample_tissue",
] = "blood"

# %%
adata.obs["anatomic_region"] = adata.obs["anatomic_region"].astype(str)
adata.obs.loc[
    (adata.obs["sample_type"] == "metastasis"),
    "anatomic_region",
] = "liver"

adata.obs.loc[
    (adata.obs["sample_type"] == "blood"),
    "anatomic_region",
] = "blood"

adata.obs.loc[
    (adata.obs["sample_type"] == "lymph node"),
    "anatomic_region",
] = "mesenteric lymph nodes"

# %%
adata.obs["anatomic_region"] = pd.Categorical(
    adata.obs["anatomic_region"],
    categories=["proximal colon", "distal colon", "blood", "liver", "mesenteric lymph nodes"],
    ordered=True,
)

# %%
adata.obs.loc[
    (adata.obs["sample_type"].isin(["blood", "metastasis", "lymph node"])),
    "anatomic_location",
] = pd.NA

adata.obs.loc[
    (adata.obs["anatomic_location"].isin(["mesenteric lymph nodes", "blood"])),
    "anatomic_location",
] = pd.NA

# %%
adata.obs["enrichment_cell_types"] = adata.obs["enrichment_cell_types"].str.replace("CD3+/CD4+/CD25mediate", "CD3+/CD4+/CD25int")
adata.obs["enrichment_cell_types"] = adata.obs["enrichment_cell_types"].str.replace("CD3+/CD4+/CD25medate", "CD3+/CD4+/CD25int")

# %%
adata.obs["study_doi"] = adata.obs["study_doi"].astype(str)
adata.obs["study_pmid"] = adata.obs["study_pmid"].astype(str)

adata.obs.loc[
    (adata.obs["dataset"] == "HTAPP_HTAN"),
    "study_doi",
] = "10.1016/j.cell.2021.08.003"

adata.obs.loc[
    (adata.obs["dataset"] == "HTAPP_HTAN"),
    "study_pmid",
] = "34450029"

adata.obs.loc[
    (adata.obs["dataset"] == "MUI_Innsbruck"),
    "study_doi",
] = "10.1101/2024.08.26.609563"

adata.obs.loc[
    (adata.obs["dataset"] == "MUI_Innsbruck"),
    "study_pmid",
] = "this study"

# %%
adata.obs["platform_fine"] = adata.obs["platform"]

mapping = {
    'BD Rhapsody': "BD Rhapsody",
    '10x': "10x 3p",
    "10x 3' v1": "10x 3p",
    "10x 3' v2": "10x 3p",
    "10x 3' v3": "10x 3p",
    "10x 5' v1": "10x 5p",
    "10x 5' v2": "10x 5p",
    "10x 5' v3": "10x 5p",
    'TruDrop': "TruDrop",
    'DNBelab C4': "DNBelab C4",
    'GEXSCOPE Singleron': "GEXSCOPE Singleron",
    'Smart-seq2': "Smart-seq2",
    'SMARTer (C1)': "SMARTer (C1)",
    'scTrio-seq2_Dong_protocol': "scTrio-seq2",
    'scTrio-seq2_Tang_protocol': "scTrio-seq2",
}
adata.obs["platform"] = adata.obs["platform_fine"].map(mapping)
adata.obs["platform"] = pd.Categorical(adata.obs["platform"], categories=["BD Rhapsody", "10x 3p", "10x 5p", "TruDrop", "DNBelab C4", "GEXSCOPE Singleron", "Smart-seq2", "SMARTer (C1)", "scTrio-seq2"])

# %%
# Recompute missing n_genes, n_counts
sc.pp.filter_cells(adata, min_counts=10)
sc.pp.filter_cells(adata, min_genes=10)

# %%
adata.obs = adata.obs[
    [
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
        "tumor_stage",
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
        "platform_fine",
        "cellranger_version",
        "reference_genome",
        "matrix_type",
        "enrichment_cell_types",
        "tissue_cell_state",
        "tissue_processing_lab",
        "hospital_location",
        "country",
        "NCBI_BioProject_accession",
        "SRA_sample_accession",
        "GEO_sample_accession",
        "ENA_sample_accession",
        "synapse_sample_accession",
        "study_id",
        "study_doi",
        "study_pmid",
        "original_obs_names",
        "cell_type_coarse_study",
        "cell_type_middle_study",
        "cell_type_study",
        "n_counts",
        "n_genes",
        "n_genes_by_counts",
        "total_counts",
        "pct_counts_in_top_20_genes",
        "pct_counts_mito",
        "S_score",
        "G2M_score",
        "phase",
        "SOLO_doublet_prob",
        "SOLO_singlet_prob",
        "SOLO_doublet_status",
        "cell_type_predicted",
        "cell_type_coarse",
        "cell_type_middle",
        "cell_type_fine",
    ]
]

# %%
adata.obs = adata.obs.fillna(pd.NA)

# %%
adata.write(f"{artifact_dir}/adata.h5ad", compression="lzf")

# %%
adata = adata[adata.obs["dataset"] != "UZH_Zurich_CD45Pos"].copy()
ah.pp.reprocess_adata_subset_scvi(adata, leiden_res=2, n_neighbors=15, use_rep="X_scANVI")

# %%
adata.obs["cell_type_fine"] = pd.Categorical(
    adata.obs["cell_type_fine"],
    categories=[
        "Monocyte classical", "Monocyte non-classical", "Macrophage", "Macrophage cycling",
        "Myeloid progenitor", "cDC1", "cDC2", "DC3", "pDC", "DC mature",
        "Granulocyte progenitor", "Neutrophil", "Eosinophil", "Mast cell",
        "Erythroid cell", "Platelet",
        "GC B cell", "B cell naive", "B cell activated naive", "B cell activated", "B cell memory",
        "Plasma IgA", "Plasma IgG", "Plasma IgM", "Plasmablast",
        "CD4", "Treg", "CD8", "NK", "ILC", "gamma-delta", "NKT", "CD4 naive", "CD8 naive",
        "CD4 stem-like", "CD8 stem-like", "CD4 cycling", "CD8 cycling", "NK cycling",
        "Crypt cell", "TA progenitor", "Colonocyte", "Colonocyte BEST4", "Goblet", "Tuft", "Enteroendocrine",
        "Cancer Colonocyte-like", "Cancer BEST4", "Cancer Goblet-like", "Cancer Crypt-like", "Cancer TA-like",
        "Cancer cell circulating",
        "Endothelial venous", "Endothelial arterial", "Endothelial lymphatic",
        "Fibroblast S1", "Fibroblast S2", "Fibroblast S3",
        "Pericyte", "Schwann cell",
        "Hepatocyte", "Fibroblastic reticular cell", "Epithelial reticular cell"
    ],
    ordered=True,
)

# %%
adata.obs["cell_type_middle"] = pd.Categorical(
    adata.obs["cell_type_middle"],
    categories=[
        "Monocyte", "Macrophage", "Dendritic cell",
        "Neutrophil", "Eosinophil", "Mast cell", "Erythroid cell", "Platelet",
        "CD4", "Treg", "CD8", "NK", "ILC", "gamma-delta", "NKT", "B cell", "Plasma cell",
        "Epithelial progenitor", "Epithelial cell", "Goblet", "Tuft", "Enteroendocrine",
        "Cancer cell", "Cancer cell circulating", "CRLM",
        "Endothelial cell", "Fibroblast", "Pericyte", "Schwann cell", "Hepatocyte"
    ],
    ordered=True,
)

# %%
adata.obs["cell_type_coarse"] = pd.Categorical(
    adata.obs["cell_type_coarse"],
    categories=["Myeloid cell", "Neutrophil", "Mast cell", "B cell", "Plasma cell", "T cell", "NK", "ILC", "Cancer cell", "Epithelial cell", "Hepatocyte", "Stromal cell", "Schwann cell"],
    ordered=True,
)

# %%
# Remove BD Rhapsody "Undetermined" and "Erythroid cell" (too few cells)
adata = adata[(adata.obs["sample_type"] != "Undetermined") & (adata.obs["cell_type_fine"] != "Erythroid cell")].copy()
adata.write(f"{artifact_dir}/core_atlas-adata.h5ad", compression="lzf")

# %% [markdown]
# ## Get patient/sample meta

# %%
# Save atlas sample meta
meta = adata.obs.groupby("sample_id", observed=False).first().sort_values(by=["dataset", "sample_id"]).reset_index()
meta[
    [
        "sample_id", "dataset", "medical_condition", "cancer_type",
        "sample_type", "tumor_source", "replicate", "sample_tissue", "anatomic_region", "anatomic_location",
        "tumor_stage_TNM", "tumor_stage_TNM_T", "tumor_stage_TNM_N", "tumor_stage_TNM_M", "tumor_size", "tumor_dimensions", "tumor_grade", "histological_type",
        "microsatellite_status", "mismatch_repair_deficiency_status", "MLH1_promoter_methylation_status",
        "MLH1_status", "KRAS_status", "BRAF_status", "APC_status", "TP53_status", "PIK3CA_status", "SMAD4_status", "NRAS_status", "MSH6_status", "FBXW7_status",
        "NOTCH1_status", "MSH2_status", "PMS2_status", "POLE_status", "ERBB2_status", "STK11_status", "HER2_status", "CTNNB1_status", "BRAS_status",
        "patient_id", "sex", "age", "ethnicity",
        "treatment_status_before_resection", "treatment_drug", "treatment_response", "RECIST",
        "platform", "cellranger_version", "reference_genome", "matrix_type", "enrichment_cell_types", "tissue_cell_state",
        "tissue_processing_lab", "hospital_location", "country",
        "NCBI_BioProject_accession", "SRA_sample_accession", "GEO_sample_accession", "ENA_sample_accession", "synapse_sample_accession", "study_id", "study_doi", "study_pmid",
    ]
]
meta.to_csv(f"{artifact_dir}/core_atlas-sample_metadata.csv", index=False)

# %%
# Save atlas patient meta
meta = adata.obs.groupby("patient_id", observed=False).first().sort_values(by=["dataset", "patient_id"]).reset_index()
meta = meta[
    [
        "patient_id", "dataset", "sex", "age", "ethnicity",
        "treatment_status_before_resection", "treatment_drug", "treatment_response", "RECIST", "medical_condition", "cancer_type",
        "microsatellite_status", "mismatch_repair_deficiency_status", "MLH1_promoter_methylation_status",
        "MLH1_status", "KRAS_status", "BRAF_status", "APC_status", "TP53_status", "PIK3CA_status", "SMAD4_status", "NRAS_status", "MSH6_status", "FBXW7_status",
        "NOTCH1_status", "MSH2_status", "PMS2_status", "POLE_status", "ERBB2_status", "STK11_status", "HER2_status", "CTNNB1_status", "BRAS_status",
        "platform", "tissue_processing_lab", "hospital_location", "country", "study_id", "study_doi", "study_pmid",
    ]
]
meta.to_csv(f"{artifact_dir}/core_atlas-patient_metadata.csv", index=False)

# %% [markdown]
# ## Final crc-atlas overview plots

# %%
adata.shape

# %%
# Core-atlas total expression values (non-zero)
adata.X.nnz

# %%
# Set display options to show all rows and columns
pd.set_option("display.max_rows", None)

# %%
adata.obs["cell_type_fine"].value_counts(dropna=False)

# %%
pd.reset_option("display.max_rows")

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    fig = sc.pl.umap(
        adata,
        color="cell_type_coarse",
        legend_fontoutline=1,
        legend_fontsize="xx-small",
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/core_atlas-umap_cell_type_coarse.png", bbox_inches="tight")

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    fig = sc.pl.umap(
        adata,
        color="cell_type_middle",
        legend_fontoutline=1,
        legend_fontsize="xx-small",
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/core_atlas-umap_cell_type_middle.png", bbox_inches="tight")

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    fig = sc.pl.umap(
        adata,
        color="cell_type_fine",
        legend_fontoutline=1,
        legend_fontsize="xx-small",
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/core_atlas-umap_cell_type_fine.png", bbox_inches="tight")

# %%
adata.obs["sample_type"] = pd.Categorical(
    adata.obs["sample_type"],
    categories=["tumor", "normal", "blood", "polyp",  "metastasis", "lymph node"],
    ordered=True,
)

# %%
adata.obs["sample_type"].value_counts(dropna=False)

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    fig = sc.pl.umap(
        adata,
        color="sample_type",
        legend_fontoutline=1,
        legend_fontsize="xx-small",
        return_fig=True,
        groups=["tumor", "polyp", "normal", "metastasis", "blood", "lymph node"],
    )
    fig.savefig(f"{artifact_dir}/core_atlas-umap_sample_type.png", bbox_inches="tight")

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    fig = sc.pl.umap(
        adata,
        color="medical_condition",
        legend_fontoutline=1,
        legend_fontsize="xx-small",
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/core_atlas-umap_medical_condition.png", bbox_inches="tight")

# %%
adata.obs["platform"].value_counts(dropna=False)

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    fig = sc.pl.umap(
        adata,
        color="platform",
        legend_fontoutline=1,
        legend_fontsize="xx-small",
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/core_atlas-umap_platform.png", bbox_inches="tight")

# %%
adata.obs["dataset"].value_counts(dropna=False)

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    fig = sc.pl.umap(
        adata,
        color="dataset",
        legend_fontoutline=1,
        legend_fontsize="xx-small",
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/core_atlas-umap_dataset.png", bbox_inches="tight")
