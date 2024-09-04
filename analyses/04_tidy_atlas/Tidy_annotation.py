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
# # Tidy scANVI integrated crc atlas cell annotation
As I already annotated the different sample types, will just use this and clean up.

# %%
# ensure reproducibility -> set numba multithreaded mode
import os
from collections import OrderedDict

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
adata_epi_path = nxfvars.get(
    "adata_epi_path",
    "../../results/v1/final/h5ads/crc_atlas-epithelial-adata.h5ad",
)
artifact_dir = nxfvars.get(
    "artifact_dir",
    "/data/scratch/marteau/downloads/seed/",
)

# %%
adata = sc.read_h5ad(adata_path)
adata_epi = sc.read_h5ad(adata_epi_path)

# %% [markdown]
# ## Remove doublets

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    fig = sc.pl.umap(
        adata,
        color="SOLO_doublet_status",
        legend_fontoutline=1,
        legend_fontsize="xx-small",
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/crc_atlas-SOLO_doublet_status.png", bbox_inches="tight")

# %%
# Remove high mito (mostly BD datasets)
adata.obs["is_droplet"] = np.where(
    (adata.obs["pct_counts_mito"] > 50),
    "empty droplet",
    "cell",
)

# %%
adata = adata[
    (adata.obs["SOLO_doublet_status"] == "singlet")
    & (adata.obs["is_droplet"] == "cell")
    & ~(adata.obs["dataset"].isin(["MUI_Innsbruck_AbSeq", "UZH_Zurich_healthy_blood"]))
].copy()

# %%
adata.obs = adata.obs[
    [
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
        "n_genes_by_counts",
        "log1p_n_genes_by_counts",
        "total_counts",
        "log1p_total_counts",
        "pct_counts_in_top_20_genes",
        "total_counts_mito",
        "log1p_total_counts_mito",
        "pct_counts_mito",
        "total_counts_ribo",
        "log1p_total_counts_ribo",
        "pct_counts_ribo",
        "total_counts_hb",
        "log1p_total_counts_hb",
        "pct_counts_hb",
        "S_score",
        "G2M_score",
        "phase",
        "SOLO_doublet_prob",
        "SOLO_singlet_prob",
        "SOLO_doublet_status",
        "leiden",
        "cell_type_fine_predicted",
        "cell_type_coarse",
        "cell_type_middle",
        "cell_type_fine",
    ]
]

adata.obs = adata.obs.rename(columns={"cell_type_fine_predicted": "cell_type_predicted"})

del adata.uns["SOLO_doublet_status_colors"]

# %%
ah.pp.reprocess_adata_subset_scvi(adata, leiden_res=2, n_neighbors=15, use_rep="X_scANVI")

# %% [markdown]
# ## Tidy cell annotation using leiden clustering

# %%
def _helper_ct(adata, leiden, min_counts):
    keep = adata[adata.obs["leiden"] == leiden].obs["cell_type_fine"].value_counts()[
        lambda x: x > min_counts
    ].index.tolist()

    keep = [label for label in keep if label != "unknown"]
    return keep

# %%
adata.obs["cell_type_fine"] = adata.obs["cell_type_fine"].astype(str)

# Leiden 0
adata.obs.loc[
    (adata.obs["leiden"] == "0")
    & (adata.obs["sample_type"].isin(["normal", "polyp"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "0", 100))),
    "cell_type_fine",
] = "Colonocyte"

adata.obs.loc[
    (adata.obs["leiden"] == "0")
    & ~(adata.obs["sample_type"].isin(["normal", "polyp", "blood"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "0", 100))),
    "cell_type_fine",
] = "Tumor Colonocyte-like"

# %%
# Leiden 1
adata.obs.loc[
    (adata.obs["leiden"] == "1")
    & (adata.obs["sample_type"].isin(["normal", "polyp"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "1", 100))),
    "cell_type_fine",
] = "TA progenitor"

adata.obs.loc[
    (adata.obs["leiden"] == "1")
    & ~(adata.obs["sample_type"].isin(["normal", "polyp", "blood"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "1", 100))),
    "cell_type_fine",
] = "Tumor TA-like"

# %%
# Leiden 2
adata.obs.loc[
    (adata.obs["leiden"] == "2")
    & (adata.obs["sample_type"].isin(["normal", "polyp"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "2", 10))),
    "cell_type_fine",
] = "Colonocyte"

adata.obs.loc[
    (adata.obs["leiden"] == "2")
    & ~(adata.obs["sample_type"].isin(["normal", "polyp", "blood"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "2", 10))),
    "cell_type_fine",
] = "Tumor Colonocyte-like"

# %%
# Leiden 3
adata.obs.loc[
    (adata.obs["leiden"] == "3")
    & (adata.obs["sample_type"].isin(["normal", "polyp"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "3", 30))),
    "cell_type_fine",
] = "Colonocyte"

adata.obs.loc[
    (adata.obs["leiden"] == "3")
    & ~(adata.obs["sample_type"].isin(["normal", "polyp", "blood"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "3", 30))),
    "cell_type_fine",
] = "Tumor Colonocyte-like"

# %%
# Leiden 4
adata.obs.loc[
    (adata.obs["leiden"] == "4")
    & (adata.obs["sample_type"].isin(["normal", "polyp"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "4", 30))),
    "cell_type_fine",
] = "Colonocyte BEST4"

adata.obs.loc[
    (adata.obs["leiden"] == "4")
    & ~(adata.obs["sample_type"].isin(["normal", "polyp", "blood"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "4", 30))),
    "cell_type_fine",
] = "Tumor BEST4"

# %%
# Leiden 5
adata.obs.loc[
    (adata.obs["leiden"] == "5")
    & (adata.obs["sample_type"].isin(["normal", "polyp"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "5", 10))),
    "cell_type_fine",
] = "Colonocyte"

adata.obs.loc[
    (adata.obs["leiden"] == "5")
    & ~(adata.obs["sample_type"].isin(["normal", "polyp", "blood"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "5", 10))),
    "cell_type_fine",
] = "Tumor Colonocyte-like"

# %%
# Leiden 6
adata.obs.loc[
    (adata.obs["leiden"] == "6")
    & (adata.obs["sample_type"].isin(["normal", "polyp"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "6", 100))),
    "cell_type_fine",
] = "Colonocyte"

adata.obs.loc[
    (adata.obs["leiden"] == "6")
    & ~(adata.obs["sample_type"].isin(["normal", "polyp", "blood"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "6", 100))),
    "cell_type_fine",
] = "Tumor Crypt-like"

# %%
# Leiden 7
adata.obs.loc[
    (adata.obs["leiden"] == "7")
    & (adata.obs["sample_type"].isin(["normal", "polyp"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "7", 50))),
    "cell_type_fine",
] = "Goblet"

adata.obs.loc[
    (adata.obs["leiden"] == "7")
    & ~(adata.obs["sample_type"].isin(["normal", "polyp", "blood"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "7", 50))),
    "cell_type_fine",
] = "Tumor Goblet-like"

# %%
# Leiden 8
adata.obs.loc[
    (adata.obs["leiden"] == "8")
    & (adata.obs["sample_type"].isin(["normal", "polyp"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "8", 50))),
    "cell_type_fine",
] = "Goblet"

adata.obs.loc[
    (adata.obs["leiden"] == "8")
    & ~(adata.obs["sample_type"].isin(["normal", "polyp", "blood"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "8", 50))),
    "cell_type_fine",
] = "Tumor Goblet-like"

# %%
# Leiden 9
adata.obs.loc[
    (adata.obs["leiden"] == "9")
    & ~(adata.obs["sample_type"].isin(["blood"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "9", 10))),
    "cell_type_fine",
] = "Tuft"

# %%
# Leiden 10
adata.obs.loc[
    (adata.obs["leiden"] == "10")
    & ~(adata.obs["sample_type"].isin(["blood"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "10", 10))),
    "cell_type_fine",
] = "Tuft"

# %%
# Leiden 11
adata.obs.loc[
    (adata.obs["leiden"] == "11")
    & (adata.obs["sample_type"].isin(["normal", "polyp"]))
    & ~(adata.obs["cell_type_fine"].isin([item for item in _helper_ct(adata, "11", 100) if item not in ["CD8", "CD4", "Plasma IgA", "Plasmablast", "Fibroblast S3"]])),
    "cell_type_fine",
] = "TA progenitor"

adata.obs.loc[
    (adata.obs["leiden"] == "11")
    & ~(adata.obs["sample_type"].isin(["normal", "polyp", "blood"]))
    & ~(adata.obs["cell_type_fine"].isin([item for item in _helper_ct(adata, "11", 100) if item not in ["CD8", "CD4", "Plasma IgA", "Plasmablast", "Fibroblast S3"]])),
    "cell_type_fine",
] = "Tumor Colonocyte-like"

# %%
# Leiden 12
adata.obs.loc[
    (adata.obs["leiden"] == "12")
    & ~(adata.obs["cell_type_fine"].isin([item for item in _helper_ct(adata, "12", 45) if item not in ["Plasma IgG", "Plasma IgA"]])),
    "cell_type_fine",
] = "CD8"

# %%
# Leiden 13
adata.obs.loc[
    (adata.obs["leiden"] == "13")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "13", 5))),
    "cell_type_fine",
] = "CD8 naive"

# %%
# Leiden 14
adata.obs.loc[
    (adata.obs["leiden"] == "14")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "14", 400))),
    "cell_type_fine",
] = "Neutrophil"

# %%
# Leiden 15
adata.obs.loc[
    (adata.obs["leiden"] == "15")
    & (adata.obs["sample_type"].isin(["normal", "polyp"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "15", 80))),
    "cell_type_fine",
] = "Colonocyte"

adata.obs.loc[
    (adata.obs["leiden"] == "15")
    & ~(adata.obs["sample_type"].isin(["normal", "polyp", "blood"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "15", 80))),
    "cell_type_fine",
] = "Tumor Colonocyte-like"

# %%
# Leiden 16
adata.obs.loc[
    (adata.obs["leiden"] == "16")
    & (adata.obs["sample_type"].isin(["normal", "polyp"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "16", 7))),
    "cell_type_fine",
] = "Colonocyte"

adata.obs.loc[
    (adata.obs["leiden"] == "16")
    & ~(adata.obs["sample_type"].isin(["normal", "polyp", "blood"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "16", 7))),
    "cell_type_fine",
] = "Tumor Colonocyte-like"

# %%
# Leiden 17
adata.obs.loc[
    (adata.obs["leiden"] == "17")
    & (adata.obs["sample_type"].isin(["normal", "polyp"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "17", 50))),
    "cell_type_fine",
] = "Colonocyte"

adata.obs.loc[
    (adata.obs["leiden"] == "17")
    & ~(adata.obs["sample_type"].isin(["normal", "polyp", "blood"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "17", 50))),
    "cell_type_fine",
] = "Tumor Colonocyte-like"

# %%
# Leiden 18
adata.obs.loc[
    (adata.obs["leiden"] == "18")
    & (adata.obs["sample_type"].isin(["normal", "polyp"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "18", 10))),
    "cell_type_fine",
] = "Colonocyte"

adata.obs.loc[
    (adata.obs["leiden"] == "18")
    & ~(adata.obs["sample_type"].isin(["normal", "polyp", "blood"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "18", 10))),
    "cell_type_fine",
] = "Tumor Colonocyte-like"

# %%
# Leiden 19
adata.obs.loc[
    (adata.obs["leiden"] == "19")
    & ~(adata.obs["sample_type"].isin(["blood"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "19", 10))),
    "cell_type_fine",
] = "Enteroendocrine"

# %%
# Leiden 20
adata.obs.loc[
    (adata.obs["leiden"] == "20")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "20", 85))),
    "cell_type_fine",
] = "CD4"

# %%
# Leiden 21
adata.obs.loc[
    (adata.obs["leiden"] == "21")
    & ~(adata.obs["cell_type_fine"].isin([item for item in _helper_ct(adata, "21", 40) if item not in ["Plasma IgA"]])),
    "cell_type_fine",
] = "Treg"

# %%
# Leiden 22
adata.obs.loc[
    (adata.obs["leiden"] == "22")
    & ~(adata.obs["cell_type_fine"].isin([item for item in _helper_ct(adata, "22", 8) if item not in ["Plasmablast", "Tuft", "Macrophage cycling", "Macrophage", "GC B cell"]])),
    "cell_type_fine",
] = "CD8"

# %%
# Leiden 23
adata.obs.loc[
    (adata.obs["leiden"] == "23")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "23", 1000))),
    "cell_type_fine",
] = "GC B cell"

# %%
# Leiden 24
adata.obs.loc[
    (adata.obs["leiden"] == "24")
    & ~(adata.obs["cell_type_fine"].isin([item for item in _helper_ct(adata, "24", 100) if item not in ["Mast cell", "GC B cell"]])),
    "cell_type_fine",
] = "cDC2"

# %%
# Leiden 25
adata.obs.loc[
    (adata.obs["leiden"] == "25")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "25", 105))),
    "cell_type_fine",
] = "CD4"

# %%
# Leiden 26
adata.obs.loc[
    (adata.obs["leiden"] == "26")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "26", 100))),
    "cell_type_fine",
] = "Plasma IgA"

# %%
# Leiden 27
adata.obs.loc[
    (adata.obs["leiden"] == "27")
    & ~(adata.obs["sample_type"].isin(["blood"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "27", 50))),
    "cell_type_fine",
] = "Enteroendocrine"

# %%
# Leiden 28
adata.obs.loc[
    (adata.obs["leiden"] == "28")
    & ~(adata.obs["sample_type"].isin(["blood"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "28", 50))),
    "cell_type_fine",
] = "Fibroblast S3"

# %%
# Leiden 29
adata.obs.loc[
    (adata.obs["leiden"] == "29")
    & ~(adata.obs["sample_type"].isin(["blood"]))
    & ~(adata.obs["cell_type_fine"].isin([item for item in _helper_ct(adata, "29", 20) if item not in ["Monocyte"]])),
    "cell_type_fine",
] = "Fibroblast S1"

# %%
# Leiden 30
adata.obs.loc[
    (adata.obs["leiden"] == "30")
    & ~(adata.obs["sample_type"].isin(["blood"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "30", 40))),
    "cell_type_fine",
] = "Fibroblast S3"

# %%
# Leiden 31
adata.obs.loc[
    (adata.obs["leiden"] == "31")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "31", 40))),
    "cell_type_fine",
] = "CD8"

# %%
# Leiden 32
adata.obs.loc[
    (adata.obs["leiden"] == "32")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "32", 500))),
    "cell_type_fine",
] = "Eosinophil"

# %%
# Leiden 33
adata.obs.loc[
    (adata.obs["leiden"] == "33")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "33", 400))),
    "cell_type_fine",
] = "Plasma IgA"

# %%
# Leiden 34
adata.obs.loc[
    (adata.obs["leiden"] == "34")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "34", 700))),
    "cell_type_fine",
] = "B cell activated"

# %%
# Leiden 35
adata.obs.loc[
    (adata.obs["leiden"] == "35")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "35", 20))),
    "cell_type_fine",
] = "B cell naive"

# %%
# Leiden 36
adata.obs.loc[
    (adata.obs["leiden"] == "36")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "36", 20))),
    "cell_type_fine",
] = "B cell memory"

# %%
# Leiden 37
adata.obs.loc[
    (adata.obs["leiden"] == "37")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "37", 480))),
    "cell_type_fine",
] = "B cell activated"

# %%
# Leiden 38
adata.obs.loc[
    (adata.obs["leiden"] == "38")
    & ~(adata.obs["cell_type_fine"].isin([item for item in _helper_ct(adata, "38", 600) if item not in ["cDC progenitor"]])),
    "cell_type_fine",
] = "B cell activated"

# %%
# Leiden 39
adata.obs.loc[
    (adata.obs["leiden"] == "39")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "39", 50))),
    "cell_type_fine",
] = "CD8"

# %%
# Leiden 40
adata.obs.loc[
    (adata.obs["leiden"] == "40")
    & ~(adata.obs["cell_type_fine"].isin([item for item in _helper_ct(adata, "40", 100) if item not in ["Plasma IgA", "Plasma IgG"]])),
    "cell_type_fine",
] = "Macrophage"

# %%
# Leiden 41
adata.obs.loc[
    (adata.obs["leiden"] == "41")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "41", 100))),
    "cell_type_fine",
] = "pDC"

# %%
# Leiden 42
adata.obs.loc[
    (adata.obs["leiden"] == "42")
    & ~(adata.obs["cell_type_fine"].isin([item for item in _helper_ct(adata, "42", 49) if item not in ["Mast cell", "Neutrophil", "Cancer cell circulating"]])),
    "cell_type_fine",
] = "Monocyte"

# %%
# Leiden 43
adata.obs.loc[
    (adata.obs["leiden"] == "43")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "43", 65))),
    "cell_type_fine",
] = "cDC progenitor"

# %%
# Leiden 44
adata.obs.loc[
    (adata.obs["leiden"] == "44")
    & ~(adata.obs["cell_type_fine"].isin([item for item in _helper_ct(adata, "44", 10) if item not in ["Mast cell", "Neutrophil"]])),
    "cell_type_fine",
] = "Macrophage"

# %%
# Leiden 45
adata.obs.loc[
    (adata.obs["leiden"] == "45")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "45", 10))),
    "cell_type_fine",
] = "Plasma IgA"

# %%
# Leiden 46
adata.obs.loc[
    (adata.obs["leiden"] == "46")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "46", 90))),
    "cell_type_fine",
] = "Plasma IgG"

# %%
# Leiden 47
adata.obs.loc[
    (adata.obs["leiden"] == "47")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "47", 400))),
    "cell_type_fine",
] = "Mast cell"

# %%
# Leiden 48
adata.obs.loc[
    (adata.obs["leiden"] == "48")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "48", 10))),
    "cell_type_fine",
] = "Plasma IgA"

# %%
# Leiden 49
adata.obs.loc[
    (adata.obs["leiden"] == "49")
    & ~(adata.obs["cell_type_fine"].isin([item for item in _helper_ct(adata, "49", 90) if item not in ["B cell activated", "Plasma IgA", "cDC progenitor"]])),
    "cell_type_fine",
] = "CD8"

# %%
# Leiden 50
adata.obs.loc[
    (adata.obs["leiden"] == "50")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "50", 50))),
    "cell_type_fine",
] = "Plasmablast"

# %%
# Leiden 51
adata.obs.loc[
    (adata.obs["leiden"] == "51")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "51", 1000))),
    "cell_type_fine",
] = "Plasmablast"

# %%
# Leiden 52
adata.obs.loc[
    (adata.obs["leiden"] == "52")
    & ~(adata.obs["sample_type"].isin(["blood"]))
    & ~(adata.obs["cell_type_fine"].isin([item for item in _helper_ct(adata, "52", 50) if item not in ["Monocyte"]])),
    "cell_type_fine",
] = "Fibroblast S3"

# %%
# Leiden 53
adata.obs.loc[
    (adata.obs["leiden"] == "53")
    & ~(adata.obs["sample_type"].isin(["blood"]))
    & ~(adata.obs["cell_type_fine"].isin([item for item in _helper_ct(adata, "53", 150) if item not in ["Tumor Colonocyte-like"]])),
    "cell_type_fine",
] = "Endothelial venous"

# %%
# Leiden 54
adata.obs.loc[
    (adata.obs["leiden"] == "54")
    & ~(adata.obs["sample_type"].isin(["blood"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "54", 12))),
    "cell_type_fine",
] = "Endothelial lymphatic"

# %%
# Leiden 55
adata.obs.loc[
    (adata.obs["leiden"] == "55")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "55", 10))),
    "cell_type_fine",
] = "CD4 naive"

# %%
# Leiden 56
adata.obs.loc[
    (adata.obs["leiden"] == "56")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "56", 8))),
    "cell_type_fine",
] = "CD4 naive"

# %%
# Leiden 57
adata.obs.loc[
    (adata.obs["leiden"] == "57")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "57", 30))),
    "cell_type_fine",
] = "CD4 stem-like"

# %%
# Leiden 58
adata.obs.loc[
    (adata.obs["leiden"] == "58")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "58", 10))),
    "cell_type_fine",
] = "CD8"

# %%
# Leiden 59
adata.obs.loc[
    (adata.obs["leiden"] == "59")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "59", 1000))),
    "cell_type_fine",
] = "NK"

# %%
# Leiden 60
adata.obs.loc[
    (adata.obs["leiden"] == "60")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "60", 14))),
    "cell_type_fine",
] = "CD8"

# %%
# Leiden 61
adata.obs.loc[
    (adata.obs["leiden"] == "61")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "61", 5))),
    "cell_type_fine",
] = "CD4"

# %%
# Leiden 62
adata.obs.loc[
    (adata.obs["leiden"] == "62")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "62", 100))),
    "cell_type_fine",
] = "Granulocyte progenitor"

# %%
# Leiden 63
adata.obs.loc[
    (adata.obs["leiden"] == "63")
    & ~(adata.obs["sample_type"].isin(["blood"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "63", 100))),
    "cell_type_fine",
] = "Pericyte"

# %%
# Leiden 64
adata.obs.loc[
    (adata.obs["leiden"] == "64")
    & ~(adata.obs["sample_type"].isin(["blood"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "64", 10))),
    "cell_type_fine",
] = "Fibroblast S2"

# %%
# Leiden 65
adata.obs.loc[
    (adata.obs["leiden"] == "65")
    & ~(adata.obs["sample_type"].isin(["blood"]))
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "65", 20))),
    "cell_type_fine",
] = "Schwann cell"

# %%
# Leiden 66
adata.obs.loc[
    (adata.obs["leiden"] == "66")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "66", 10))),
    "cell_type_fine",
] = "CD8"

# %%
# Leiden 67
adata.obs.loc[
    (adata.obs["leiden"] == "67")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "67", 6))),
    "cell_type_fine",
] = "CD8"

# %%
# Leiden 68
adata.obs.loc[
    (adata.obs["leiden"] == "68")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "68", 20))),
    "cell_type_fine",
] = "Mast cell"

# %%
# Leiden 70
adata.obs.loc[
    (adata.obs["leiden"] == "70")
    & ~(adata.obs["cell_type_fine"].isin(_helper_ct(adata, "70", 5))),
    "cell_type_fine",
] = "Plasma IgA"

# %%
# Set remaining "unknown" blood cells to "cell_type_predicted" label
adata.obs.loc[
    (adata.obs["cell_type_fine"] == "unknown")
    & (adata.obs["sample_type"] == "blood"),
    "cell_type_fine",
] = adata.obs["cell_type_predicted"]

# %%
adata.obs.loc[
    (adata.obs["cell_type_fine"].isin(["Neutrophil"]))
    & ~(adata.obs["dataset"].isin(adata[adata.obs["cell_type_fine"] == "Neutrophil"].obs["dataset"].value_counts()[lambda x: x > 20].index.tolist())),
    "cell_type_fine",
] = adata.obs["cell_type_predicted"]

# Set blood cell_type_predicted nonsense labels to either "Cancer cell circulating" for epithelial
adata.obs.loc[
    (adata.obs["cell_type_fine"].isin(["Colonocyte BEST4", "Crypt cell", "TA progenitor", "Goblet", "Tumor Goblet-like", "Tumor TA-like", "Tumor Crypt-like", "Colonocyte", "Tumor Colonocyte-like", "Tuft"]))
    & (adata.obs["sample_type"] == "blood"),
    "cell_type_fine",
] = "Cancer cell circulating"

# Platelet for the rest
adata.obs.loc[
    (adata.obs["cell_type_fine"].isin(["Schwann cell", "Pericyte", "Fibroblast S2", "CD4 stem-like", "Endothelial arterial", "Endothelial venous", "Fibroblast S3", "GC B cell"]))
    & (adata.obs["sample_type"] == "blood"),
    "cell_type_fine",
] = "Platelet"

# %%
adata.obs.loc[
    (adata.obs["cell_type_fine"].isin(["CD4 naive", "CD4 stem-like"]))
    & (adata.obs["sample_type"].isin(["tumor", "polyp", "normal"])),
    "cell_type_fine",
] = "CD4"

adata.obs.loc[
    (adata.obs["cell_type_fine"].isin(["CD8 naive"]))
    & (adata.obs["sample_type"].isin(["tumor", "polyp", "normal"])),
    "cell_type_fine",
] = "CD8"

# %%
adata.obs.loc[
    (adata.obs["cell_type_fine"].isin(["Eosinophil"]))
    & (adata.obs["sample_type"] == "metastasis"),
    "cell_type_fine",
] = "Neutrophil"

adata.obs.loc[
    (adata.obs["cell_type_fine"].isin(["B cell naive", "Plasmablast"]))
    & (adata.obs["sample_type"] == "metastasis"),
    "cell_type_fine",
] = "GC B cell"

adata.obs.loc[
    (adata.obs["cell_type_fine"].isin(["Fibroblast S1", "Enteroendocrine", "Schwann cell"]))
    & (adata.obs["sample_type"] == "metastasis"),
    "cell_type_fine",
] = "Fibroblast S2"

# %%
adata.obs.loc[
    (adata.obs["cell_type_fine"].isin(["Tumor BEST4", "Tuft"]))
    & (adata.obs["sample_type"] == "lymph node"),
    "cell_type_fine",
] = "Tumor Colonocyte-like"

adata.obs.loc[
    (adata.obs["cell_type_fine"].isin(["Tumor TA-like"]))
    & (adata.obs["sample_type"] == "lymph node"),
    "cell_type_fine",
] = "Tumor Crypt-like"

adata.obs.loc[
    (adata.obs["cell_type_fine"].isin(["CD8 naive"]))
    & (adata.obs["sample_type"] == "lymph node"),
    "cell_type_fine",
] = "CD8 stem-like"

adata.obs.loc[
    (adata.obs["cell_type_fine"].isin(["CD4 naive"]))
    & (adata.obs["sample_type"] == "lymph node"),
    "cell_type_fine",
] = "CD4 stem-like"

adata.obs.loc[
    (adata.obs["cell_type_fine"].isin(["Plasmablast"]))
    & (adata.obs["sample_type"] == "lymph node"),
    "cell_type_fine",
] = "GC B cell"

adata.obs.loc[
    (adata.obs["cell_type_fine"].isin(["Enteroendocrine", "Schwann cell", "Fibroblast S1"]))
    & (adata.obs["sample_type"] == "lymph node"),
    "cell_type_fine",
] = "Fibroblast S2"

adata.obs.loc[
    (adata.obs["cell_type_fine"].isin(["Neutrophil", "cDC progenitor", "Eosinophil"]))
    & (adata.obs["sample_type"] == "lymph node"),
    "cell_type_fine",
] = "Monocyte"

adata.obs.loc[
    (adata.obs["cell_type_fine"].isin(["Tumor Goblet-like", "Tumor Colonocyte-like", "Tumor Crypt-like"]))
    & (adata.obs["sample_type"] == "lymph node")
    & (adata.obs["medical_condition"] == "healthy"),
    "cell_type_fine",
] = "Epithelial reticular cell"

adata.obs.loc[
    (adata.obs["cell_type_fine"].isin(["Fibroblast S2"]))
    & (adata.obs["sample_type"] == "lymph node")
    & (adata.obs["medical_condition"] == "healthy"),
    "cell_type_fine",
] = "Fibroblast S3"

# %%
# Switch all non Eosinophil to Neutrophil -> Can only be from these 2 datasets!
adata.obs.loc[
    (adata.obs["cell_type_fine"].isin(["Eosinophil"]))
    & ~(adata.obs["dataset"].isin(["UZH_Zurich_CD45Pos", "MUI_Innsbruck"])),
    "cell_type_fine",
] = "Neutrophil"

# %%
adata.obs["tmp_gd"] = ah.pp.score_seeds(
    adata,
    {
        "gamma-delta": {
            "positive": [["TRGC2"]],
            "negative": [["TRBC1", "TRBC2", "CD8A", "CD8B"]],
        },
     },
    pos_cutoff=0.1,
    neg_cutoff=0.3,
)

# %%
adata.obs["cell_type_fine"] = adata.obs["cell_type_fine"].astype(str)
adata.obs.loc[
    (adata.obs["tmp_gd"] != "unknown")
& (adata.obs["cell_type_fine"].isin(["CD8"])),
    "cell_type_fine"
] = "gamma-delta"

# %% [markdown]
# ## Check epithelial annotation

# %%
epi_cell_marker = {
    "Crypt cell": {
        "positive": [["LGR5", "SMOC2"]],
        "negative": [[]],
    },
    "TA progenitor": {
        "positive": [["TOP2A", "UBE2C", "PCLAF"]],
        "negative": [[]],
    },
    "Goblet": {
        "positive": [["MUC2"], ["FCGBP"], ["ZG16"]],
        "negative": [["MKI67"]],
    },
    "Colonocyte BEST4": {
        "positive": [["BEST4"], ["OTOP2"], ["SPIB"]],
        "negative": [["MKI67"]],
    },
    "Colonocyte": {
        "positive": [["FABP1", "CEACAM7", "ANPEP", "SI"]],
        "negative": [
            ["BEST4", "OTOP2", "SPIB"],
            ["TOP2A", "UBE2C", "PCLAF", "HELLS", "TK1"],
            ["LGR5", "SMOC2"],
            ["MUC2", "FCGBP", "ZG16"],
            ["MKI67"],
        ],
    },
}

# %%
adata_epi.obs["cell_type_seed"] = ah.pp.score_seeds(
    adata_epi,
    epi_cell_marker,
    pos_cutoff=0.5,
    neg_cutoff=0.1,
)

# %%
cluster_annot = {
    "Goblet": [21, 20, 17, 32],
    "Tuft": [18],
    "Enteroendocrine": [26],
    "Colonocyte BEST4": [25],
    "Colonocyte": [15, 9, 6, 1, 8, 16, 7, 23, 11, 12, 27, 28, 30, 33, 34, 31],
    "TA progenitor": [0, 4, 29, 19, 2, 5, 14, 10],
    "Crypt cell": [3, 13, 24, 22],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}

adata_epi.obs["cell_type_fine2"] = (
    adata_epi.obs["leiden"].map(cluster_annot).fillna(adata_epi.obs["leiden"])
)

# %%
adata_epi.obs.loc[
    (adata_epi.obs["cell_type_seed"] != "unknown"),
    "cell_type_fine2",
] = adata_epi.obs["cell_type_seed"]

# %%
adata_sub = adata_epi[adata_epi.obs["leiden"].isin(["20",])].copy()
sc.tl.leiden(adata_sub, 0.5, flavor="igraph")

adata_sub.obs["leiden"] = adata_sub.obs["leiden"].astype(str)
cluster_annot = {
    "T": [2, 3, 6, 7],
    "N": [5, 1, 0, 4, 8],
}
cluster_annot = {str(vi): k for k, v in cluster_annot.items() for vi in v}
adata_sub.obs["leiden"] = adata_sub.obs["leiden"].map(cluster_annot).fillna(adata_sub.obs["leiden"])

ah.pp.integrate_back(adata_epi, adata_sub, variable="leiden")

# %%
mapping = {
    "Colonocyte": "Cancer Colonocyte-like",
    "Colonocyte BEST4": "Cancer BEST4",
    "Goblet": "Cancer Goblet-like",
    "Crypt cell": "Cancer Crypt-like",
    "TA progenitor": "Cancer TA-like",
}

adata_epi.obs.loc[
    ~adata_epi.obs["sample_type"].isin(["normal", "polyp"]),
    "cell_type_fine2"
] = adata_epi.obs.loc[
    ~adata_epi.obs["sample_type"].isin(["normal", "polyp"]),
    "cell_type_fine2"
].map(lambda x: mapping.get(x, x))

adata_epi.obs.loc[
    adata_epi.obs["sample_type"].isin(["polyp"])
& ~adata_epi.obs["leiden"].isin(["9", "25", "6", "0", "1", "8", "7", "23", "11", "21", "5", "27", "28", "30", "N"]),
    "cell_type_fine2"
] = adata_epi.obs.loc[
   adata_epi.obs["sample_type"].isin(["polyp"])
& ~adata_epi.obs["leiden"].isin(["9", "25", "6", "0", "1", "8", "7", "23", "11", "21", "5", "27", "28", "30", "N"]),
    "cell_type_fine2"
].map(lambda x: mapping.get(x, x))

# %%
adata_epi.obs.loc[
    (adata_epi.obs["sample_type"] == "polyp")
& (adata_epi.obs["cell_type_fine2"] == "Cancer BEST4"),
    "cell_type_fine2",
] = "Colonocyte BEST4"

adata_epi.obs.loc[
    (adata_epi.obs["sample_type"] == "lymph node")
& (adata_epi.obs["medical_condition"] == "healthy"),
    "cell_type_fine2",
] = "Epithelial reticular cell"

adata_epi.obs.loc[
    (adata_epi.obs["sample_type"] == "lymph node")
& (adata_epi.obs["cell_type_fine2"].isin(["Tuft", "Enteroendocrine"])),
    "cell_type_fine2",
] = "Cancer Crypt-like"

adata_epi.obs.loc[
    (adata_epi.obs["sample_type"] == "metastasis")
& (adata_epi.obs["cell_type_fine2"].isin(["Enteroendocrine"])),
    "cell_type_fine2",
] = "Cancer Colonocyte-like"

# %%
adata_epi.obs["cell_type_fine2"].value_counts()

# %%
sc.pl.umap(adata_epi, color="cell_type_fine2")

# %%
# Map back annotation
adata.obs["cell_type_fine"] = adata.obs["cell_type_fine"].astype(str)
adata.obs.loc[
    adata.obs_names.isin(adata_epi.obs_names),
    "cell_type_fine",
] = adata_epi.obs["cell_type_fine2"]

# %%
adata_mono = adata[adata.obs["cell_type_fine"] == "Monocyte"].copy()

# %%
# classical vs non-classical Monocyte
adata_mono.obs["cell_type_seed"] = ah.pp.score_seeds(
    adata_mono,
    {
    "Monocyte non-classical": {
        "positive": [["FCGR3A"]],
        "negative": [[]],
        }
    },
    unidentified_label="Monocyte classical",
    pos_cutoff=0.7,
    neg_cutoff=0,
)

# %%
adata.obs["cell_type_fine"] = adata.obs["cell_type_fine"].astype(str)
adata.obs.loc[
    adata.obs_names.isin(adata_mono.obs_names),
    "cell_type_fine",
] = adata_mono.obs["cell_type_seed"]

# %%
# NK cells
adata_t = adata[adata.obs["cell_type_fine"].isin(["CD8", "NK", "gamma-delta"])].copy()

# %%
t_cell_marker = {
    "NK": {
        "positive": [["NKG7", "FCGR3A", "TYROBP", "S1PR5"]],
        "negative": [["CD3E", "TRAC", "CD8A", "CD8B", "TRDV1", "TRGC1", "TRGC2"]],
    },
    "gamma-delta": {
        "positive": [["TRDC", "TRDV1", "TRGC1", "TRGC2"]],
        "negative": [["TRBC1", "TRBC2", "CD8A", "CD8B"]],
    },
}

# %%
adata_t.obs["cell_type_seed"] = ah.pp.score_seeds(
    adata_t,
    t_cell_marker,
    pos_cutoff=0.5,
    neg_cutoff=0.1,
)

# %%
adata_t.obs["tmp_gd"] = ah.pp.score_seeds(
    adata_t,
    {
    "gd": {
        "positive": [["TRDC", ]],
        "negative": [[]],
        }
    },
    pos_cutoff=0.8,
    neg_cutoff=0,
)

# %%
adata_t.obs["cell_type_seed"] = adata_t.obs["cell_type_seed"].astype(str)
adata_t.obs.loc[
    (adata_t.obs["tmp_gd"] == "gd")
 & (adata_t.obs["cell_type_seed"] == "NK"),
    "cell_type_seed",
] = "gamma-delta"

# %%
adata_t.obs["cell_type_seed"] = ah.pp.score_seeds(
    adata_t,
    t_cell_marker,
    pos_cutoff=0.5,
    neg_cutoff=0.1,
)

# %%
adata_t.obs["tmp_cd8"] = ah.pp.score_seeds(
    adata_t,
    {
    "CD8": {
        "positive": [["TRBC1", "TRBC2"]],
        "negative": [[]],
        }
    },
    pos_cutoff=2,
    neg_cutoff=0,
)

adata_t.obs.loc[
    (adata_t.obs["tmp_cd8"] == "CD8")
 & (adata_t.obs["cell_type_seed"] == "NK"),
    "cell_type_seed",
] = "CD8"

# %%
adata.obs["cell_type_fine"] = adata.obs["cell_type_fine"].astype(str)
adata_t = adata_t[adata_t.obs["cell_type_seed"] == "NK"].copy()
adata.obs.loc[
    adata.obs_names.isin(adata_t.obs_names),
    "cell_type_fine",
] = adata_t.obs["cell_type_seed"]

# %%
del adata.obs["tmp_gd"]
adata.obs["cell_type_fine"] = adata.obs["cell_type_fine"].replace({"NK cycling": "NK", "cDC progenitor": "Myeloid progenitor"})

# %% [markdown]
# ## Tidy coarse cell annotations

# %%
adata.obs["cell_type_fine"] = adata.obs["cell_type_fine"].str.replace("Tumor", "Cancer")

cluster_annot = {
    "Monocyte classical": "Monocyte",
    "Monocyte non-classical": "Monocyte",
    "Macrophage": "Macrophage",
    "Macrophage cycling": "Macrophage",
    "Myeloid progenitor": "Dendritic cell",
    "cDC1": "Dendritic cell",
    "cDC2": "Dendritic cell",
    "DC3": "Dendritic cell",
    "pDC": "Dendritic cell",
    "DC mature": "Dendritic cell",
    "Granulocyte progenitor": "Neutrophil",
    "Neutrophil": "Neutrophil",
    "Eosinophil": "Eosinophil",
    "Mast cell": "Mast cell",
    "Erythroid cell": "Erythroid cell",
    "Platelet": "Platelet",
    "CD4": "CD4",
    "Treg": "Treg",
    "CD8": "CD8",
    "NK": "NK",
    "ILC": "ILC",
    "gamma-delta": "gamma-delta",
    "NKT": "NKT",
    "CD4 naive": "CD4",
    "CD8 naive": "CD8",
    "CD4 stem-like": "CD4",
    "CD8 stem-like": "CD8",
    "CD4 cycling": "CD4",
    "CD8 cycling": "CD8",
    "GC B cell": "B cell",
    "B cell naive": "B cell",
    "B cell activated naive": "B cell",
    "B cell activated": "B cell",
    "B cell memory": "B cell",
    "Plasma IgA": "Plasma cell",
    "Plasma IgG": "Plasma cell",
    "Plasma IgM": "Plasma cell",
    "Plasmablast": "Plasma cell",
    "Crypt cell": "Epithelial progenitor",
    "TA progenitor": "Epithelial progenitor",
    "Colonocyte": "Epithelial cell",
    "Colonocyte BEST4": "Epithelial cell",
    "Goblet": "Goblet",
    "Tuft": "Tuft",
    "Enteroendocrine": "Enteroendocrine",
    "Cancer Colonocyte-like": "Cancer cell",
    "Cancer BEST4": "Cancer cell",
    "Cancer Goblet-like": "Cancer cell",
    "Cancer Crypt-like": "Cancer cell",
    "Cancer TA-like": "Cancer cell",
    "Cancer cell circulating": "Cancer cell circulating",
    "Endothelial venous": "Endothelial cell",
    "Endothelial arterial": "Endothelial cell",
    "Endothelial lymphatic": "Endothelial cell",
    "Fibroblast S1": "Fibroblast",
    "Fibroblast S2": "Fibroblast",
    "Fibroblast S3": "Fibroblast",
    "Pericyte": "Pericyte",
    "Schwann cell": "Schwann cell",
    "Hepatocyte": "Hepatocyte",
    "Fibroblastic reticular cell": "Fibroblast",
    "Epithelial reticular cell": "Epithelial cell",
}
adata.obs["cell_type_middle"] = (
    adata.obs["cell_type_fine"].map(cluster_annot)
)

# %%
adata.obs.loc[
    (adata.obs["cell_type_middle"] == "Cancer cell")
    & (adata.obs["sample_type"] == "metastasis"),
    "cell_type_middle",
] = "CRLM"

# %%
cluster_annot = {
    "Monocyte": "Myeloid cell",
    "Macrophage": "Myeloid cell",
    "Dendritic cell": "Myeloid cell",
    "Neutrophil": "Neutrophil",
    "Eosinophil": "Neutrophil",
    "Mast cell": "Mast cell",
    "Erythroid cell": "Myeloid cell",
    "Platelet": "Myeloid cell",
    "CD4": "T cell",
    "Treg": "T cell",
    "CD8": "T cell",
    "NK": "NK",
    "ILC": "ILC",
    "gamma-delta": "T cell",
    "NKT": "T cell",
    "B cell": "B cell",
    "Plasma cell": "Plasma cell",
    "Epithelial progenitor": "Epithelial cell",
    "Epithelial cell": "Epithelial cell",
    "Goblet": "Epithelial cell",
    "Tuft": "Epithelial cell",
    "Enteroendocrine": "Epithelial cell",
    "Cancer cell": "Cancer cell",
    "Cancer cell circulating": "Cancer cell",
    "CRLM": "Cancer cell",
    "Endothelial cell": "Stromal cell",
    "Fibroblast": "Stromal cell",
    "Pericyte": "Stromal cell",
    "Schwann cell": "Schwann cell",
    "Hepatocyte": "Hepatocyte",
}
adata.obs["cell_type_coarse"] = (
    adata.obs["cell_type_middle"].map(cluster_annot)
)

# %%
adata.write(f"{artifact_dir}/adata.h5ad", compression="lzf")
