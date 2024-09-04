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
# # Export core-atlas

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

# %% tags=["parameters"]
adata_path = nxfvars.get(
    "adata_path",
    "../../results/v1/artifacts/build_atlas/annotate_cell_types/tidy_atlas/artifacts/core_atlas-adata.h5ad",
)
artifact_dir = nxfvars.get("artifact_dir", "/data/scratch/marteau/downloads/seed/")

# %%
adata = sc.read_h5ad(adata_path)

# %% [markdown]
# ## Add cellxgene metadata

# %%
# switch var_names to ensembl ids
adata.var_names = adata.var["ensembl"]
adata.var_names.name = None

# %%
adata.obs["donor_id"] = adata.obs["patient_id"]

adata.obs["is_primary_data"] = False
adata.obs.loc[
    (adata.obs["dataset"] == "MUI_Innsbruck"),
    "is_primary_data",
] = True

# %%
adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
adata.obs["organism"] = "Homo sapiens"
adata.obs["suspension_type"] = "cell"
adata.obs["tissue_type"] = "tissue"

# %%
mapping = {
    "tumor": "MONDO:0005575",
    "normal": "PATO:0000461",
    "blood": "UBERON:0000178",
    "polyp": "MONDO:0021400",
    "metastasis": "MONDO:0041448",
    "lymph node": "MONDO:0041448",
}
adata.obs["disease_ontology_term_id"] = adata.obs["sample_type"].map(mapping)

# %%
mapping = {
    "MONDO:0041448": "metastasis from malignant tumor of colon",
    "PATO:0000461": "normal",
    "MONDO:0005575": "colorectal cancer",
    "UBERON:0000178": "blood",
    "MONDO:0021400": "polyp of colon",
}
adata.obs["disease"] = adata.obs["disease_ontology_term_id"].astype(str).map(mapping)

# %%
mapping = {
    "BD Rhapsody": "EFO:0700003",
    "10x": "EFO:0008995",
    "10x 3' v1": "EFO:0009901",
    "10x 3' v2": "EFO:0009899",
    "10x 3' v3": "EFO:0009922",
    "10x 5' v1": "EFO:0011025",
    "10x 5' v2": "EFO:0009900",
    "10x 5' v3": "EFO:0022605",
    "TruDrop": "EFO:0700010",
    "DNBelab C4": "unknown",
    "GEXSCOPE Singleron": "EFO:0700011",
    "Smart-seq2": "EFO:0008442",
    "SMARTer (C1)": "EFO:0010058",
    "scTrio-seq2_Dong_protocol": "EFO:0010007",
    "scTrio-seq2_Tang_protocol": "EFO:0010007",
}

adata.obs["assay_ontology_term_id"] = adata.obs["platform_fine"].map(mapping)

# %%
mapping = {
    "EFO:0700003": "BD Rhapsody Whole Transcriptome Analysis",
    "EFO:0008995": "10x technology",
    "EFO:0009901": "10x 3' v1",
    "EFO:0009899": "10x 3' v2",
    "EFO:0009922": "10x 3' v3",
    "EFO:0011025": "10x 5' v1",
    "EFO:0009900": "10x 5' v2",
    "EFO:0022605": "10x 5' v3",
    "EFO:0700010": "TruDrop",
    "EFO:0700011": "GEXSCOPE technology",
    "EFO:0008442": "Smart-seq2 protocol",
    "EFO:0010058": "Fluidigm C1-based library preparation",
    "EFO:0010007": "scTrio-seq",
    "unknown": "unknown",
}

adata.obs["assay"] = adata.obs["assay_ontology_term_id"].map(mapping)

# %%
mapping = {
    "colon": "UBERON:0001155",
    "liver": "UBERON:0002107",
    "blood": "UBERON:0000178",
    "mesenteric lymph nodes": "UBERON:0002509",
    "nan": "unknown",
}
adata.obs["tissue_ontology_term_id"] = adata.obs["sample_tissue"].astype(str).map(mapping)

# %%
mapping = {
    "UBERON:0002509": "mesenteric lymph node",
    "UBERON:0001155": "colon",
    "UBERON:0002107": "liver",
    "UBERON:0000178": "blood",
}
adata.obs["tissue"] = adata.obs["tissue_ontology_term_id"].astype(str).map(mapping)

# %%
mapping = {
    "female": "PATO:0000383",
    "male": "PATO:0000384",
    "nan": "unknown",
}
adata.obs["sex_ontology_term_id"] = adata.obs["sex"].astype(str).map(mapping)

# %%
mapping = {
    "asian": "HANCESTRO:0008",
    "black or african american": "HANCESTRO:0568",
    "caucasian": "HANCESTRO:0005",
    "hispanic": "HANCESTRO:0612",
    "white": "HANCESTRO:0005",
    "other": "unknown",
    "nan": "unknown",
}
adata.obs["self_reported_ethnicity_ontology_term_id"] = adata.obs["ethnicity"].astype(str).map(mapping)
adata.obs = adata.obs.rename(columns={"ethnicity": "self_reported_ethnicity"})

# %%
cluster_annot = {
    "Monocyte classical": "CL:0000860",
    "Monocyte non-classical": "CL:0000875",
    "Macrophage": "CL:0000235",
    "Macrophage cycling": "CL:0000235",
    "Myeloid progenitor": "CL:0000557",
    "cDC1": "CL:0002399",
    "cDC2": "CL:0002399",
    "DC3": "CL:0002399",
    "pDC": "CL:0000784",
    "DC mature": "CL:0000841",
    "Granulocyte progenitor": "CL:0000557",
    "Neutrophil": "CL:0000775",
    "Eosinophil": "CL:0000771",
    "Mast cell": "CL:0000097",
    "Platelet": "CL:0000233",
    "CD4": "CL:0000624",
    "Treg": "CL:0000815",
    "CD8": "CL:0000625",
    "NK": "CL:0000623",
    "ILC": "CL:0001065",
    "gamma-delta": "CL:0000798",
    "NKT": "CL:0000814",
    "CD4 naive": "CL:0000895",
    "CD8 naive": "CL:0000900",
    "CD4 stem-like": "CL:0000624",
    "CD8 stem-like": "CL:0000625",
    "CD4 cycling": "CL:0000624",
    "CD8 cycling": "CL:0000625",
    "GC B cell": "CL:0000844",
    "B cell naive": "CL:0000788",
    "B cell activated naive": "CL:0000785",
    "B cell activated": "CL:0000785",
    "B cell memory": "CL:0000787",
    "Plasma IgA": "CL:0000987",
    "Plasma IgG": "CL:0000985",
    "Plasma IgM": "CL:0000986",
    "Plasmablast": "CL:0000980",
    "Crypt cell": "CL:0009043",
    "TA progenitor": "CL:0009011",
    "Colonocyte": "CL:1000347",
    "Colonocyte BEST4": "CL:4030026",
    "Goblet": "CL:0009039",
    "Tuft": "CL:0009041",
    "Enteroendocrine": "CL:0009042",
    "Cancer Colonocyte-like": "CL:0001064",
    "Cancer BEST4": "CL:0001064",
    "Cancer Goblet-like": "CL:0001064",
    "Cancer Crypt-like": "CL:0001064",
    "Cancer TA-like": "CL:0001064",
    "Cancer cell circulating": "CL:0001064",
    "Endothelial venous": "CL:0002543",
    "Endothelial arterial": "CL:1000413",
    "Endothelial lymphatic": "CL:0002138",
    "Fibroblast S1": "CL:0000057",
    "Fibroblast S2": "CL:0000057",
    "Fibroblast S3": "CL:0000057",
    "Pericyte": "CL:0000669",
    "Schwann cell": "CL:0002573",
    "Hepatocyte": "CL:0000182",
    "Fibroblastic reticular cell": "CL:0000432",
    "Epithelial reticular cell": "CL:0000432",
}
adata.obs["cell_type_ontology_term_id"] = adata.obs["cell_type_fine"].map(cluster_annot)

# %%
cluster_annot = {
    "CL:0001064": "malignant cell",
    "CL:1000347": "enterocyte of colon",
    "CL:4030026": "BEST4+ intestinal epithelial cell, human",
    "CL:0009043": "intestinal crypt stem cell of colon",
    "CL:0009011": "transit amplifying cell of colon",
    "CL:0009039": "colon goblet cell",
    "CL:0009042": "enteroendocrine cell of colon",
    "CL:0009041": "tuft cell of colon",
    "CL:0000844": "germinal center B cell",
    "CL:0000788": "naive B cell",
    "CL:0000785": "mature B cell",
    "CL:0000787": "memory B cell",
    "CL:0000980": "plasmablast",
    "CL:0000987": "IgA plasma cell",
    "CL:0000985": "IgG plasma cell",
    "CL:0000986": "IgM plasma cell",
    "CL:0000625": "CD8-positive, alpha-beta T cell",
    "CL:0000624": "CD4-positive, alpha-beta T cell",
    "CL:0000815": "regulatory T cell",
    "CL:0000895": "naive thymus-derived CD4-positive, alpha-beta T cell",
    "CL:0000900": "naive thymus-derived CD8-positive, alpha-beta T cell",
    "CL:0000798": "gamma-delta T cell",
    "CL:0000623": "natural killer cell",
    "CL:0001065": "innate lymphoid cell",
    "CL:0000814": "mature NK T cell",
    "CL:0000860": "classical monocyte",
    "CL:0000875": "non-classical monocyte",
    "CL:0000235": "macrophage",
    "CL:0000557": "granulocyte monocyte progenitor cell",
    "CL:0000775": "neutrophil",
    "CL:0000771": "eosinophil",
    "CL:0000097": "mast cell",
    "CL:0000233": "platelet",
    "CL:0002399": "CD1c-positive myeloid dendritic cell",
    "CL:0000841": "mature conventional dendritic cell",
    "CL:0000784": "plasmacytoid dendritic cell",
    "CL:0000669": "pericyte",
    "CL:0000057": "fibroblast",
    "CL:1000413": "endothelial cell of artery",
    "CL:0002543": "vein endothelial cell",
    "CL:0002138": "endothelial cell of lymphatic vessel",
    "CL:0002573": "Schwann cell",
    "CL:0000432": "reticular cell",
    "CL:0000182": "hepatocyte",
}
adata.obs["cell_type"] = adata.obs["cell_type_ontology_term_id"].map(cluster_annot)

# %%
cluster_annot = {
    "20": "HsapDv:0000114",
    "22": "HsapDv:0000116",
    "26": "HsapDv:0000120",
    "29": "HsapDv:0000123",
    "30": "HsapDv:0000124",
    "31": "HsapDv:0000125",
    "33": "HsapDv:0000127",
    "34": "HsapDv:0000128",
    "35": "HsapDv:0000129",
    "36": "HsapDv:0000130",
    "37": "HsapDv:0000131",
    "38": "HsapDv:0000132",
    "40": "HsapDv:0000134",
    "41": "HsapDv:0000135",
    "42": "HsapDv:0000136",
    "43": "HsapDv:0000137",
    "44": "HsapDv:0000138",
    "45": "HsapDv:0000139",
    "46": "HsapDv:0000140",
    "47": "HsapDv:0000141",
    "48": "HsapDv:0000142",
    "49": "HsapDv:0000143",
    "50": "HsapDv:0000144",
    "51": "HsapDv:0000145",
    "52": "HsapDv:0000146",
    "53": "HsapDv:0000147",
    "54": "HsapDv:0000148",
    "55": "HsapDv:0000149",
    "56": "HsapDv:0000150",
    "57": "HsapDv:0000151",
    "58": "HsapDv:0000152",
    "59": "HsapDv:0000153",
    "60": "HsapDv:0000154",
    "61": "HsapDv:0000155",
    "62": "HsapDv:0000156",
    "63": "HsapDv:0000157",
    "64": "HsapDv:0000158",
    "65": "HsapDv:0000159",
    "66": "HsapDv:0000160",
    "67": "HsapDv:0000161",
    "68": "HsapDv:0000162",
    "69": "HsapDv:0000163",
    "70": "HsapDv:0000164",
    "71": "HsapDv:0000165",
    "72": "HsapDv:0000166",
    "73": "HsapDv:0000167",
    "74": "HsapDv:0000168",
    "75": "HsapDv:0000169",
    "76": "HsapDv:0000170",
    "77": "HsapDv:0000171",
    "78": "HsapDv:0000172",
    "79": "HsapDv:0000173",
    "80": "HsapDv:0000206",
    "81": "HsapDv:0000207",
    "82": "HsapDv:0000208",
    "83": "HsapDv:0000209",
    "84": "HsapDv:0000210",
    "85": "HsapDv:0000211",
    "86": "HsapDv:0000212",
    "89": "HsapDv_0000215",
    "90": "HsapDv_0000216",
    "91": "HsapDv_0000217",
    "20-25": "HsapDv:0000116",
    "30-40": "HsapDv:0000129",
    "40-44": "HsapDv:0000136",
    "50-54": "HsapDv:0000146",
    "50-55": "HsapDv:0000146",
    "50-60": "HsapDv:0000149",
    "51-55": "HsapDv:0000147",
    "55-59": "HsapDv:0000151",
    "55-60": "HsapDv:0000151",
    "57–71": "HsapDv:0000158",
    "60-64": "HsapDv:0000156",
    "61-65": "HsapDv:0000157",
    "65-70": "HsapDv:0000161",
    "66-70": "HsapDv:0000162",
    "70-74": "HsapDv:0000166",
    "70-80": "HsapDv:0000169",
    "71-75": "HsapDv:0000167",
    "76-80": "HsapDv:0000172",
    "80-85": "HsapDv:0000208",
    "86-90": "HsapDv_0000214",
    "median: 50 [47-74]": "unknown",
    "median: 63 (35-87)": "unknown",
    "nan": "unknown",
}
adata.obs["development_stage_ontology_term_id"] = adata.obs["age"].astype(str).map(cluster_annot)

# %%
cluster_annot = {
    "20": "20-year-old human stage",
    "22": "22-year-old human stage",
    "26": "26-year-old human stage",
    "29": "29-year-old human stage",
    "30": "30-year-old human stage",
    "31": "31-year-old human stage",
    "33": "33-year-old human stage",
    "34": "34-year-old human stage",
    "35": "35-year-old human stage",
    "36": "36-year-old human stage",
    "37": "37-year-old human stage",
    "38": "38-year-old human stage",
    "40": "40-year-old human stage",
    "41": "41-year-old human stage",
    "42": "42-year-old human stage",
    "43": "43-year-old human stage",
    "44": "44-year-old human stage",
    "45": "45-year-old human stage",
    "46": "46-year-old human stage",
    "47": "47-year-old human stage",
    "48": "48-year-old human stage",
    "49": "49-year-old human stage",
    "50": "50-year-old human stage",
    "51": "51-year-old human stage",
    "52": "52-year-old human stage",
    "53": "53-year-old human stage",
    "54": "54-year-old human stage",
    "55": "55-year-old human stage",
    "56": "56-year-old human stage",
    "57": "57-year-old human stage",
    "58": "58-year-old human stage",
    "59": "59-year-old human stage",
    "60": "60-year-old human stage",
    "61": "61-year-old human stage",
    "62": "62-year-old human stage",
    "63": "63-year-old human stage",
    "64": "64-year-old human stage",
    "65": "65-year-old human stage",
    "66": "66-year-old human stage",
    "67": "67-year-old human stage",
    "68": "68-year-old human stage",
    "69": "69-year-old human stage",
    "70": "70-year-old human stage",
    "71": "71-year-old human stage",
    "72": "72-year-old human stage",
    "73": "73-year-old human stage",
    "74": "74-year-old human stage",
    "75": "75-year-old human stage",
    "76": "76-year-old human stage",
    "77": "77-year-old human stage",
    "78": "78-year-old human stage",
    "79": "79-year-old human stage",
    "80": "80-year-old human stage",
    "81": "81-year-old human stage",
    "82": "82-year-old human stage",
    "83": "83-year-old human stage",
    "84": "84-year-old human stage",
    "85": "85-year-old human stage",
    "86": "86-year-old human stage",
    "89": "89-year-old human stage",
    "90": "90-year-old human stage",
    "91": "91-year-old human stage",
    "20-25": "22-year-old human stage",
    "30-40": "35-year-old human stage",
    "40-44": "42-year-old human stage",
    "50-54": "52-year-old human stage",
    "50-55": "52-year-old human stage",
    "50-60": "55-year-old human stage",
    "51-55": "53-year-old human stage",
    "55-59": "57-year-old human stage",
    "55-60": "57-year-old human stage",
    "57–71": "64-year-old human stage",
    "60-64": "62-year-old human stage",
    "61-65": "63-year-old human stage",
    "65-70": "67-year-old human stage",
    "66-70": "68-year-old human stage",
    "70-74": "72-year-old human stage",
    "70-80": "75-year-old human stage",
    "71-75": "73-year-old human stage",
    "76-80": "78-year-old human stage",
    "80-85": "82-year-old human stage",
    "86-90": "88-year-old human stage",
    "median: 50 [47-74]": "unknown",
    "median: 63 (35-87)": "unknown",
    "nan": "unknown",
}
adata.obs["development_stage"] = adata.obs["age"].astype(str).map(cluster_annot)

# %% [markdown]
# ## Export atlas

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
        "is_primary_data",
        "suspension_type",
        "tissue_type",
        "donor_id",
        "disease",
        "disease_ontology_term_id",
        "assay",
        "assay_ontology_term_id",
        "tissue",
        "tissue_ontology_term_id",
        "sex_ontology_term_id",
        "self_reported_ethnicity",
        "self_reported_ethnicity_ontology_term_id",
        "organism",
        "organism_ontology_term_id",
        "development_stage",
        "development_stage_ontology_term_id",
        "cell_type",
        "cell_type_ontology_term_id",
    ]
]

# %%
adata.obs["age"] = adata.obs["age"].str.replace("–", "-")

# %%
# Make reduce adata size
for col in ["connectivities", "distances"]:
    del adata.obsp[col]

# %%
for col in ["leiden", "leiden_sizes", "neighbors", "paga", "umap"]:
    del adata.uns[col]

# %%
for col in ["mito", "ribo", "hb", "log1p_mean_counts", "pct_dropout_by_counts", "log1p_total_counts"]:
    del adata.var[col]

# %%
adata.layers["counts"] = adata.X
adata.X = adata.layers["log1p_norm"]
del adata.layers["log1p_norm"]

# %%
adata.write(f"{artifact_dir}/core_atlas-adata.h5ad", compression="lzf")
