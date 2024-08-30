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
# # Harmonize datasets

# %%
import os
from pathlib import Path

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
datasets_path = nxfvars.get("datasets_path", "../../results/v1/artifacts/build_atlas/load_datasets/01-merge/")
annotation = nxfvars.get("annotation", "../../tables/annotation")
reference_meta = nxfvars.get("reference_meta", "../../tables/reference_meta.yaml")
cpus = nxfvars.get("cpus", 24)

# %%
os.environ["NUMBA_NUM_THREADS"] = str(cpus)
threadpool_limits(cpus)
sc.settings.n_jobs = cpus

# %%
# Read the YAML file and load it into a dictionary
with open(reference_meta, "r") as f:
    ref_meta_dict = yaml.load(f, Loader=yaml.Loader)

# %% [markdown]
# ## 1. Load adatas

# %%
# Get file paths and reorder by study and sample id
files = [str(x) for x in Path(datasets_path).glob("*.h5ad")]
files = [x for x in files if not isinstance(x, float)]
files.sort()

dataset = [Path(file).stem.split("-adata")[0] for file in files]

# %%
dataset

# %%
# Concat adatas from specified directory
adatas = process_map(sc.read_h5ad, files, max_workers=cpus)

# Convert adatas list to named adatas dict
adatas = {dataset: adata for dataset, adata in zip(dataset, adatas)}


# %% [markdown]
# ### Burclaff_2022

# %%
adatas["Burclaff_2022"] = adatas["Burclaff_2022"][
    ~(adatas["Burclaff_2022"].obs["sample_tissue"] == "small intestine")
].copy()

# %% [markdown]
# ### Conde_2022

# %%
adatas["Conde_2022"] = adatas["Conde_2022"][
    ~(adatas["Conde_2022"].obs["sample_tissue"] == "small intestine")
].copy()

# %% [markdown]
# ### Elmentaite_2021
# has samples that are not from colon. In addition, Elmentaite_2021_gutcellatlas has patients that were not available as fastqs.

# %%
# Drop "appendix", "ileum", "duodenum", "jejunum" samples from all datasets that contain "Elmentaite"
for item in [item for item in dataset if "Elmentaite" in item]:
    adatas[item] = adatas[item][
        ~(
            adatas[item]
            .obs["anatomic_location"]
            .isin(["appendix", "ileum", "duodenum", "jejunum"])
        )
    ].copy()

# %% [markdown]
# ### GarridoTrigo_2023

# %%
# Drop "inflamed" samples
adatas["GarridoTrigo_2023"] = adatas["GarridoTrigo_2023"][
    ~(adatas["GarridoTrigo_2023"].obs["sample_type"] == "inflamed")
].copy()

# %% [markdown]
# ### Chen_2024_Cancer_Cell

# %%
# Drop "small intestine" samples
adatas["Chen_2024"] = adatas["Chen_2024"][
    ~(adatas["Chen_2024"].obs["sample_tissue"] == "small intestine")
].copy()

# %% [markdown]
# ### Han_2020

# %%
# Drop small intestine and blood samples + adipose?
adatas["Han_2020"] = adatas["Han_2020"][
    ~(adatas["Han_2020"].obs["sample_tissue"] == "small intestine")
].copy()

adatas["Han_2020"] = adatas["Han_2020"][
    ~(adatas["Han_2020"].obs["anatomic_location"].isin(["blood", "adipose"]))
].copy()

# %% [markdown]
# ### He_2020

# %%
adatas["He_2020"] = adatas["He_2020"][
    ~(adatas["He_2020"].obs["sample_tissue"] == "small intestine")
].copy()

# %% [markdown]
# ### Khaliq_2022

# %%
# Rename PiK3CA column
adatas["Khaliq_2022"].obs.rename(
    columns={"PiK3CA_status_driver_mut": "PIK3CA_status_driver_mut"}, inplace=True
)

# %% [markdown]
# ### Li_2017_Nat_Genet

# %%
# Normalize Smart-seq2 matrix by gene length (from ensembl)
# see: https://docs.scvi-tools.org/en/stable/tutorials/notebooks/scrna/tabula_muris.html#normalize-smartseq2-matrix-by-gene-length
# Some of the non protein coding transcripts are super high after this ?!
adatas["Li_2017"].X = adatas["Li_2017"].X /adatas["Li_2017"].var["Length"].values * np.median(adatas["Li_2017"].var["Length"].values)
adatas["Li_2017"].X = np.rint(adatas["Li_2017"].X)
adatas["Li_2017"].X = csr_matrix(adatas["Li_2017"].X)

# %% [markdown]
# ### Liu_2022_CD45Pos

# %%
# Drop datset completely ? -> Only normalized counts available!
#adatas.pop("Liu_2022_CD45Pos", None)
#dataset.remove("Liu_2022_CD45Pos")

# %% [markdown]
# ### Liu_2024_mixCD45PosCD45Neg

# %%
# Remove normal liver samples
adatas["Liu_2024_mixCD45PosCD45Neg"] = adatas["Liu_2024_mixCD45PosCD45Neg"][
    ~(
        (adatas["Liu_2024_mixCD45PosCD45Neg"].obs["sample_type"] == "normal")
        & (adatas["Liu_2024_mixCD45PosCD45Neg"].obs["sample_tissue"] == "liver")
    )
].copy()

# %% [markdown]
# ### Mazzurana_2021_Cell_Res

# %%
# Normalize Smart-seq2 matrix by gene length (from ensembl)
# Some of the non protein coding transcripts are super high after this ?!
adatas["Mazzurana_2021_CD127Pos"].X = adatas["Mazzurana_2021_CD127Pos"].X /adatas["Mazzurana_2021_CD127Pos"].var["length"].values * np.median(adatas["Mazzurana_2021_CD127Pos"].var["length"].values)
adatas["Mazzurana_2021_CD127Pos"].X = np.rint(adatas["Mazzurana_2021_CD127Pos"].X)
adatas["Mazzurana_2021_CD127Pos"].X = csr_matrix(adatas["Mazzurana_2021_CD127Pos"].X)

# %% [markdown]
# ### MUI_Innsbruck and MUI_Innsbruck_AbSeq

# %%
adatas["MUI_Innsbruck"] = adatas["MUI_Innsbruck"][
    ~(adatas["MUI_Innsbruck"].obs["sample_tissue"].astype(str) == "nan")
].copy()

# %%
adatas["MUI_Innsbruck_AbSeq"] = adatas["MUI_Innsbruck_AbSeq"][
    adatas["MUI_Innsbruck_AbSeq"]
    .obs["Sample_Name"]
    .isin(["Undetermined", "Multiplet", "CML_s", "healthy_us", "healthy_s"])
].copy()

# %% [markdown]
# ### Masuda_2022_CD3Pos

# %%
# Drop datset completely -> No sample mapping available -> Some samples stimulated in culture
#adatas.pop("Masuda_2022_CD3Pos", None)
#dataset.remove("Masuda_2022_CD3Pos")

# %% [markdown]
# ### Parikh_2019

# %%
# Drop "Ulcerative colitis" samples (for now) -> Possible to keep "sample_type" = "normal", equivalent to Ulcerative colitis matched normal sample
adatas["Parikh_2019"] = adatas["Parikh_2019"][
    ~(adatas["Parikh_2019"].obs["condition"] == "Ulcerative colitis")
].copy()

# %%
# Drop weird gene symbols from dataset "UCSC annotation"
adatas["Parikh_2019"] = adatas["Parikh_2019"][
    :, adatas["Parikh_2019"].var_names.str.startswith("ENSG")
].copy()

# %%
mapping = {
    "n (%) male: 1 (33%)": "nan",
}
adatas["Parikh_2019"].obs["sex"] = adatas["Parikh_2019"].obs["sex"].map(mapping)

# %%
del adatas["Parikh_2019"].layers["log_norm"]

# %% [markdown]
# ### Pelka_2021

# %%
# Drop CiteSeq data from all datasets that contain "Pelka_2021_10X" -> mybe save for later?
for item in [item for item in dataset if "Pelka_2021_10X" in item]:
    adatas[item] = adatas[item][:, adatas[item].var_names.str.startswith("ENSG")].copy()

# %% [markdown]
# ### Sathe_2023

# %%
# Remove normal liver samples
adatas["Sathe_2023"] = adatas["Sathe_2023"][
    ~(
        (adatas["Sathe_2023"].obs["sample_type"] == "normal")
        & (adatas["Sathe_2023"].obs["sample_tissue"] == "liver")
    )
].copy()

# %% [markdown]
# ### Scheid_2023_CD27PosCD38Pos

# %%
# Drop "Ulcerative colitis" samples
adatas["Scheid_2023_CD27PosCD38Pos"] = adatas["Scheid_2023_CD27PosCD38Pos"][
    adatas["Scheid_2023_CD27PosCD38Pos"].obs["sample_type"] == "normal"
].copy()

# %% [markdown]
# ### Tian_2023

# %%
# Rename BRAF column
adatas["Tian_2023"].obs = adatas["Tian_2023"].obs.rename(
    columns={"BRAFV600E_status_driver_mut": "BRAF_status_driver_mut"}
)

# %%
del adatas["Tian_2023"].layers["log_norm"]

# %% [markdown]
# ### Uhlitz_2021
# This dataset has much to detailed driver mutation infos.

# %%
def aggregate_driver(row, prefix):
    for column in row.index:
        if column.startswith(prefix) and row[column] == "mut":
            return "mut"
    return "wt"


# %%
# list common driver sub strings
driver_mut = [
    "TP53",
    "KRAS",
    "BRAF",
    "PIK3CA",
    "APC",
    "FBXW7",
    "NOTCH1",
    "TGFBR2",
    "DNMT1",
    "AXIN2",
    "PI4KA",
    "HRAS",
]
# Aggregate to new columns
for prefix in driver_mut:
    adatas["Uhlitz_2021"].obs[f"{prefix}_status_driver"] = adatas[
        "Uhlitz_2021"
    ].obs.apply(lambda row: aggregate_driver(row, prefix), axis=1)

# Drop "old" columns
columns_to_drop = [
    col for col in adatas["Uhlitz_2021"].obs.columns if "_status_driver_mut" in col
]
adatas["Uhlitz_2021"].obs.drop(columns=columns_to_drop, inplace=True)

# Append back "_mut" to match columns in other datasets
adatas["Uhlitz_2021"].obs.columns = [
    col + "_mut" if col.endswith("_status_driver") else col
    for col in adatas["Uhlitz_2021"].obs.columns
]

# %% [markdown]
# ### UZH_Zurich datasets

# %%
adatas["UZH_Zurich_CD45Pos"] = adatas["UZH_Zurich_CD45Pos"][
    adatas["UZH_Zurich_CD45Pos"]
    .obs["Sample_Name"]
    .isin(["Undetermined", "Multiplet", "T", "N", "B", "HB"])
].copy()

# %%
adatas["UZH_Zurich_healthy_blood"] = adatas["UZH_Zurich_healthy_blood"][
    adatas["UZH_Zurich_healthy_blood"]
    .obs["Sample_Name"]
    .isin(["Undetermined", "Multiplet", "HB11", "HB10", "HB6", "HB8", "HB7", "HB9"])
].copy()

# %% [markdown]
# ### Wang_2020

# %%
# Drop "small intestine" samples
adatas["Wang_2020"] = adatas["Wang_2020"][
    ~(adatas["Wang_2020"].obs["sample_tissue"] == "small intestine")
].copy()

# %% [markdown]
# ### Wang_2023_CD45Pos

# %%
# Remove normal liver samples
adatas["Wang_2023_CD45Pos"] = adatas["Wang_2023_CD45Pos"][
    ~(
        (adatas["Wang_2023_CD45Pos"].obs["sample_type"] == "normal")
        & (adatas["Wang_2023_CD45Pos"].obs["sample_tissue"] == "liver")
    )
].copy()

# %% [markdown]
# ### Wu_2022_CD45Pos

# %%
adatas["Wu_2022_CD45Pos"] = adatas["Wu_2022_CD45Pos"][
    :, adatas["Wu_2022_CD45Pos"].var_names.str.startswith("ENSG")
].copy()

# Remove normal liver samples
adatas["Wu_2022_CD45Pos"] = adatas["Wu_2022_CD45Pos"][
    ~(
        (adatas["Wu_2022_CD45Pos"].obs["sample_type"] == "normal")
        & (adatas["Wu_2022_CD45Pos"].obs["sample_tissue"] == "liver")
    )
].copy()

# %% [markdown]
# ### Zhang_2020

# %%
del adatas["Zhang_2020_10X_CD45Pos"].obsm["X_umap"]
del adatas["Zhang_2020_CD45Pos_CD45Neg"].obsm["X_umap"]

# %% [markdown]
# ## 2. Filter datasets before concatenating

# %%
def _filter_obs(_adata):
    sc.pp.filter_cells(_adata, min_counts=400)
    sc.pp.filter_cells(_adata, min_genes=100)
    return _adata


# %%
filtered_adatas = process_map(
    _filter_obs,
    [adata for _, adata in adatas.items()],
    max_workers=cpus,
)
# Convert adatas list to named adatas dict
filtered_adatas = {dataset: adata for dataset, adata in zip(dataset, filtered_adatas)}

# %% [markdown]
# # Concat adatas

# %%
adata = anndata.concat(filtered_adatas, index_unique="-", join="outer", fill_value=0)
adata.obs_names = adata.obs_names.str.rsplit("-", n=1).str[0]
adata.X = csr_matrix(adata.X.astype(int))

# %% [markdown]
# ### 1. Harmonize gene annotations
# Add back gtf info, filter not expressed genes. For now I will keep all expressed genes, I can still filter them out later on. I am doing this now to apply the next filtering steps to the same genes.

# %%
annot = {}
for f in Path(annotation).glob("*_gene_annotation_table.csv"):
    gtf = f.name.replace("_gene_annotation_table.csv", "")
    df = pd.read_csv(f)

    if gtf.startswith("ensembl") and "gene_biotype" in df.columns:
        # Rename columns as in gencode gtf
        df["gene_biotype"] = df["gene_biotype"].fillna(value=df["gene_source"])
        del df["gene_source"]
        df = df.rename(
            columns={
                "gene_id": "Geneid",
                "gene_name": "GeneSymbol",
                "chromosome": "Chromosome",
                "start": "Start",
                "end": "End",
                "gene_biotype": "Class",
                "strand": "Strand",
                "length": "Length",
            }
        )
    elif len(df.columns) != 9:
        df = df.rename(
            columns={
                "gene_id": "Geneid",
                "gene_name": "GeneSymbol",
            }
        )
    annot[gtf] = df

annotation = pd.concat(
    [df.assign(Version=gtf) for gtf, df in annot.items()], ignore_index=True
)
annotation.insert(0, "ensembl", annotation["Geneid"].apply(ah.pp.remove_gene_version))
annotation["OriginalOrder"] = annotation.groupby("Version").cumcount()

# Sort df by tuple gencode/ensembl, version number and ".cumcount()" the original gtf order (by chromosome and position)
def custom_sort_key(version):
    prefix, num = version.split(".v")
    num = num.split("_")[0]
    return (prefix, int(num))


annotation = (
    annotation.assign(sort_key=annotation["Version"].apply(custom_sort_key))
    .sort_values(by=["sort_key", "OriginalOrder"], ascending=[False, True])
    .drop(columns=["sort_key", "OriginalOrder"])
)
# Drop duplicates in column ensembl and keep "first" makes sure we keep the latest available gene id
annotation = annotation.drop_duplicates(subset="ensembl", keep="first").reset_index(
    drop=True
)

# %%
# map back gtf info to .var
adata.var["ensembl"] = adata.var_names

for column in [col for col in annotation.columns if col != "ensembl"]:
    adata.var[column] = adata.var["ensembl"].map(
        annotation.set_index("ensembl")[column]
    )

# %%
# Reorder by gtf (i.e. chromosome position)
gene_index = annotation[annotation["ensembl"].isin(adata.var_names)]["ensembl"].values
# missing_genes = [gene for gene in adata.var_names if gene not in gene_index]
missing_genes = adata[:, ~adata.var_names.isin(annotation["ensembl"])].var_names.values
gene_index = np.append(gene_index, missing_genes)
adata = adata[:, gene_index]

# %%
# Should be dropped anyway using the 25 % per dataset rule
adata = adata[:, adata.var_names.str.startswith("ENSG")].copy()

# %%
# Remove not expressed genes
sc.pp.filter_genes(adata, min_cells=10)
sc.pp.filter_genes(adata, min_counts=10)

# %%
adata.shape

# %% [markdown]
# ### Get fraction of genes expressed across a minimum of 25% naive datasets

# %%
# Exclude enriched datasets
adata_naive = adata[adata.obs["enrichment_cell_types"] == "naive"]
# Exclude Tian_2023 -> large portion of the dataset is missing; Exclude Huang_2024 -> only immune cells; Exclude Borras_2023_SMC5 -> only CD8 Tcells
adata_naive = adata_naive[~adata_naive.obs["dataset"].isin(["Tian_2023", "Huang_2024", "Borras_2023_SMC5"])]

# %%
# Filter genes: must be expressed in at least 25 percent of the not enriched datasets
res = pd.DataFrame(
    columns=adata_naive.var_names, index=adata_naive.obs["dataset"].cat.categories
)
for sample in adata_naive.obs["dataset"].cat.categories:
    res.loc[sample] = adata_naive[adata_naive.obs["dataset"].isin([sample]), :].X.sum(0)

keep = res.columns[res[res == 0].count(axis=0) / len(res.index) <= 0.25]

# %%
adata[:, keep].shape

# %%
keep_dict = {key: True for key in keep}
adata.var["Dataset_25pct_Overlap"] = adata.var_names.map(keep_dict).fillna(False)
adata.var["Dataset_25pct_Overlap"].value_counts()

# %%
adata_var = adata.var
sym = (
    adata_var[adata_var.duplicated(subset=["GeneSymbol"], keep=False)]["GeneSymbol"]
    .value_counts()
    .reset_index()
)
sym[sym["count"] > 5]

# %%
values_to_replace = [
    "gene_status KNOWN",
    "gene_status NOVEL",
    "gene_status PUTATIVE",
    "Y_RNA",
    "Metazoa_SRP",
    "_",
]

# Replace values with pd.NA in column GeneSymbol -> will map to ensembl!
adata.var["GeneSymbol"] = adata.var["GeneSymbol"].replace(values_to_replace, pd.NA)

# %%
# switch var_names back to symbols after harmonizing
adata.var_names = adata.var["GeneSymbol"].fillna(adata.var["ensembl"])
adata.var_names.name = None
adata.var_names_make_unique()

# %%
adata_var_gtf = adata.var.reset_index(names="var_names")
adata_var_gtf.to_csv(f"{artifact_dir}/adata_var_gtf.csv", index=False)

# switch var_names back to ensembl after harmonizing
adata.var_names = adata.var["ensembl"]
adata.var_names.name = None

# %% [markdown] tags=[]
# ### 2. Harmonize patient/sample meta

# %%
# Same patient can have multiple samples such as "naive" + "CD45+" -> Need to use "study_id" not to count same patient multiple times!
# For sample should be ok to consider each seperately, even when same region was sampled twice (replicate) and processed in the same way
# adata.obs["sample_id"] = (
#    adata.obs["dataset"].astype(str) + "." + adata.obs["sample_id"].astype(str)
# )

adata.obs["patient_id"] = (
    adata.obs["study_id"].astype(str) + "." + adata.obs["patient_id"].astype(str)
)
# An additional enrichment of T cells was performed for two patients (SC040 and SC044) of the KUL5 dataset
# Set patient_id to match same patient from original study!
adata.obs.loc[adata.obs["patient_id"] == "Borras_2023_Cell_Discov.SC040", "patient_id"] = "Joanito_2022_Nat_Genet.SC040"
adata.obs.loc[adata.obs["patient_id"] == "Borras_2023_Cell_Discov.SC044", "patient_id"] = "Joanito_2022_Nat_Genet.SC044"

adata.obs["patient_id"] = adata.obs["patient_id"].str.replace("-", "_")

# %%
driver_cols = [item for item in adata.obs.columns if "_status_driver_mut" in item]
driver_cols += ["patient_id"]

# %%
df_mutations = (
    adata.obs[driver_cols]
    .groupby("patient_id", observed=True)
    .first()
    .notna()
    .sum()
    .sort_values(ascending=False)
    .reset_index(name="number of patients")
    .rename(columns={"index": "mutation"})
)

df_mutations["mutation"] = df_mutations["mutation"].str.replace(
    "_status_driver_mut", ""
)

# %%
# Set display options to show all rows and columns
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# %%
df_mutations

# %%
pd.reset_option("display.max_rows")
pd.reset_option("display.max_columns")

# %%
adata.obs["medical_condition"] = adata.obs["medical_condition"].astype(str)
adata.obs.loc[
    adata.obs["dataset"].isin(
        ["VUMC_HTAN_cohort3", "VUMC_HTAN_discovery", "VUMC_HTAN_validation"]
    ),
    "medical_condition",
] = "colorectal polyp"

# %%
# Get driver_mut column names and replace values for normal samples with "nan"
tumor_related_columns = [
    col for col in adata.obs.columns if "_status_driver_mut" in col
]

adata.obs[tumor_related_columns] = adata.obs[tumor_related_columns].astype(str)

# Set the values to "nan" for tumor_related_columns sample_type == "normal"
adata.obs.loc[adata.obs["sample_type"] == "normal", tumor_related_columns] = "nan"

# %%
# Replace values for normal samples with "normal" in "tumor_source"
adata.obs["tumor_source"] = adata.obs["tumor_source"].astype(str)
adata.obs.loc[adata.obs["sample_type"] == "normal", "tumor_source"] = "normal"

# %%
# Replace values in "sample_type"/"sample_tissue"
adata.obs["sample_type"] = adata.obs["sample_type"].astype(str)
adata.obs.loc[adata.obs["sample_tissue"] == "lymph node", "sample_type"] = "lymph node"
adata.obs["sample_tissue"] = adata.obs["sample_tissue"].astype(str)
adata.obs.loc[adata.obs["sample_tissue"] == "lymph node", "sample_tissue"] = "mesenteric lymph nodes"

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
# Replace roman charcters
mapping = {
    "III": "III",
    "IV": "IV",
    "IIA": "IIA",
    "IIIB": "IIIB",
    "II": "II",
    "IIIC": "IIIC",
    "IIB": "IIB",
    "IIIA": "IIIA",
    "nan": np.nan,
    "IVA": "IVA",
    "I": "I",
    "ⅣA": "IVA",
    "IIC": "IIC",
    "IVB": "IVB",
    "IVC": "IVC",
    "ⅡB": "IIB",
    "ⅢB": "IIIB",
    "ⅡA": "IIA",
    "Ⅱ": "II",
    "ⅢC": "IIIC",
}
adata.obs["tumor_stage_TNM"] = adata.obs["tumor_stage_TNM"].astype(str).map(mapping)

# %% [markdown]
# -> Infer missing sex info from transcriptional data?

# %%
# Example Paper Joanito_2022_Nat_Genet: The study involves 26 Singaporean, 23 Korean and 14 Belgian patients diagnosed with CRC who underwent surgery
# -> Inferring this might be a bit of a stretch though ...
adata.obs["ethnicity"] = adata.obs["ethnicity"].astype(str)
mapping = {
    "Austria": "white",
    "Germany": "white",
    "Belgium": "white",
    "Switzerland": "white",
    "Spain": "white",
    "Sweden": "white",
    "France": "nan", # white | african american | etc ...
    "Ireland": "white",
    "Netherlands": "white",
    "United Kingdom": "nan",  # white | african american | asian
    "US": "nan",  # white | african american | etc ...
    "Singapore": "asian",
    "China": "asian",
    "South Korea": "asian",
    "nan": "nan",
}
adata.obs["ethnicity"] = (
    adata.obs["ethnicity"].astype(str).fillna(adata.obs["country"].map(mapping))
)

# %%
ref_meta_cols = []
for key, sub_dict in ref_meta_dict.items():
    ref_meta_cols.append(key)

# %%
ref_meta_cols += [item for item in adata.obs.columns if "_status_driver_mut" in item]
ref_meta_cols += ["multiplex_batch", "Sample_Name"]

# %%
# Subset adata for columns in the reference meta
adata.obs = adata.obs[[col for col in ref_meta_cols if col in adata.obs.columns]].copy()

# %%
# Replace "NaN", "nan", and "na" with pd.NA in the obs/var DataFrame
adata.obs = adata.obs.applymap(lambda x: str(x).replace("NaN", str(pd.NA)))
adata.obs.replace(
    {"NaN": pd.NA, "nan": pd.NA, "na": pd.NA, np.nan: pd.NA, "<NA>": pd.NA},
    inplace=True,
)
adata.var = adata.var.applymap(lambda x: str(x).replace("NaN", str(pd.NA)))
adata.var.replace(
    {"NaN": pd.NA, "nan": pd.NA, "na": pd.NA, np.nan: pd.NA, "<NA>": pd.NA},
    inplace=True,
)

# %%
# Replace "_driver_mut" in column names
adata.obs.columns = [
    col.replace("_driver_mut", "") if "_status_driver_mut" in col else col
    for col in adata.obs.columns
]

# %%
# Before writing to .h5ad append str to numerical columns to prevent weird errors when saving
for column in ["age", "tumor_size", "tumor_dimensions"]:
    adata.obs[column] = adata.obs[column].map(
        lambda x: str(x) + ("years" if "age" in column else "cm") if pd.notna(x) else x
    )

# %% [markdown] tags=[]
# #### Set batch key

# %%
# Set batch key for denoising step to sample_id
adata.obs["batch"] = adata.obs["sample_id"].astype(str)

# %%
# Set batch key for BD datasets to multiplex_batch
# -> Need to append study_id/dataset to batch key before final scvi integration of the full adata -> patient_id/sample_id duplicates across datasets !!!
adata.obs.loc[adata.obs["dataset"] == "MUI_Innsbruck", "batch"] = adata.obs.loc[
    adata.obs["dataset"] == "MUI_Innsbruck", "multiplex_batch"
]

adata.obs.loc[adata.obs["dataset"] == "MUI_Innsbruck_AbSeq", "batch"] = adata.obs.loc[
    adata.obs["dataset"] == "MUI_Innsbruck_AbSeq", "multiplex_batch"
]

adata.obs.loc[adata.obs["dataset"] == "UZH_Zurich_CD45Pos", "batch"] = adata.obs.loc[
    adata.obs["dataset"] == "UZH_Zurich_CD45Pos", "multiplex_batch"
]

adata.obs.loc[adata.obs["dataset"] == "UZH_Zurich_healthy_blood", "batch"] = adata.obs.loc[
    adata.obs["dataset"] == "UZH_Zurich_healthy_blood", "multiplex_batch"
]

# %%
adata.obs = adata.obs[
    [
        "batch",
        "study_id",
        "dataset",
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
        "medical_condition",
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
        "Sample_Name",
    ]
].copy()

# %%
assert np.all(np.modf(adata.X.data)[0] == 0), "X does not contain all integers"
assert adata.var_names.is_unique
assert adata.obs_names.is_unique

# %%
len(adata.obs["patient_id"].unique())

# %%
len(adata.obs["sample_id"].unique())

# %%
len((adata.obs["dataset"].astype(str) + "-" + adata.obs["sample_id"].astype(str)).unique())

# %%
adata.X.nnz

# %%
# Set display options to show all rows and columns
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# %%
adata.obs["batch"].value_counts()

# %%
adata.obs["dataset"].value_counts()

# %%
pd.reset_option("display.max_rows")
pd.reset_option("display.max_columns")

# %%
# Save harmonized sample meta
adata.obs.groupby("sample_id", observed=False).first().to_csv(f"{artifact_dir}/merged_sample_metadata.csv")

# %% [markdown]
# ### Save adatas by dataset

# %%
datasets = [
    adata[adata.obs["dataset"] == dataset, :].copy()
    for dataset in adata.obs["dataset"].unique()
    if adata[adata.obs["dataset"] == dataset, :].shape[0] != 0
]
# Convert adatas list to named adatas dict
datasets = {dataset: adata for dataset, adata in zip(dataset, datasets)}


# %%
def _write_adata(_adata):

    # Remove all nan columns, else write_h5ad will fail
    _adata.obs = _adata.obs.dropna(axis=1, how="all")

    # Remove not expressed genes from dataset
    sc.pp.filter_genes(_adata, min_counts=1)

    # Get the dataset name from the subset
    _dataset = _adata.obs["dataset"].values[0]
    _adata.write_h5ad(f"{artifact_dir}/{_dataset}-adata.h5ad", compression="lzf")

    return _dataset


# %%
process_map(
    _write_adata,
    [adata for _, adata in datasets.items()],
    max_workers=cpus,
)

# %% [markdown]
# ### Save raw adatas by dataset (scAR input)

# %%
datsets_fastq_scAR = (
    adata[adata.obs["cellranger_version"] == "7.1.0"].obs["dataset"].unique().tolist()
)
datsets_fastq_scAR.append("HTAPP_HTAN")
datsets_fastq_scAR.append("Ji_2024_scopeV2")
datsets_fastq_scAR.append("Ji_2024_scopeV1")
datsets_fastq_scAR.append("MUI_Innsbruck")
datsets_fastq_scAR.append("UZH_Zurich_CD45Pos")
datsets_fastq_scAR.append("UZH_Zurich_healthy_blood")
datsets_fastq_scAR.append("MUI_Innsbruck_AbSeq")
datsets_fastq_scAR.append("Sathe_2023")
datsets_fastq_scAR.append("WUSTL_HTAN")
datsets_fastq_scAR.append("Thomas_2024_Nat_Med_CD45Pos")
datsets_fastq_scAR

# %%
raw_adatas = {key: adatas[key] for key in datsets_fastq_scAR}

# %%
# get gene dict to switch raw to unique symbol var_names
var_names_dict = adata_var_gtf.set_index("ensembl")["var_names"].to_dict()
gene_overlap_dict = adata.var.set_index("ensembl")["Dataset_25pct_Overlap"].to_dict()

# get batch key dict used to split the adata by for scAR denosing
# as this is already per dataset subset, no need to add "dataset" to "sample_id"
batch_dict = (
    adata.obs[["sample_id", "batch"]]
    .groupby("sample_id", observed=False)
    .first()["batch"]
    .to_dict()
)


# %%
def _write_raw(_raw):

    _raw.obs["batch"] = _raw.obs["sample_id"].map(batch_dict)
    # No need for all metadata in raw
    _raw.obs = _raw.obs[
        ["batch", "study_id", "dataset", "sample_id", "sample_type", "patient_id"]
    ]
    # Remove all nan columns, else write_h5ad will fail
    _raw.obs = _raw.obs.dropna(axis=1, how="all")

    # switch raw var_names to harmonized gene symbols (same as in corresponding adata)
    _raw.var["var_names"] = _raw.var_names.map(var_names_dict).fillna("drop")
    # Remove not expressed genes from dataset
    _raw = _raw[:, _raw.var["var_names"] != "drop"].copy()
    sc.pp.filter_genes(_raw, min_counts=1)

    #_raw.var_names = _raw.var["var_names"]
    #_raw.var_names.name = None
    #assert _raw.var_names.is_unique
    
    _raw.var["Dataset_25pct_Overlap"] = _raw.var_names.map(gene_overlap_dict)

    # Get the dataset name from the subset
    _dataset = _raw.obs["dataset"].values[0]
    _raw.write_h5ad(f"{artifact_dir}/{_dataset}-raw.h5ad", compression="lzf")

    return _dataset


# %%
process_map(
    _write_raw,
    [raw for _, raw in raw_adatas.items()],
    max_workers=cpus,
)
