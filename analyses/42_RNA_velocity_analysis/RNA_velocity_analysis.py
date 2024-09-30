# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: scvelo2
#     language: python
#     name: scvelo2
# ---

# %%
import glob
from multiprocessing import Pool

import anndata as ad
import matplotlib as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import seaborn as sns

# Set the number of max CPUs to be used by the processes
from threadpoolctl import threadpool_limits

cpus = 4
threadpool_limits(cpus)

# %% [markdown]
# # Do not run - code to load in and merge the LOOM files produced by velocyto (if merged h5ad file is not available)

# %% [markdown]
# Get the customised LOOM files (here I specify only the MUI dataset)

# %%
# loom_files = []
# loom_path = "/data/projects/2022/CRCA/results/RNA_velocity/velocyto_results/custom_loom/*MUI_Innsbruck-*.loom"
# for file in glob.glob(loom_path, recursive=True):
#     loom_files.append(file)

# %% [markdown]
# Load in the files as adata - this will take ages due to the size of the loom files, need to use multiprocessing
# Function to process a single Loom file (will be used for multiprocessing pool)

# %%
# def process_loom_file(lfile):
#     print("processing: " + lfile + "\n")
#     ladata = scv.read(lfile, cache=False)
#     ladata.var_names_make_unique()
#     return ladata

# %% [markdown]
# Number of processes to use

# %%
# num_processes = os.cpu_count()  # Run only if you want to use the number of all available CPU cores
# num_processes = 6

# %% [markdown]
# Create a multiprocessing Pool

# %%
# with Pool(processes=num_processes) as pool:
#     ladata_list = pool.map(process_loom_file, loom_files)

# %% [markdown]
# Concatenate the processed Loom files

# %%
# ladata_merged = ad.concat(ladata_list)

# %% [markdown]
# Load in the H5AD data

# %%
# adata = sc.read_h5ad(
#     "/home/fotakis/myScratch/CRC_atlas_Neutro/data/MUI_only_neutro_anno_final.h5ad"
# )

# %% [markdown]
# Merge the H5AD and LOOM adata

# %%
# adata_merged = scv.utils.merge(adata, ladata_merged)

# %% [markdown]
# Do not run - save the merged data

# %%
# adata_merged.write_h5ad("../data/MUI_only_neutro_anno_final_velo.h5ad")

# %% [markdown]
# Do not run - save the loom data

# %%
# ladata_merged.write_h5ad("../data/MUI_only_neutro_LOOM.h5ad")

# %% [markdown]
# # Start here:

# %% [markdown]
# ## 1. Dynamical (E/M) Model

# %%
# Load in the annotated h5ad
adata = sc.read_h5ad(
    "/data/projects/2022/CRCA/results/v1/final/h5ads/mui_innsbruck_neutrophil_BN_TAN_subsets-adata.h5ad"
)

# %%
# Load in the LOOM h5ad
ladata_merged = sc.read_h5ad("../data/MUI_only_neutro_LOOM.h5ad")

# %%
# Merge the H5AD and LOOM adata
adata_merged = scv.utils.merge(adata, ladata_merged)

# %%
# Filter & normalise
scv.pp.filter_and_normalize(adata_merged, min_shared_counts=20, n_top_genes=2000)
scv.pp.moments(adata_merged, n_pcs=30, n_neighbors=30)

# %%
# Visualise spliced-unspliced proportions
scv.pl.proportions(adata_merged, groupby="cell_type_fine")

# %%
# Recover the velocity dynamics
scv.tl.recover_dynamics(adata_merged, n_jobs=10)

# %%
# Based on the above dynamics - run the EM model
scv.tl.velocity(adata_merged, group="cell_type_fine", mode="dynamical")
scv.tl.velocity_graph(adata_merged, n_jobs=10)  # mode_neighbors='connectivities'

# %%
# Plot and save the velocity graph (EM)
# Note: The default figure params are changed here (I find these params produce more "readable" plots)
# Feel free to change
sc.set_figure_params(figsize=(6, 6), dpi=400)
# remove the grid lines
plt.rcParams["axes.grid"] = False
scv.pl.velocity_embedding_stream(
    adata_merged,
    basis="umap",
    color="cell_type_fine",
    title="E/M model",
    legend_loc="on right",
    save="scvelo_EM_neutro_latest.png",
)

# %% [markdown]
# ## 2. Psuedotime

# %%
# We first need to assess the root cell.
# In order to do that we need to plot the diffusion maps
sc.tl.diffmap(adata_merged)

# %%
# Check the first 2 diffusion components
sc.pl.scatter(
    adata_merged,
    basis="diffmap",
    color=["cell_type_fine"],
    components=[1, 2],
)

# %%
# In order to decide for a root cell we need to look at the extremities of the graph
# Since we expect that the initial state lies in the BN clusters we can focus on the upper right extremity
# As it is evident the most likely root cell would belong to cluster BN2
# Set the root cell:
adata_merged.uns["iroot"] = np.flatnonzero(adata_merged.obs["cell_type_fine"] == "BN1")[
    0
]

# %%
# Print the root cell - needed for the pseudotime clalculation:
adata_merged.uns["iroot"]

# %%
# Compute the pseudotime using the root cell
scv.tl.velocity_pseudotime(adata_merged, root_key=9)

# %%
# Plot and save results
scv.pl.scatter(
    adata_merged, color="velocity_pseudotime", cmap="gnuplot", save="MUI_pseudotime.png"
)

# %% [markdown]
# ## 3. PAGA trajectory

# %%
# this is needed due to a current bug - bugfix is coming soon.
adata_merged.uns["neighbors"]["distances"] = adata_merged.obsp["distances"]
adata_merged.uns["neighbors"]["connectivities"] = adata_merged.obsp["connectivities"]

# Set "minimum_spanning_tree = True" for dotted lines connecting the clusters
# scv.tl.paga(adata_merged, groups="cell_type_fine", minimum_spanning_tree=False, root_key=56, end_key=249)
scv.tl.paga(
    adata_merged,
    groups="cell_type_fine",
    minimum_spanning_tree=False,
    use_time_prior=True,
)  # , root_key="BN1", end_key=249)
# df = scv.get_df(adata_merged, "paga/transitions_confidence", precision=2).T # Use this with "minimum_spanning_tree = True"
df = scv.get_df(adata_merged, "paga/transitions_confidence", precision=1).T
df.style.background_gradient(cmap="Blues").format("{:.2g}")

# %%
# Produce and save the PAGA plot
scv.pl.paga(
    adata_merged,
    basis="umap",
    size=50,
    alpha=0.1,
    min_edge_width=2,
    node_size_scale=1.5,
    color="cell_type_fine",
    dpi=400,
    threshold=0,
    save="scvelo_PAGA_neutro_latest.png",
)

# %%
