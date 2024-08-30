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
# # As these are the in house datasets, have a seperate look at our own data
The BD datasets were annotated seperately and I will simply map back this annotation here!

# %%
# ensure reproducibility -> set numba multithreaded mode
import os
from collections import OrderedDict
from itertools import chain

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
    "../../results/v1/artifacts/build_atlas/integrate_datasets/neighbors_leiden_umap_seed/leiden_paga_umap/bd-umap.h5ad",
)
bd_is_droplet_path = nxfvars.get(
    "bd_is_droplet_path",
    "../../tables/bd-is_droplet.csv",
)
artifact_dir = nxfvars.get(
    "artifact_dir",
    "/data/scratch/marteau/downloads/seed/",
)

# %%
adata = sc.read_h5ad(adata_path)
bd_is_droplet = pd.read_csv(bd_is_droplet_path)

# %%
scales_counts = sc.pp.normalize_total(adata, target_sum=None, inplace=False)
adata.layers["log1p_norm"] = sc.pp.log1p(scales_counts["X"], copy=True)

# %%
for col in ["is_droplet", "cell_type_coarse", "cell_type_fine"]:    
    is_droplet_dict = bd_is_droplet.set_index("obs_names")[col].to_dict()
    adata.obs[col] = adata.obs_names.map(is_droplet_dict)

# %%
sc.pl.umap(adata, color="is_droplet")

# %%
adata = adata[adata.obs["is_droplet"] != "empty droplet"].copy()
ah.pp.reprocess_adata_subset_scvi(adata, leiden_res=2, n_neighbors=15, use_rep="X_scVI")
adata.write(f"{artifact_dir}/bd-adata.h5ad", compression="lzf")

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    fig = sc.pl.umap(
        adata,
        color="cell_type_coarse",
        legend_fontoutline=1,
        legend_fontsize="xx-small",
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/bd-umap_cell_type_coarse.png", bbox_inches="tight")

# %%
with plt.rc_context({"figure.figsize": (6, 6)}):
    fig = sc.pl.umap(
        adata,
        color="cell_type_fine",
        legend_fontoutline=1,
        legend_fontsize="xx-small",
        return_fig=True,
    )
    fig.savefig(f"{artifact_dir}/bd-umap_cell_type_fine.png", bbox_inches="tight")

# %%
adata.shape

# %%
adata.obs["cell_type_fine"].value_counts(dropna=False)

# %%
bd_is_droplet.to_csv(f"{artifact_dir}/bd-is_droplet.csv", index=False)
