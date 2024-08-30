#!/usr/bin/env python3

import argparse
import sys

import pandas as pd
import scanpy as sc
import scvi


def parse_args(args=None):
    Description = "Run scvi-tools integration"
    Epilog = "Example usage: python scVI.py <input_adata> <batch_key> <labels_key> <model_gene_dispersion> <model_out> <adata_out> <cpus>"

    parser = argparse.ArgumentParser(description=Description, epilog=Epilog)
    parser.add_argument("--input_adata")
    parser.add_argument("--batch_key", type=str)
    parser.add_argument("--labels_key", type=str)
    parser.add_argument("--model_gene_dispersion", type=str)
    parser.add_argument("--model_out")
    parser.add_argument("--adata_out")
    parser.add_argument("--cpus", type=int)
    return parser.parse_args(args)


def set_all_seeds(seed=0):
    import os
    import random

    import numpy as np
    import torch

    scvi.settings.seed = seed
    os.environ["PYTHONHASHSEED"] = str(seed)  # Python general
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)  # Numpy random
    random.seed(seed)  # Python random
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multiGPU
        torch.set_float32_matmul_precision("high")


def Run_scVI(
    input_adata,
    *,
    batch_key,
    labels_key=None,
    model_gene_dispersion=None,
    model_out,
    adata_out,
    cpus,
    set_all_seeds=set_all_seeds,
):
    """Run scvi-tools integration"""

    import multiprocessing
    from threadpoolctl import threadpool_limits

    threadpool_limits(cpus)
    sc.settings.n_jobs = cpus
    set_all_seeds()

    input_adata = sc.read_h5ad(input_adata)

    adata_scvi = input_adata[:, input_adata.var["highly_variable"]].copy()
    print("scvi adata dimensions:", adata_scvi.shape)

    if labels_key is not None:
        scvi.model.SCVI.setup_anndata(
            adata_scvi,
            # layer="counts",
            labels_key=labels_key,  # set labels key to dataset/platform to account for different gene dispersion across platforms/datasets?
            batch_key=batch_key,
            #   categorical_covariate_keys=[labels_key, batch_key],
            #   continuous_covariate_keys=["pct_counts_mito", "pct_counts_ribo", "pct_counts_hb"], not supported for SOLO
        )
    else:
        scvi.model.SCVI.setup_anndata(
            adata_scvi,
            batch_key=batch_key,
        )

    if model_gene_dispersion is not None:
        model = scvi.model.SCVI(adata_scvi, dispersion=model_gene_dispersion)
    else:
        model = scvi.model.SCVI(adata_scvi)

    model.train(accelerator="gpu")
    model.save(model_out, save_anndata=True)

    input_adata.obsm["X_scVI"] = model.get_latent_representation()
    input_adata.write(adata_out, compression="lzf")

def main(args=None):
    args = parse_args(args)

    Run_scVI(
        input_adata=args.input_adata,
        batch_key=args.batch_key,
        labels_key=args.labels_key,
        model_gene_dispersion=args.model_gene_dispersion,
        model_out=args.model_out,
        adata_out=args.adata_out,
        cpus=args.cpus,
    )


if __name__ == "__main__":
    sys.exit(main())
