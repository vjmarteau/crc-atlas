#!/usr/bin/env python3

import argparse
import sys

import scanpy as sc
import scvi


def parse_args(args=None):
    Description = "Run Ambient RNA removal (scAR) algorithm per sample"
    Epilog = "Example usage: python scAR.py <raw_adata> <filtered_adata> <adata_out> <cpus>"

    parser = argparse.ArgumentParser(description=Description, epilog=Epilog)
    parser.add_argument("--raw_adata")
    parser.add_argument("--filtered_adata")
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


def Run_scAR(raw_adata, filtered_adata, adata_out, cpus, set_all_seeds=set_all_seeds):
    from scipy.sparse import csr_matrix
    import multiprocessing
    from threadpoolctl import threadpool_limits

    threadpool_limits(cpus)
    sc.settings.n_jobs = cpus
    set_all_seeds()
    
    # Ambient RNA removal (scAR)

    assert len(raw_adata.obs_names) > len(filtered_adata.obs_names) # check that raw_adata has more barcodes than filtered_adata

    scvi.external.SCAR.setup_anndata(filtered_adata)

    try:
        profile = scvi.external.SCAR.get_ambient_profile(
            adata=filtered_adata, raw_adata=raw_adata, prob=0.995
        )
    except Exception as e:
        print("Encountered exception:", e)
        print("Retrying with prob=0.99")
        profile = scvi.external.SCAR.get_ambient_profile(
            adata=filtered_adata, raw_adata=raw_adata, prob=0.99
        )

    model = scvi.external.SCAR(filtered_adata)

    try:
        model.train(accelerator="gpu")

    except ValueError as ve:
        if "Expected more than 1 value per channel when training" in str(ve):
            print("Encountered exception:", ve)
            print("Retrying with batch_size=130")
            model.train(accelerator="gpu", batch_size=130) # see https://discourse.scverse.org/t/solo-scvi-train-error-related-to-batch-size/1591

    filtered_adata.obsm["X_scAR"] = model.get_latent_representation()
    filtered_adata.layers["denoised"] = model.get_denoised_counts()
    filtered_adata.layers["denoised"] = csr_matrix(filtered_adata.layers["denoised"]) # not sure if this is needed?
    filtered_adata.layers["counts"] = filtered_adata.X.copy()
    filtered_adata.X = filtered_adata.layers["denoised"]

    filtered_adata.write_h5ad(adata_out, compression="lzf")


def main(args=None):
    args = parse_args(args)

    Run_scAR(
        raw_adata=sc.read_h5ad(args.raw_adata),
        filtered_adata=sc.read_h5ad(args.filtered_adata),
        adata_out=args.adata_out,
        cpus=args.cpus,
    )


if __name__ == "__main__":
    sys.exit(main())
