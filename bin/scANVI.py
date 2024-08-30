#!/usr/bin/env python3

import argparse
import sys

import pandas as pd
import scanpy as sc
import scvi


def parse_args(args=None):
    Description = "Run scvi-tools scANVI integration"
    Epilog = "Example usage: python scANVI.py <input_adata> <input_model> <batch_key> <labels_key> <model_out> <adata_out> <cpus>"

    parser = argparse.ArgumentParser(description=Description, epilog=Epilog)
    parser.add_argument("--input_adata")
    parser.add_argument("--input_model")
    parser.add_argument("--batch_key", type=str)
    parser.add_argument("--labels_key", type=str)
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


def Run_scANVI(input_adata, input_model, batch_key, labels_key, model_out, adata_out, cpus, set_all_seeds=set_all_seeds):
    import multiprocessing
    from threadpoolctl import threadpool_limits

    threadpool_limits(cpus)
    sc.settings.n_jobs = cpus
    set_all_seeds()

    scvi_adata = input_adata[:, input_adata.var["highly_variable"]].copy()

    scvi.model.SCVI.setup_anndata(
            scvi_adata,
            batch_key=batch_key,
        )

    scvi_model = scvi.model.SCVI.load(
        dir_path=input_model,
        adata=scvi_adata,
        accelerator="gpu",
    )

    # scANVI
    print("scANVI integration")

    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        adata=scvi_adata,
        labels_key=labels_key,
        unlabeled_category="unknown",
    )

    scanvi_model.train(accelerator="gpu")
    scanvi_model.save(model_out, save_anndata=True)

    input_adata.obsm[f"X_scANVI"] = scanvi_model.get_latent_representation(scvi_adata)
    input_adata.obs[f"{labels_key}_predicted"] = scanvi_model.predict(scvi_adata)

    input_adata.write(adata_out, compression="lzf")


def main(args=None):
    args = parse_args(args)

    Run_scANVI(
        input_adata=sc.read_h5ad(args.input_adata),
        input_model=args.input_model,
        batch_key=args.batch_key,
        labels_key=args.labels_key,
        model_out=args.model_out,
        adata_out=args.adata_out,
        cpus=args.cpus,
    )


if __name__ == "__main__":
    sys.exit(main())
