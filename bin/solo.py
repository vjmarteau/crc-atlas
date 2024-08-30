#!/usr/bin/env python3

import argparse
import sys

import pandas as pd
import scanpy as sc
import scvi


def parse_args(args=None):
    Description = "Run scvi-tools doublet detection (SOLO)"
    Epilog = "Example usage: python solo.py <input_adata> <input_model> <batch_key> <labels_key> <doublet_status> <cpus>"

    parser = argparse.ArgumentParser(description=Description, epilog=Epilog)
    parser.add_argument("--input_adata")
    parser.add_argument("--input_model")
    parser.add_argument("--batch_key", type=str)
    parser.add_argument("--labels_key", type=str)
    parser.add_argument("--doublet_status")
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


def Run_solo(input_adata, input_model, batch_key, labels_key, doublet_status, cpus, set_all_seeds=set_all_seeds):
    import multiprocessing
    from threadpoolctl import threadpool_limits

    threadpool_limits(cpus)
    sc.settings.n_jobs = cpus
    set_all_seeds()

    # Doublet detection (SOLO)

    if labels_key is not None:
        scvi.model.SCVI.setup_anndata(
            input_adata,
            # layer="counts",
            labels_key=labels_key,
            batch_key=batch_key,
            #   categorical_covariate_keys=[labels_key, batch_key],
            #   continuous_covariate_keys=["pct_counts_mito", "pct_counts_ribo", "pct_counts_hb"], not supported for SOLO
        )
    else:
        scvi.model.SCVI.setup_anndata(
            input_adata,
            batch_key=batch_key,
        )

    scvi_model = scvi.model.SCVI.load(
        dir_path=input_model,
        adata=input_adata[:, input_adata.var["highly_variable"]].copy(),
        accelerator="gpu",
    )

    solo_models = [
        scvi.external.SOLO.from_scvi_model(scvi_model, restrict_to_batch=b)
        for b in input_adata.obs[batch_key].unique()
    ]


    def run_solo(solo_model):
        try:
            solo_model.train(accelerator="gpu")

        except ValueError as ve:
            if "Expected more than 1 value per channel when training" in str(ve):
                print("Encountered exception:", ve)
                print("Retrying with batch_size=130")
                solo_model.train(
                    accelerator="gpu",
                    batch_size=130,
                )  # see https://discourse.scverse.org/t/solo-scvi-train-error-related-to-batch-size/1591

        res = solo_model.predict(soft=True)
        res["SOLO_is_doublet"] = solo_model.predict(soft=False)
        res = res.rename(
            columns={"doublet": "SOLO_doublet_prob", "singlet": "SOLO_singlet_prob"}
        )
        return res


    solo_res = []

    # Skip samples with very few cells!
    if solo_models:
        for solo_model in solo_models:
            try:
                solo_res.append(run_solo(solo_model))
            except TypeError as te:
                print("Skipping solo_model due to error:", te)
                continue

    solo_res = pd.concat(solo_res)
    solo_res.to_csv(doublet_status)


def main(args=None):
    args = parse_args(args)

    Run_solo(
        input_adata=sc.read_h5ad(args.input_adata),
        input_model=args.input_model,
        batch_key=args.batch_key,
        labels_key=args.labels_key,
        doublet_status=args.doublet_status,
        cpus=args.cpus,
    )


if __name__ == "__main__":
    sys.exit(main())
