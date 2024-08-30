from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.stats import median_abs_deviation


def annot_empty_droplets(adata: AnnData, boolean: np.ndarray) -> np.ndarray:
    """Annotates droplets in the 'adata' object based on the provided `boolean` condition."""

    if "is_droplet" not in adata.obs.columns:
        adata.obs["is_droplet"] = "cell"

    is_droplet = np.where(
        (adata.obs["is_droplet"] == "cell") & boolean, "cell", "droplet"
    )
    return is_droplet


def filter_droplets(
    adata: AnnData,
    *,
    variable: str = "cell_type",
    subset: str = None,
    min_counts: int = 100,
    min_genes: int = 100,
) -> pd.Series:
    """Filter droplets based on min_counts and min_genes in adata subset.
    """
    
    adata_subset = adata[adata.obs[variable].isin([subset])].obs[[variable, "total_counts", "n_genes"]].copy()
    
    condition_mask = adata_subset.apply(
        lambda x: x["total_counts"] < min_counts or x["n_genes"] < min_genes, axis=1
    )
    
    values = np.where(condition_mask, "droplet", adata_subset[variable].astype("str"))

    is_droplet = adata.obs[variable].copy()
    is_droplet.loc[adata_subset.index] = values.astype("str")

    return is_droplet


def is_outlier(adata: AnnData, metric_col: str, *, groupby: Optional[str] = None, n_mads: float = 5) -> pd.Series:
    """Detect outliers by median absolute deviation (MAD).

    Adapted from https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html#motivation

    Parameters
    ----------
    adata
        AnnData object
    metric_col
        column in adata.obs to consider
    groupby
        grouping variable. If specified, outliers will be determined for each value of this grouping variable separately.
        E.g. `dataset`.
    n_mads
        label points that are outside of median +/- nmads * MAD.
    """

    def _is_outlier(df):
        """Check if metric value deviates from the median by more than n_mads MADs."""
        metric_values = df[metric_col]
        return np.abs(metric_values - np.median(metric_values)) > n_mads * median_abs_deviation(metric_values)

    if groupby is None:
        return _is_outlier(adata.obs)
    else:
        return adata.obs.groupby(groupby, observed=False).apply(_is_outlier).droplevel(groupby).reindex(adata.obs_names)  # type: ignore


def score_outliers(
    adata: AnnData,
    *,
    groupby: str = "sample",
    n_mads: float = 5,
    n_mads_mito: float = 3,
    include_mito: bool = True,
) -> pd.Series:
    """
    Wrapper function for `is_outlier`. Returns "outlier" if at least two metric value deviate from from the median by more than n_mads MADs.

    Parameters
    ----------
    adata
        AnnData object
    groupby
        grouping variable. If specified, outliers will be determined for each value of this grouping variable separately.
        E.g. `dataset`, `sample`, `cell_type`
    n_mads
        label points that are outside of median +/- nmads * MAD.
    n_mads_mito
        mito specific label points that are outside of median +/- nmads * MAD.
    include_mito
        Whether to include `is_outlier_mito`. Defaults to True.
    """

    if all(col not in adata.var.columns for col in ["mito", "ribo", "hb"]):
        # mitochondrial genes
        adata.var["mito"] = adata.var_names.str.startswith("mt-")
        # ribosomal genes
        adata.var["ribo"] = adata.var_names.str.startswith(("Rps", "Rpl"))
        # hemoglobin genes.
        adata.var["hb"] = adata.var_names.str.contains("^Hb[^(p)]")

        sc.pp.calculate_qc_metrics(
            adata,
            qc_vars=["mito", "ribo", "hb"],
            inplace=True,
            percent_top=[20],
            log1p=True,
        )

    qc_metrics = [
        "log1p_total_counts",
        "log1p_n_genes_by_counts",
        "pct_counts_in_top_20_genes",
        "pct_counts_mito",
    ]
    outlier_columns = [
        "is_outlier_counts",
        "is_outlier_genes",
        "is_outlier_top_20",
        "is_outlier_mito",
    ]

    df = pd.DataFrame()
    for qc_metric, outlier_col in zip(qc_metrics, outlier_columns):
        df[outlier_col] = is_outlier(
            adata,
            qc_metric,
            n_mads=(n_mads if "mito" not in outlier_col else n_mads_mito),
            groupby=groupby,
        )

    if not include_mito:
        outlier_columns.remove("is_outlier_mito")

    condition_mask = np.sum(df.loc[:, outlier_columns], axis=1) >= 2

    outlier = pd.Series(
        np.where(condition_mask, "outlier", "normal"),
        index=adata.obs_names,
        name="is_outlier",
    )

    return outlier
