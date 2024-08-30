import re
from typing import List

import anndata
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import csc_matrix, csr_matrix


def remove_gene_version(string: str) -> str:
    """
    Remove gene version from ensembl gene id that start with "ENSG".
    """
    string = str(string)
    if string.startswith("ENSG"):
        return re.sub(r"\..*", "", string)
    else:
        return string


def append_duplicate_suffix(df: pd.DataFrame, column: str, sep: str) -> pd.DataFrame:
    """
    Appends a numeric suffix to each duplicated value in the specified column based on index position.
    """
    suffix_dict = {}
    df_sorted = df.sort_values(by=column)
    df_sorted = df_sorted.reset_index().sort_values(by="index").set_index("index")
    for i, val in enumerate(df_sorted[column]):
        if val in suffix_dict:
            suffix_dict[val] += 1
            df_sorted.at[df_sorted.index[i], column] = f"{val}{sep}{suffix_dict[val]}"
        else:
            suffix_dict[val] = 0
    return df_sorted.sort_index()


def find_unmapped_genes(adata: anndata.AnnData, column: str = "var_names") -> List[str]:
    """
    Finds genes in the specified AnnData object that are not mapped to any ensembl id.
    """
    unmapped = []
    if column in adata.var.columns:
        col_data = adata.var[column]
    else:
        col_data = adata.var_names
    unmapped += col_data[~adata.var_names.str.startswith("ENSG")].tolist()
    return unmapped


def aggregate_duplicate_gene_ids(
    adata: anndata.AnnData, gene_names: List[str]
) -> anndata.AnnData:
    """
    Collapse duplicate gene IDs in an AnnData object by summing their expression values.

    Parameters
    ----------
    adata : anndata.AnnData
    The AnnData object to process.
    gene_names : List[str]
    A list of gene names to collapse by summing their expression values.

    Returns
    -------
    anndata.AnnData
    The modified AnnData object with collapsed gene IDs.

    Notes
    -----
    Gene IDs that are not in the input list will be kept in the output AnnData object.
    Gene IDs that are empty strings, "nan", or np.nan will be excluded from the input list.
    """
    gene_names = [g for g in gene_names if g not in ["", "nan", np.nan]]
    if not gene_names:
        return adata
    
    adata.X = adata.X.tocsc()
    sums = [csc_matrix.sum(adata[:, gene].X, axis=1) for gene in gene_names]
    
    # Reshape the list of arrays to match the dimensions of adata.X
    new_X = np.concatenate(sums).reshape(-1, len(gene_names))
    
    adata_sums = anndata.AnnData(
        X=new_X,
        obs=adata.obs,
        var=pd.DataFrame(gene_names).set_index(0).rename_axis(None),
    )
    # Remove duplicated genes before concat
    adata = adata[:, ~(adata.var_names.isin(gene_names))]
    adata_tmp = anndata.concat([adata, adata_sums], axis=1)
    adata_tmp.obs = adata.obs # Hope this preserves the right order ...
    return adata_tmp


def undo_log_norm(adata: anndata.AnnData) -> anndata.AnnData:
    """
    Reverse a log-normalization, assuming that each
    sample has at least one cell with exactely one count. This assumption
    is reasonable, at least for 10x data.
    """
    adata.layers["log_norm"] = adata.X.copy()
    x_log_norm = adata.X.tocsr()
    x_log_norm.sort_indices()
    x_norm = x_log_norm.copy()
    x_norm.data = np.expm1(x_norm.data)

    # assuming that each sample has at least one cell with exactely one count
    size_factors = np.array([np.min(x_norm[i, :].data) for i in range(x_norm.shape[0])])
    x_raw_counts = scipy.sparse.diags(1 / size_factors) @ x_norm
    x_raw_counts.data = np.rint(x_raw_counts.data)
    x_raw_counts.sort_indices()
    adata.X = x_raw_counts.copy()
    return adata
