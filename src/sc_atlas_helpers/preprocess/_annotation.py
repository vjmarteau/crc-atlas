from collections import abc as cabc
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData


def score_seeds(
    adata: AnnData,
    seed_marker_genes: Dict[str, Dict[str, list]],
    *,
    pos_cutoff: Union[float, List[float]] = 0.1,
    neg_cutoff: Union[float, List[float]] = 0,
    unidentified_label: str = "unknown",
    layer: str = "log1p_norm",
) -> pd.Series:
    """Label seed cell types based on input pos/neg marker gene expression"""

    def _score_ct(
        adata, seed_marker_genes, layer, pos_cutoff, neg_cutoff, unidentified_label
    ):
        tmp_df = pd.DataFrame(index=adata.obs_names)

        # Ensure cutoff values are sequences
        if not isinstance(pos_cutoff, cabc.Sequence) or isinstance(pos_cutoff, str):
            pos_cutoff = [pos_cutoff] * len(seed_marker_genes)
        if not isinstance(neg_cutoff, cabc.Sequence) or isinstance(neg_cutoff, str):
            neg_cutoff = [neg_cutoff] * len(seed_marker_genes)

        for idx, (cell_type, markers) in enumerate(seed_marker_genes.items()):
            positive_conditions = []
            negative_conditions = []

            pos_cutoff_val = pos_cutoff[idx]
            neg_cutoff_val = neg_cutoff[idx]

            for positive_markers_list in markers["positive"]:
                positive_condition = (
                    np.ravel(
                        adata[:, adata.var_names.isin(positive_markers_list)]
                        .layers[layer]
                        .sum(1)
                    )
                    > pos_cutoff_val
                )
                positive_conditions.append(positive_condition)

            for negative_markers_list in markers["negative"]:
                negative_condition = ~(
                    np.ravel(
                        adata[:, adata.var_names.isin(negative_markers_list)]
                        .layers[layer]
                        .sum(1)
                    )
                    > neg_cutoff_val
                )
                negative_conditions.append(negative_condition)

            combined_positive_condition = np.all(positive_conditions, axis=0)
            combined_negative_condition = np.all(negative_conditions, axis=0)

            condition_met = combined_positive_condition & combined_negative_condition
            tmp_df[f"tmp_{cell_type}"] = np.where(
                condition_met, cell_type, unidentified_label
            )

        return tmp_df

    def _combine_ct(row):
        for ct in seed_ct:
            if row[ct] != unidentified_label:
                return row[ct]
        return unidentified_label

    seed_df = _score_ct(
        adata, seed_marker_genes, layer, pos_cutoff, neg_cutoff, unidentified_label
    )
    seed_ct = [f"tmp_{ct}" for ct in seed_marker_genes.keys()]

    return seed_df.apply(_combine_ct, axis=1)


def annot_cells(
    adata: AnnData,
    label: str,
    *,
    label_neg: str = None,
    pos_marker: Union[str, List[str]] = ["None"],
    cutoff_pos_marker_expression: str = "> 0",
    neg_marker: Union[str, List[str]] = ["None"],
    cutoff_neg_marker_expression: str = "> 0",
    cutoff_total_counts: str = None,
    cutoff_n_genes_by_counts: str = None,
    cutoff_n_genes: str = None,
    cutoff_mito: str = None,
    layer: str = "log1p_norm",
    new_col: str = "new_col",
    display: bool = False,
    legend_loc: str = "right margin",
    **kwargs,
) -> pd.Series:
    """
    Annotate cells based on boolean conditions such as the presence or absence of marker gene expression and/or QC metric thresholds.

    Parameters
    ----------
    adata
        AnnData object
    label
        Label assigned when overall conditions are True
    label_neg
        Label assigned when overall conditions are False. Defaults to first input kwargs.
    pos_marker
        Positive marker genes. If multiple genes are specified, their expression values are summed. The more genes are specified, the higher the likelihood of annotating more cells based on their cumulative expression.
    cutoff_pos_marker_expression
        A string defining the expression-based cutoff for positive markers. Defaults to "> 0".
    neg_marker
        Negative marker genes. If multiple genes are specified, their expression values are summed. Likelihood of annotating cells decreases with the number of neagtive genes.
    cutoff_neg_marker_expression
        String defining the threshold for negative marker expression. Default is "> 0". Cells with positive expression above this threshold will be excluded from the selection.
    cutoff_total_counts
        Threshold for total counts
    cutoff_n_genes_by_counts
        Threshold for the number of genes by counts
    cutoff_n_genes
        Threshold for the number of expressed genes
    cutoff_mito
        Threshold for mitochondrial gene expression
    min_expression
        Minimum expression level. Defaults to 0.
    layer
        Name of the AnnData object layer where gene expression cutoff is applied. This layer is also used for summing expression values when considering multiple positive marker genes. By default adata.layers["log1p_norm"] is used.
    new_col
        Name of column for assigned labels. Defaults to "new_col"
    display
        Plot umap coloured by adata.obs["new_col"]
    legend_loc
        Location of the legend. Defaults to "right margin".
    **kwargs
        Additional keyword arguments. Example: leiden["0", "1"], Supplying one is mandatory!
    """

    # todo: have mandatory first groupby argument in the form of leiden["0", "1", etc]?, append to keys = [], subsets = []

    def _process_condition(metric_col: float, condition: str) -> bool:
        """ Split input condition string in operator and float and apply to metric_col"""
        
        if condition is not None:
            operator = condition.split()[0]
            threshold = float(condition.split()[1])
            
            comparison_function = {
                ">": lambda x: x > threshold,
                "<": lambda x: x < threshold,
                ">=": lambda x: x >= threshold,
                "<=": lambda x: x <= threshold,
                "==": lambda x: x == threshold,
                "!=": lambda x: x != threshold,
            }
            return comparison_function[operator](metric_col)
        return None
    
    # List different default conditions
    conditions = [
        (pos_marker != ["None"], _process_condition(metric_col=np.ravel(adata[:, adata.var_names.isin(pos_marker)].layers[layer].sum(1)), condition=cutoff_pos_marker_expression)),
        (neg_marker != ["None"], ~(_process_condition(metric_col=np.ravel(adata[:, adata.var_names.isin(neg_marker)].layers[layer].sum(1)), condition=cutoff_neg_marker_expression))),
        (cutoff_total_counts is not None, _process_condition(metric_col=adata.obs['total_counts'], condition=cutoff_total_counts)),
        (cutoff_n_genes_by_counts is not None, _process_condition(metric_col=adata.obs['n_genes_by_counts'], condition=cutoff_n_genes_by_counts)),
        (cutoff_n_genes is not None, _process_condition(metric_col=adata.obs['n_genes'], condition=cutoff_n_genes)),
        (cutoff_mito is not None, _process_condition(metric_col=adata.obs['pct_counts_mito'], condition=cutoff_mito))
    ]

    # kwargs: Add possibility to look within subclusters e.g leiden=1 and sample=s1, or group=wt, etc.
    # First specified key is used as reference to fill in unmapped

    if len(kwargs) == 0:
        raise ValueError("At least one keyword argument in **kwargs is required.")

    keys = []
    subsets = []
    if len(kwargs) > 0:
        for key, subset in kwargs.items():
            bool_cluster = np.ravel(adata.obs[key].astype(str).isin(subset))
            conditions.append((True, bool_cluster))
            keys.append(key)
            subsets.append(subset)

    # Remove conditions that were not spcified and aggregate
    conditions = [result for condition, result in conditions if condition]
    all_conditions_met = np.ravel(tuple(all(x) for x in zip(*conditions)))

    if label_neg == None:
        label_neg = adata.obs[keys[0]].astype(str)

    # Add column using boolean
    adata.obs[new_col] = np.where(
        all_conditions_met,
        label,
        label_neg
    )

    if display == True:
        sc.pl.umap(
            adata,
            color=new_col,
            legend_loc=legend_loc,
            legend_fontoutline=1,
            legend_fontsize="xx-small",
            title=None,
        )

        # updated_subset = subsets[0]
        # updated_subset.append(label)
        # print(adata[adata.obs[new_col].isin(updated_subset)].obs[new_col].value_counts())
        print(adata.obs[new_col].value_counts()) # show all or only subset?

    return adata.obs[new_col]


def score_cell_types_by_marker_expression(
    adata: AnnData,
    cell_types_dict: Dict[str, List[List[str]]],
    groupby: str,
    *,
    pct_thresh_pos_cells_in_cluster: float = 50,
    cutoff_pos_marker_expression: str = "> 0",
    cutoff_neg_marker_expression: str = "> 0",
    layer: str = "log1p_norm",
) -> pd.Series:
    """
    Score cell types based on marker gene expression within the categories defined by the `groupby` column.

    Parameters
    ----------
    adata
        AnnData object
    cell_types_dict
        Dictionary mapping cell types to positive and negative markers
    groupby
        Key for grouping cells.
    min_expression
        Minimum expression level. Defaults to 0.
    pct_thresh
        Percentage threshold for the number of cells within a cluster expressing the marker to be assigned a cell type label; otherwise, the cluster reverts to its original grouping.
    """   
    adata.obs["cell_type"] = adata.obs[groupby]
    
    for cell_type, markers in cell_types_dict.items():
        markers[0] = markers[0] if markers[0] else ["None"]
        markers[1] = markers[1] if markers[1] else ["None"]

        adata.obs["cell_type"] = annot_cells(
            adata,
            label=cell_type,
            pos_marker=markers[0],
            cutoff_pos_marker_expression=cutoff_pos_marker_expression,
            neg_marker=markers[1],
            cutoff_neg_marker_expression=cutoff_neg_marker_expression,
            layer=layer,
            new_col="cell_type",
            display=False,
            cell_type=adata.obs["cell_type"].unique().tolist(),
        )
        
    result_df = pd.DataFrame()

    for value in adata.obs[groupby].unique():
        value_counts = adata[adata.obs[groupby].isin([value])].obs["cell_type"].value_counts().reset_index()
        cluster_size = adata[adata.obs[groupby].isin([value])].shape[0]
        value_counts[groupby] = value
        value_counts["size"] = cluster_size
        result_df = pd.concat([result_df, value_counts], ignore_index=True)

    result_df["percent"] = pd.to_numeric(result_df["count"] / result_df["size"] * 100, errors='coerce')
    result_df['cell_type'] = result_df.apply(lambda row: row.name if row['percent'] < pct_thresh_pos_cells_in_cluster else row['cell_type'], axis=1)
    
    cell_type = result_df.groupby(groupby).first()["cell_type"].to_dict()
    return adata.obs[groupby].map(cell_type).astype(str)


def reprocess_adata_subset_scvi(
    adata: AnnData,
    n_neighbors: int = 10,
    leiden_res: float = 1,
    use_rep: str = "X_scVI",
    neighbors_kws: Optional[Dict] = None,
) -> None:
    """Recompute UMAP and leiden on a adata subset when scVI is used (no additional
    batch correction)"""
    neighbors_kws = {} if neighbors_kws is None else neighbors_kws
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors, **neighbors_kws)
    sc.tl.leiden(adata, resolution=leiden_res, flavor="igraph", n_iterations=-1)
    sc.tl.paga(adata)
    sc.pl.paga(adata, plot=False)
    sc.tl.umap(adata, init_pos='paga')


def annotate_cell_types(
    adata: AnnData,
    cell_type_map: Dict[str, list],
    key_added: str = "cell_type",
    default_cell_type: str = "other",
    column: str = "leiden",
) -> None:
    """Generate a cell-type column from a Mapping cell_type -> [list of clusters]"""
    res = np.full(adata.shape[0], default_cell_type, dtype=object)
    for ct, clusters in cell_type_map.items():
        clusters = [str(x) for x in clusters]
        res[adata.obs[column].isin(clusters)] = ct
    adata.obs[key_added] = res
    sc.pl.umap(adata, color=key_added)


def integrate_back(
    adata: AnnData,
    adata_subset: AnnData,
    variable: str = "cell_type",
) -> None:
    """Merge cell type annotations performed on a subset back into the main
    AnnData object"""
    adata.obs[variable] = adata.obs[variable].astype("str")
    adata.obs.loc[adata_subset.obs.index, variable] = adata_subset.obs[variable].astype("str")
