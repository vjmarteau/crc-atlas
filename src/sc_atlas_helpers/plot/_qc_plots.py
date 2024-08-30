from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

        
def umap_qc_metrics(
    adata: AnnData,
    *,
    vmax_total_counts: str = "p99",
    vmax_n_genes_by_counts: str = "p99",
    vmax_n_genes: str = "p99",
    save_plot: Optional[bool] = False,
    prefix: str = "",
    save_dir: Optional[str] = None,
    file_ext: str = "pdf",
    display: bool = True,
) -> plt.Figure:
    """
    Plot umap of quality control (QC) metrics. Wrapper function for scanpy plot umap.

    Parameters
    ----------
    adata
        AnnData object
    vmax_total_counts
        Maximum value for total counts color scale. Defaults to "p99"
    vmax_n_genes_by_counts
        Maximum value for number of genes by counts color scale. Defaults to "p99"
    vmax_n_genes
        Maximum value for number of genes color scale. Defaults to "p99"
    save_plot
        Whether to save the plot. Defaults to False.
    prefix
        Prefix appended to the file name when saving.
    save_dir
        Directory to save the plot. Defaults to None.
    display
        Show umap plots coloured by qc_metrics
    """

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes=axes.flatten()
    sc.pl.umap(
        adata,
        color="pct_counts_mito",
        cmap="inferno",
        vmin=0,
        vmax="p99",
        sort_order=False,
        show=False,
        ax=axes[0],
    )
    sc.pl.umap(
        adata,
        color="total_counts",
        cmap="inferno",
        vmax=vmax_total_counts,
        sort_order=False,
        show=False,
        ax=axes[1],
    )
    sc.pl.umap(
        adata,
        color="n_genes_by_counts",
        cmap="inferno",
        vmax=vmax_n_genes_by_counts,
        sort_order=False,
        show=False,
        ax=axes[2],
    )
    sc.pl.umap(
        adata,
        color="n_genes",
        cmap="inferno",
        vmax=vmax_n_genes,
        sort_order=False,
        show=False,
        ax=axes[3],
    )
    plt.tight_layout()
    
    if save_plot == True:
        fig.savefig(f"{save_dir}/{prefix}_umap_qc_metrics.{file_ext}", bbox_inches="tight")
        if display == True:
            plt.show()
        plt.close(fig)


def violin_qc_metrics(
    adata: AnnData,
    *,
    groupby: str = "sample_id",
    figwidth: int = 10,
    save_plot: Optional[bool] = False,
    prefix: str = "",
    save_dir: Optional[str] = None,
    file_ext: str = "pdf",
    display: bool = True,
) -> plt.Figure:
    """
    Plot violin plots of quality control (QC) metrics. Wrapper function for scanpy plot violin.

    Parameters
    ----------
    adata
        AnnData object
    save_plot
        Whether to save the plot. Defaults to False.
    prefix
        Prefix appended to the file name when saving.
    save_dir
        Directory to save the plot. Defaults to None.
    display
        Show umap plots coloured by qc_metrics
    """

    fig, axes = plt.subplots(4, 1, figsize=(figwidth, 5))
    axes = axes.flatten()
    sc.pl.violin(
        adata,
        "total_counts",
        groupby=groupby,
        log=True,
        cut=0,
        rotation=90,
        ylabel="",
        show=False,
        ax=axes[0],
    )
    axes[0].set_title("total_counts")
    sc.pl.violin(
        adata,
        "n_genes_by_counts",
        groupby=groupby,
        log=True,
        cut=0,
        rotation=90,
        ylabel="",
        show=False,
        ax=axes[1],
    )
    axes[1].set_title("n_genes_by_counts")
    sc.pl.violin(
        adata,
        "pct_counts_mito",
        groupby=groupby,
        rotation=90,
        show=False,
        ylabel="",
        ax=axes[2],
    )
    axes[2].set_title("pct_counts_mito")
    sc.pl.violin(
        adata,
        "pct_counts_ribo",
        groupby=groupby,
        rotation=90,
        show=False,
        ylabel="",
        ax=axes[3],
    )
    axes[3].set_title("pct_counts_ribo")
    plt.tight_layout()

    if save_plot == True:
        fig.savefig(
            f"{save_dir}/{prefix}_violin_plot_qc_metrics.{file_ext}",
            bbox_inches="tight",
        )
        if display == True:
            plt.show()
        plt.close(fig)


def umap_covariates(
    adata: AnnData,
    covariates: list[str],
    *,
    save_plot: Optional[bool] = False,
    prefix: str = "",
    save_dir: Optional[str] = None,
    file_ext: str = "pdf",
    display: bool = True,
) -> plt.Figure:
    """
    Plot umap of covariates. Wrapper function for scanpy plot umap.

    Parameters
    ----------
    adata
        AnnData object
    covariates
        List of covariates to plot.
    save_plot
        Whether to save the plot. Defaults to False.
    prefix
        Prefix appended to the file name when saving.
    save_dir
        Directory to save the plot. Defaults to None.
    display
        Show umap plots coloured by qc_metrics
    """

    fig = sc.pl.umap(
        adata,
        color=covariates,
        frameon=False,
        return_fig=True,
        ncols=2,
    )

    num_axes = len(fig.axes)  # Get the number of subplots

    if num_axes >= 5:
        for j in np.arange(1, 5, 1):
            fig.axes[j].set_ylabel("")
            fig.axes[j].set_xlabel("")
    elif num_axes >= 2:
        for j in np.arange(1, 2, 1):
            fig.axes[j].set_ylabel("")
            fig.axes[j].set_xlabel("")

    if save_plot == True:
        fig.savefig(f"{save_dir}/{prefix}_umap_covariates.{file_ext}", bbox_inches="tight")
        if display == True:
            plt.show()
        plt.close(fig)


def umap_marker_genes(
    adata: AnnData,
    marker_genes: pd.DataFrame,
    *,
    col_cell_type: str = "cell_type",
    col_gene_id: str = "gene_identifier",
    subset: Optional[List[str]] = None,
    cmap: str = "inferno",
    layer: str = "log1p_norm",
    save_plot: Optional[bool] = False,
    prefix: str = "",
    save_dir: Optional[str] = None,
    file_ext: str = "pdf"
) -> Dict[str, plt.Figure]:
    """
    Plot umap of marker genes per cell type. Wrapper function for scanpy plot umap.

    Parameters
    ----------
    adata
        AnnData object
    marker_genes
        DataFrame containing marker genes and associated cell types. Mandatory columns: "cell_type" and "gene_identifier".
    subset
        Subset of cell types to plot markers for. By default all available markers for all cell types are plotted.
    cmap
        Colormap for the UMAP plot. Defaults to "inferno".
    layer
        Layer in the AnnData object to use for plotting. By default adata.layers["log1p_norm"] is used.
    save_plot
        Whether to save the plot. Defaults to False.
    prefix
        Prefix appended to the file name when saving.
    save_dir
        Directory to save the plot. Defaults to None.
    """
    
    #if not all(col in marker_genes.columns for col in ["cell_type", "gene_identifier"]):
    #    missing_columns = [col for col in ['cell_type', "gene_identifier"] if col not in marker_genes.columns]
    #    raise ValueError(f"marker_genes DataFrame is missing the following columns: {missing_columns}")

    marker = {
        key: marker_genes.loc[lambda x: x[col_cell_type] == key, col_gene_id]
        .dropna()
        .tolist()
        for key in marker_genes[col_cell_type].unique()
    }
    
    if subset is not None:
        marker = {key: marker[key] for key in subset}
    
    marker_genes_in_data = dict()
    for ct, markers in marker.items():
        markers_found = list()
        for marker in markers:
            if marker in adata.var.index:
                markers_found.append(marker)
        marker_genes_in_data[ct] = markers_found
    marker_genes_in_data = {k: v for k, v in marker_genes_in_data.items() if v}


    stem_cts = marker_genes_in_data.keys()
    marker_plots_dict = {}

    for ct in stem_cts:
        fig = sc.pl.umap(
            adata,
            color=marker_genes_in_data[ct],
            layer=layer,
            vmin=0,
            vmax="p99",
            sort_order=False,
            frameon=False,
            cmap=cmap,
            show=False,
            return_fig=True,
        )
        fig.suptitle(ct, fontsize=16, fontweight='bold')
        
        num_axes = len(fig.axes)  # Get the number of subplots

        if num_axes >= 5:
            for j in np.arange(1, 5, 1):
                fig.axes[j].set_ylabel("")
                fig.axes[j].set_xlabel("")
        elif num_axes >= 2:
            for j in np.arange(1, 2, 1):
                fig.axes[j].set_ylabel("")
                fig.axes[j].set_xlabel("")

        marker_plots_dict[ct] = fig
        plt.close(fig)

        if save_plot == True:
            ct = ct.replace(" ", "_")
            fig.savefig(f"{save_dir}/{prefix}{ct}_marker_genes.{file_ext}", bbox_inches="tight")
        
    return marker_plots_dict
