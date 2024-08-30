from ._colors import (
    altair_scale,
    altair_scale_mpl,
    plot_all_palettes,
    plot_palette,
    set_scale_anndata,
)
from ._qc_plots import (
    umap_covariates,
    umap_marker_genes,
    umap_qc_metrics,
    violin_qc_metrics
)
from ._stat_plots import (
    swarm_box_corr
)

__all__ = [
    "swarm_box_corr",
    "umap_qc_metrics",
    "violin_qc_metrics",
    "umap_covariates",
    "umap_marker_genes",
    "set_scale_anndata",
    "altair_scale",
    "altair_scale_mpl",
    "plot_palette",
    "plot_all_palettes",
]
