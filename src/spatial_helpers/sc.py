# Functions for processing single-cell data

import numpy as np
import random
import sys
import pandas as pd
import anndata as ad
import scanpy as sc
import copy
import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


### Preprocessing ------------------------------------------------------------------------------------------------------------

def set_all_seeds(seed=0):
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if 'numpy' in sys.modules:
        import numpy
        numpy.random.seed(seed)
    if 'random' in sys.modules:
        import random
        random.seed(seed)
    if 'scvi' in sys.modules:
        import scvi
        scvi.settings.seed = seed
    if 'tensorflow' in sys.modules:
        import tensorflow
        tensorflow.random.set_seed(seed)
    if 'torch' in sys.modules:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('high')
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def run_scvi(adata, key='scvi', batch_key='batch', layer='counts', n_layers=2, n_latent=30, labels_key=None, size_factor_key=None, categorical_covariate_keys=None, continuous_covariate_keys=None, get_expr=False, save=None, **kwargs):
    
    import torch
    import scvi
    import shutil
    from scipy.sparse import csr_matrix
    
    seed = 0
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('high')
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    scvi.settings.seed = seed
    
    scvi.model.SCVI.setup_anndata(adata,
                                  layer=layer,
                                  batch_key=batch_key,
                                  labels_key=labels_key,
                                  size_factor_key=size_factor_key,
                                  categorical_covariate_keys=categorical_covariate_keys,
                                  continuous_covariate_keys=continuous_covariate_keys)
    
    model = scvi.model.SCVI(adata, n_layers=n_layers, n_latent=n_latent, **kwargs)
    model.train()

    if save is not None:
        shutil.rmtree(save, ignore_errors=True)
        model.save(save)
        
    adata.obsm[key] = model.get_latent_representation()

    if get_expr is True:
        adata.layers[key] = csr_matrix(model.get_normalized_expression(transform_batch=None))

    return adata


def calc_pca(adata, key=None, layer=None, n_comps=None, use_highly_variable=None):

    X = None
    if layer is not None:
        if adata.X is not None:
            X = adata.X.copy()
        adata.X = adata.layers[layer].copy()
        
    pcaparams_orig = None
    pca_orig = None
    pcs_orig = None
    if 'pca' in adata.uns.keys():
        pcaparams_orig = adata.uns['pca'].copy()
    if 'X_pca' in adata.obsm.keys():
        pca_orig = adata.obsm['X_pca'].copy()
    if 'PCs' in adata.varm.keys():
        pcs_orig = adata.varm['PCs'].copy()
    
    sc.pp.pca(adata, n_comps=n_comps, use_highly_variable=use_highly_variable, svd_solver='arpack')

    if key is not None:
        adata.uns[key] = adata.uns['pca'].copy()
        adata.obsm[key] = adata.obsm['X_pca'].copy()
        adata.varm[key + '_PCs'] = adata.varm['PCs'].copy()
        if key != 'pca':
            del adata.uns['pca']
        if key != 'X_pca':
            del adata.obsm['X_pca']
        del adata.varm['PCs']
    
    if pcaparams_orig is not None:
        adata.uns['pca'] = pcaparams_orig
    if pca_orig is not None:
        adata.obsm['X_pca'] = pca_orig
    if pcs_orig is not None:
        adata.varm['PCs'] = pcs_orig

    if X is not None:
        adata.X = X.copy()

    return adata


def pp(adata, layer=None, key=None, n_neighbors=15, resolution=1, use_rep=None, n_comps=None, n_pcs=None, use_highly_variable=None, flavor='igraph', n_iterations=2, rank_method='wilcoxon', rank_raw = False):
    
    adata = adata.copy()
    X = None
    if adata.X is not None:
        X = adata.X.copy()
    if layer is None:
        layer = 'X'
        adata.layers[layer] = adata.X.copy()
    else:
        adata.X = adata.layers[layer]
    print('Layer:', layer)
    if key is None:
        key = layer + '_'

    if use_rep is None:
        use_rep = key + 'pca'
        adata = calc_pca(adata, key=use_rep, n_comps=n_comps, use_highly_variable=use_highly_variable)
    
    if use_rep is False:
        neighbors_key = layer + '_nb'
        n_pcs = 0
    else:
        neighbors_key = use_rep + '_nb'
    sc.pp.neighbors(adata, use_rep = None if use_rep is False else use_rep, n_pcs = n_pcs, n_neighbors = n_neighbors, key_added = neighbors_key)
    
    umap_key = neighbors_key + '_umap'
    sc.tl.umap(adata, neighbors_key = neighbors_key)
    adata.obsm[umap_key] = adata.obsm['X_umap'].copy()
    del adata.obsm['X_umap']
    adata.uns[umap_key] = adata.uns['umap'].copy()
    del adata.uns['umap']

    cluster_key = neighbors_key + '_leiden_'
    if not isinstance(resolution, list):
        resolution = [resolution]
    for res in resolution:
        sc.tl.leiden(adata, resolution = res, neighbors_key = neighbors_key, key_added = cluster_key + str(res), n_iterations = n_iterations) # flavor = flavor, 
        adata = rename_clusters_by_size(adata, cluster_key = cluster_key + str(res))
    
    for res in resolution:
        sc.tl.rank_genes_groups(adata, layer = layer, key_added = cluster_key + str(res) + '_rank', groupby = cluster_key + str(res), use_raw = rank_raw, method = rank_method)

    if X is not None:
        adata.X = X.copy()
    
    return adata


def rename_clusters_by_size(adata, cluster_key='leiden'):
    clusters = adata.obs[cluster_key]
    cluster_counts = clusters.value_counts()
    sorted_clusters = cluster_counts.sort_values(ascending=False).index
    new_labels = range(len(sorted_clusters))

    cluster_mapping = {old: new for old, new in zip(sorted_clusters, new_labels)}
    adata.obs[cluster_key] = clusters.map(cluster_mapping)
    adata.obs[cluster_key] = pd.Categorical(adata.obs[cluster_key], categories=new_labels, ordered=True)

    return adata


def rename_clusters_by_markers(adata, key='rank_genes_groups', n=None):
    result = {}
    if n is None:
        n = random.randrange(10, 30)
    data = adata.uns[key]['names']
    for i, field in enumerate(data.dtype.names):
        top_genes = [data[j][field] for j in range(n)]
        concatenated_genes = '-'.join(top_genes)
        result[field] = concatenated_genes
        
    return result


def add_suffixes(input_dict):
    value_count = {}
    updated_dict = {}
    for key, value in input_dict.items():
        if value not in value_count:
            value_count[value] = 1
            updated_dict[key] = value
        else:
            value_count[value] += 1
            updated_dict[key] = f"{value}_{value_count[value]}"
    
    return updated_dict


    
### Tangram ------------------------------------------------------------------------------------------------------------

def run_tangram(adata, refdata=None, tgmap=None, ref_label='celltype', threshold=0.9, prob_threshold=0, adata_layer=None, refdata_layer=None, genes=None, gene_to_lowercase=False, device='cuda'):

    import tangram as tg
    from pandas import DataFrame
    
    if tgmap is None:

        features = list(adata.var.index.intersection(refdata.var.index).values)
    
        if ref_label not in refdata.obs.columns:
            raise Exception('Reference label ' + ref_label + ' not found in refdata.obs!')
    
        if adata_layer is not None:
            adata.X = adata.layers[adata_layer]
    
        if refdata_layer is not None:
            refdata.X = refdata.layers[refdata_layer]
    
        tg.pp_adatas(refdata, adata, genes=genes, gene_to_lowercase=gene_to_lowercase)
    
        tgmap = tg.map_cells_to_space(refdata,
                                      adata,
                                      device=device,
                                      mode='clusters',
                                      density_prior='uniform',
                                      cluster_label=ref_label)

    tg.project_cell_annotations(tgmap, adata, annotation=ref_label, threshold=prob_threshold)

    tg_key = 'tangram_' + ref_label
    adata.obsm[tg_key] = adata.obsm['tangram_ct_pred'].copy()
    del adata.obsm['tangram_ct_pred']

    pred = get_tangram_predictions(adata, key=tg_key, threshold=threshold)
    adata.obs[tg_key] = pred['prediction'].copy()
    adata.obs[tg_key + '_score'] = adata.obsm[tg_key].max(axis=1)
    
    return tgmap, adata


def get_tangram_predictions(adata, key='tangram_ct_pred', threshold=0.9, bg_correct=False, bg_mad_thres=2):
    
    df = adata.obsm[key]
    celltype_names = list(df.columns)

    scores = df.to_numpy()

    if bg_correct is True:
        mad = np.median(np.absolute(scores - np.median(scores, axis=0)), axis=0)
        bg = np.median(scores, axis=0) + bg_mad_thres*mad
        scores = scores - bg
        scores[scores < 0] = 0

    min_scores = scores.min(axis=1, keepdims=True)
    max_scores = scores.max(axis=1, keepdims=True)
    scores_norm = (scores - min_scores) / (max_scores - min_scores)

    max_scores = scores_norm.max(axis=1)
    max_indices = np.argmax(scores_norm, axis=1)
    second_max_scores = np.where(scores_norm == max_scores[:, None], -np.inf, scores_norm).max(axis=1)

    predictions = []
    for idx in range(len(scores_norm)):
        top_celltypes = [celltype_names[i] for i in range(len(scores_norm[idx]))
                            if scores_norm[idx][i] == max_scores[idx] or scores_norm[idx][i] == second_max_scores[idx]]
        if max_scores[idx] - second_max_scores[idx] < threshold:
            predictions.append([celltype_names[max_indices[idx]]])
        else:
            predictions.append(top_celltypes)

    pred_df = pd.DataFrame({
        'prediction': [' & '.join(types) for types in predictions],
    })
    pred_df.index = df.index.copy()

    return pred_df


def get_tangram_annotation(adata, predictions_key='tangram', cluster_key='leiden', n_thres=10000, p_thres=0.4, rel=True):
    confmat = pd.crosstab(adata.obs[predictions_key], adata.obs[cluster_key])
    confmat_frac = confmat.div(confmat.sum(axis=0), axis=1)
    confmat = confmat[(confmat.sum(axis=1)>n_thres) | np.sum(p_thres > 0.5)]
    if rel is True:
        confmat = confmat.div(confmat.sum(axis=0), axis=1)
    relabel_dict = confmat.idxmax(axis=0).to_dict()
    return confmat, relabel_dict


def plot_composition(df, group_key, composition_key, colors=None, na_color='#ECECEC', fontsize=20, fontsize_xticks=None, fontsize_yticks=None, width=5, height=5, rel=True, observed=True, save=None):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    dfplot = (df.groupby([group_key, composition_key], observed=observed).size().unstack())

    if rel:
        dfplot = dfplot.div(dfplot.sum(axis=1), axis=0)

    if fontsize_xticks is None:
        fontsize_xticks = fontsize
    if fontsize_yticks is None:
        fontsize_yticks = fontsize

    if colors is not None:
        color_mapping = {category: colors.get(category, na_color) for category in dfplot.columns}
        colors = [color_mapping.get(col, '#ECECEC') for col in dfplot.columns]
    
    with mpl.rc_context({'font.size': fontsize, 
                         'axes.labelsize': fontsize,
                         'axes.titlesize': fontsize,
                         'xtick.labelsize': fontsize_xticks,
                         'ytick.labelsize': fontsize_yticks,
                         'legend.fontsize': fontsize,
                        }):
        fig, ax = plt.subplots(1, 1, figsize=(width, height))
        dfplot.plot(kind="barh", stacked=True, ax=ax, width=0.7, color=colors)
        legend = ax.legend(bbox_to_anchor=(1, 1), labelspacing=0.15, frameon=False, loc="upper left")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)
        if rel:
            ax.set_xlim(0,1)
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major')
        plt.tight_layout()

        if save is not None:
            plt.savefig(save, bbox_inches='tight', pad_inches=0, dpi=300)

        return ax

