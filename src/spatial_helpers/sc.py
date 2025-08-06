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


def marker_dotplot(adata, group_by='celltype', rank_key='rank_genes_groups', group=None, genes=None, log2FC_thres=1, padj_thres=1, font_size=10, n_genes=8, swap_axes=True, figsize=(12, 10), dpi=400, save=None):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if genes is not None:
        de_genes=genes
    else:
        df = sc.get.rank_genes_groups_df(adata, group=group, key=rank_key)
        df['neglfc'] = -np.abs(df.logfoldchanges)
        df = df.sort_values(by=['pvals', 'neglfc'])
        del df['neglfc']
        df.rename(columns={'names': 'gene', 'logfoldchanges': 'log2FC', 'pvals': 'pvalue', 'pvals_adj': 'padj'}, inplace=True)
        df = df[df['log2FC'] > log2FC_thres]
        df = df[df['padj'] <= padj_thres]
        de_genes=df.gene[0:n_genes]
    
    with mpl.rc_context({'font.size': font_size, 
                         'axes.labelsize': font_size,
                         'axes.titlesize': font_size,
                         'xtick.labelsize': font_size,
                         'ytick.labelsize': font_size,
                        }):
        plt.figure(figsize=figsize)
        fig = sc.pl.dotplot(adata, layer='norm', var_names=de_genes, groupby=group_by, dendrogram=False, swap_axes=swap_axes, return_fig=True)
        fig.legend(colorbar_title='mean expression', size_title='% of cells ', width=1.7)
        if save is not None:
            fig.savefig(save, bbox_inches='tight', dpi=dpi)
        
    return list(de_genes)


def run_pydeseq2(adata, formula='~ *CONTRAST*', contrasts={'trt_vs_ctrl': ['condition', 'trt', 'ctrl']}, aggregate_by='patient', n_random_split=1, min_cells=20, layer=None, filter_genes=None, alpha = 0.05, min_rep_refit=7, inference=None, sort=True, n_jobs=1):
    
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    results = {}
    conditions_unique = set(contrast[0] for contrast in contrasts.values())

    dds_fit = {}
    for cond in conditions_unique:
        formula = formula.replace('*CONTRAST*', cond)
        counts, metadf = aggregate_counts(adata, group_by=cond, aggregate_by=aggregate_by, layer=layer, n_random_split=n_random_split, min_cells=min_cells)
        dds = DeseqDataSet(adata=None, counts=counts.astype(int), design=formula, metadata=metadf, min_replicates=min_rep_refit, quiet=True, n_cpus=n_jobs)
        dds.deseq2()
        dds_fit[cond] = dds

    if inference is None:
        inference = DefaultInference(n_cpus=n_jobs)
        
    for cont_name, cont in contrasts.items():
        ds = DeseqStats(dds_fit[cont[0]], contrast=cont, alpha=alpha, cooks_filter=True, independent_filter=True, quiet=True, inference=inference)
        ds.summary()
        df = ds.results_df
        if sort:
            df = df.loc[df.abs().sort_values(by=['pvalue', 'log2FoldChange']).index]
        results[cont_name] = df

    return results, dds_fit


def aggregate_counts(adata, group_by='condition', aggregate_by='patient', layer=None,
                     n_random_split=1, min_cells=10, fun='sum', n_jobs=1, prefer='threads', batch_size=None):

    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from joblib import Parallel, delayed

    if group_by not in adata.obs.columns or aggregate_by not in adata.obs.columns:
        raise ValueError("Both 'group_by' and 'aggregate_by' must be columns in adata.obs")

    if layer is not None:
        counts = adata.layers[layer]
    else:
        counts = adata.X

    obs_df = adata.obs.copy()
    var_names = adata.var_names.to_list()
    groups_all = obs_df.groupby([group_by, aggregate_by], observed=False).size()
    groups = groups_all[groups_all >= min_cells].index.tolist()

    if batch_size is None:
        batch_size = max(1, len(groups) // (n_jobs * 4))

    def aggregate_group_counts(group, obs_df, counts):
        condition, patient = group
        group_cells = (obs_df[group_by] == condition) & (obs_df[aggregate_by] == patient)
        ix = np.where(group_cells)[0]

        if n_random_split > 1:
            np.random.shuffle(ix)
            ix = np.array_split(ix, n_random_split)
        else:
            ix = [ix]

        local_counts = []
        local_metadata = []
        local_names = []

        for i, split in enumerate(ix):
            if len(split) == 0:
                continue

            if sp.issparse(counts):
                data = counts[split].sum(axis=0).A1 if fun == 'sum' else counts[split].mean(axis=0).A1
            else:
                data = np.sum(counts[split], axis=0) if fun == 'sum' else np.mean(counts[split], axis=0)

            local_counts.append(data)
            group_name = f"{condition}_{patient}"
            if n_random_split > 1:
                group_name = f"{group_name}_{i}"
            local_names.append(group_name)

            metadata_entry = obs_df.iloc[split].copy()
            metadata_entry = metadata_entry.loc[:, metadata_entry.nunique() == 1].iloc[0]
            metadata_entry[group_by] = condition
            metadata_entry[aggregate_by] = f"{patient}_{i}" if n_random_split > 1 else patient
            local_metadata.append(metadata_entry)

        return local_counts, local_names, local_metadata

    results = Parallel(n_jobs=n_jobs, prefer=prefer, batch_size=batch_size)(
        delayed(aggregate_group_counts)(group, obs_df, counts) for group in groups
    )

    aggregated_counts = []
    group_names = []
    aggregated_metadata = []

    for counts_list, names_list, metadata_list in results:
        aggregated_counts.extend(counts_list)
        group_names.extend(names_list)
        aggregated_metadata.extend(metadata_list)

    aggregated_counts_df = pd.DataFrame(aggregated_counts, columns=var_names)
    aggregated_counts_df.index = group_names

    aggregated_metadata_df = pd.DataFrame(aggregated_metadata)
    aggregated_metadata_df.index = group_names

    return aggregated_counts_df, aggregated_metadata_df


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

