# Functions for processing spatial data

import os
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import scanpy as sc
import spatialdata as sd
import spatialdata_io as sdio
import spatialdata_plot
from joblib import Parallel, delayed


### Spatial analysis functions ------------------------------------------------------------------------------------------------------------

def spatial_neighbors(adata, sample_key=None, coordinates_key='spatial', n_neighs=6, max_distance=None, delaunay=False, percentile=None, transform=None, set_diag=False, key_added=None):
    
    import squidpy as sq

    if key_added is None:
        if delaunay:
            key_added = 'delaunay'
        else:
            key_added = 'spatial_n' + str(n_neighs)
        if max_distance is not None:
            key_added = key_added + 'r' + str(max_distance)
    
    sq.gr.spatial_neighbors(adata,
                            spatial_key=coordinates_key,
                            coord_type='generic',
                            n_neighs=n_neighs,
                            delaunay=delaunay,
                            n_rings=1,
                            percentile=percentile,
                            transform=transform,
                            set_diag=set_diag,
                            key_added=key_added,
                            copy=False)

    if sample_key is not None:
        categories = adata.obs[sample_key].values
        adj_matrix = adata.obsp[adata.uns[key_added+'_neighbors']['connectivities_key']]
        row, col = adj_matrix.nonzero()
        for r, c in zip(row, col):
            if categories[r] != categories[c]:
                adj_matrix[r, c] = 0
                adj_matrix[c, r] = 0

        adata.obsp[adata.uns[key_added+'_neighbors']['connectivities_key']] = adj_matrix

    if max_distance is not None:
        adata = filter_neighbor_graph(adata, neigh_key=key_added, neigh_key_filt=key_added, max_distance=max_distance)

    print('Spatial neighbors graph key added: ' + key_added)

    return adata


def get_neighbors(adata, obs_key='celltype', neighbors_key='spatial_connectivities', n_jobs=1):

    celltypes = adata.obs[obs_key]
    unique_celltypes = celltypes.unique()
    num_cells = adata.obsp[neighbors_key].shape[0]
    celltype_to_idx = {ct: idx for idx, ct in enumerate(unique_celltypes)}
    
    def process_chunk(chunk_indices):
        chunk_results = np.zeros((len(chunk_indices), len(unique_celltypes)), dtype=int)
        for j, i in enumerate(chunk_indices):
            row = adata.obsp[neighbors_key][i]
            neighbor_indices = row.nonzero()[1]
            neighbor_celltypes = celltypes.iloc[neighbor_indices].values
            for ct in neighbor_celltypes:
                chunk_results[j, celltype_to_idx[ct]] += 1
        return chunk_indices, chunk_results

    indices = np.arange(num_cells)
    chunks = np.array_split(indices, n_jobs)
    parallel_results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk)(chunk) for chunk in chunks
    )
    
    neighbors = np.zeros((num_cells, len(unique_celltypes)), dtype=int)
    for chunk_indices, chunk_counts in parallel_results:
        neighbors[chunk_indices] = chunk_counts

    neighbors_df = pd.DataFrame(neighbors, columns=unique_celltypes, index=adata.obs.index)
    return neighbors_df


def filter_neighbor_graph(adata, neigh_key='spatial', neigh_key_filt='spatial_filt', max_distance=20):
    
    from scipy.sparse import csr_matrix
    
    connectivities = adata.obsp[neigh_key + '_connectivities']
    distances = adata.obsp[neigh_key + '_distances']
    
    row_idx, col_idx = connectivities.nonzero()
    dist_values = np.array(distances[row_idx, col_idx]).flatten()
    mask = dist_values <= max_distance
    
    filtered_row_idx = row_idx[mask].astype(np.int32)
    filtered_col_idx = col_idx[mask].astype(np.int32)
    filtered_distances = dist_values[mask].astype(np.float32)
    
    connectivities_filt = csr_matrix(
        (np.ones(len(filtered_row_idx), dtype=np.float32), (filtered_row_idx, filtered_col_idx)), 
        shape=connectivities.shape
    )
    
    distances_filt = csr_matrix(
        (filtered_distances, (filtered_row_idx, filtered_col_idx)), 
        shape=distances.shape
    )

    adata.obsp[neigh_key_filt + '_connectivities'] = connectivities_filt
    adata.obsp[neigh_key_filt + '_distances'] = distances_filt
    adata.uns[neigh_key_filt + '_neighbors'] = adata.uns[neigh_key + '_neighbors'].copy()
    adata.uns[neigh_key_filt + '_neighbors']['connectivities_key'] = neigh_key_filt + '_connectivities'
    adata.uns[neigh_key_filt + '_neighbors']['distances_key'] = neigh_key_filt + '_distances'
    adata.uns[neigh_key_filt + '_neighbors']['params']['radius'] = max_distance

    return adata
    

def filter_n_neighbors(adata, key='spatial_connectivities', min_neighbors=5):
    adata = adata[adata.obsp[key].sum(axis=1) >= min_neighbors].copy()
    return adata


def find_aggregates(adata, celltype, n_neighbors_key='celltype_neighbors', aggr_key='celltype_aggregate', obs_key='celltype', neighbors_key='spatial_connectivities', n_neighbors=None, max_iterations=None):
    if n_neighbors is None:
        n_neighbors = random.randrange(4, 50)
    if max_iterations is None:
        max_iterations = random.randrange(5, 15)
        
    adata.obs[aggr_key+'_seed'] = (adata.obs[obs_key] == celltype) & (adata.obs[n_neighbors_key] >= n_neighbors)
    
    adata.obs[aggr_key] = adata.obs[aggr_key+'_seed'].copy()
    for i in range(max_iterations):
        aggregate_indices = np.where(adata.obs[aggr_key].values)[0]
        if len(aggregate_indices) == 0:
            break
        neighbor_indices = np.unique(adata.obsp[neighbors_key][aggregate_indices].nonzero()[1])
        is_neighbor = adata.obs[obs_key].iloc[neighbor_indices] == celltype
        adata.obs[aggr_key].iloc[neighbor_indices[is_neighbor]] = True
    
    return adata


def check_adj_matrix(adj_matrix, df, category_column):

    from scipy.sparse import csr_matrix
    categories = df[category_column].values
    if not isinstance(adj_matrix, csr_matrix):
        adj_matrix = csr_matrix(adj_matrix)
    
    row, col = adj_matrix.nonzero()
    for r, c in zip(row, col):
        if categories[r] != categories[c]:
            return True
    
    return False


def filter_bg_genes(adata, genes, target_celltype, spatial_clusters='CN', expr_thres=0.75, corr_thres=0.5):
    
    expr = adata.layers['norm']
    spatial_clusters_data = adata.obs[spatial_clusters]

    gene_names = adata.var.index
    aggr_counts = {}
    for celltype in adata.obs['celltype'].unique():
        celltype_expr = expr[adata.obs['celltype'] == celltype, :].mean(axis=0)
        aggr_counts[celltype] = np.asarray(celltype_expr).flatten()
    aggr_counts = pd.DataFrame(aggr_counts, index=gene_names)

    corr_matrix = aggr_counts.corr(method='spearman')
    genes_to_remove = []
    for gene in genes:
        target_gene_expr = aggr_counts.loc[gene, target_celltype]
        for celltype in aggr_counts.columns:
            if celltype == target_celltype:
                continue
            other_gene_expr = aggr_counts.loc[gene, celltype]
            if target_gene_expr < (expr_thres * other_gene_expr):
                gene_corr = corr_matrix.loc[target_celltype, celltype]
                if gene_corr > corr_thres:
                    genes_to_remove.append(gene)
                    break
    filtered_genes = [gene for gene in genes if gene not in genes_to_remove]
    return filtered_genes


### NicheCompass ------------------------------------------------------------------------------------------------------------

def nc_prepare_gps(omnipath=True, nichenet=True, species='human', dir='nichecompass', load_from_disk=None, save_to_disk=True):

    from nichecompass.models import NicheCompass
    from nichecompass.utils import (add_gps_from_gp_dict_to_adata,
                                    compute_communication_gp_network,
                                    visualize_communication_gp_network,
                                    create_new_color_dict,
                                    extract_gp_dict_from_nichenet_lrt_interactions,
                                    extract_gp_dict_from_omnipath_lr_interactions,
                                    filter_and_combine_gp_dict_gps_v2,
                                    generate_enriched_gp_info_plots)

    os.makedirs(os.path.join(dir, 'prior_data'), exist_ok=True)

    if omnipath is True:
        lr_network_file_path = os.path.join(dir, 'prior_data', 'omnipath_lr_network.csv')
        gene_orthologs_mapping_file_path = None

        if load_from_disk is None:
            if os.path.exists(lr_network_file_path):
                load_from_disk = True
                print('Loading omnipath from disk.')
        
        omnipath_gp_dict = extract_gp_dict_from_omnipath_lr_interactions(species=species,
                                                                         save_to_disk=save_to_disk,
                                                                         load_from_disk=load_from_disk,
                                                                         lr_network_file_path=lr_network_file_path,
                                                                         gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,
                                                                         plot_gp_gene_count_distributions=False)

    if nichenet is True:
        nichenet_lr_network_file_path = os.path.join(dir, 'prior_data', 'nichenet_lr_network_v2.csv')
        nichenet_ligand_target_matrix_file_path = os.path.join(dir, 'prior_data', 'nichenet_ligand_target_matrix_v2_.csv')
        gene_orthologs_mapping_file_path = os.path.join(dir, 'prior_data', 'human_mouse_gene_orthologs.csv')
        
        if load_from_disk is None:
                if os.path.exists(nichenet_lr_network_file_path) & os.path.exists(nichenet_ligand_target_matrix_file_path):
                    load_from_disk = True
        
        nichenet_gp_dict =  extract_gp_dict_from_nichenet_lrt_interactions(species=species,
                                                                          version='v2',
                                                                          keep_target_genes_ratio=1.0,
                                                                          max_n_target_genes_per_gp=250,
                                                                          load_from_disk=load_from_disk,
                                                                          save_to_disk=save_to_disk,
                                                                          lr_network_file_path=nichenet_lr_network_file_path,
                                                                          ligand_target_matrix_file_path=nichenet_ligand_target_matrix_file_path,
                                                                          gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,
                                                                          plot_gp_gene_count_distributions=False)

    gp_dicts = [omnipath_gp_dict, nichenet_gp_dict]
    combined_gp_dict = filter_and_combine_gp_dict_gps_v2(gp_dicts, verbose=False)
        
    return combined_gp_dict


def nc_run(adata,
           gp_dict,
           adj_key='spatial_connectivities',
           layer=None,
           batch_keys=None,
           conv_layer_encoder='gcnconv',
           n_epochs=400,
           n_epochs_all_gps=25,
           lr=0.001,
           lambda_edge_recon=500000.,
           lambda_gene_expr_recon=300.,
           lambda_l1_masked=0., 
           lambda_l1_addon=30., 
           edge_batch_size=4096, 
           n_sampled_neighbors=4,
           latent_dtype=np.float16,
           genes_uppercase=True):

    from nichecompass.models import NicheCompass
    from nichecompass.utils import (add_gps_from_gp_dict_to_adata,
                                    compute_communication_gp_network,
                                    visualize_communication_gp_network,
                                    create_new_color_dict,
                                    extract_gp_dict_from_mebocost_ms_interactions,
                                    extract_gp_dict_from_nichenet_lrt_interactions,
                                    extract_gp_dict_from_omnipath_lr_interactions,
                                    filter_and_combine_gp_dict_gps_v2,
                                    generate_enriched_gp_info_plots)

    add_gps_from_gp_dict_to_adata(
        gp_dict=gp_dict,
        adata=adata,
        genes_uppercase=genes_uppercase,
        gp_targets_mask_key = 'nichecompass_gp_targets',
        gp_targets_categories_mask_key = 'nichecompass_gp_targets_categories',
        targets_categories_label_encoder_key = 'nichecompass_targets_categories_label_encoder',
        gp_sources_mask_key = 'nichecompass_gp_sources',
        gp_sources_categories_mask_key = 'nichecompass_gp_sources_categories',
        sources_categories_label_encoder_key = 'nichecompass_sources_categories_label_encoder',
        gp_names_key = 'nichecompass_gp_names',
        source_genes_idx_key = 'nichecompass_source_genes_idx',
        target_genes_idx_key = 'nichecompass_target_genes_idx',
        genes_idx_key = 'nichecompass_genes_idx',
        min_genes_per_gp = 1,
        min_source_genes_per_gp = 0,
        min_target_genes_per_gp = 0,
        max_genes_per_gp = None,
        max_source_genes_per_gp = None,
        max_target_genes_per_gp = None,
        filter_genes_not_in_masks = False,
        add_fc_gps_instead_of_gp_dict_gps = False,
        plot_gp_gene_count_distributions = False)

    model = NicheCompass(adata,
                         counts_key=layer,
                         adj_key=adj_key,
                         gp_names_key='nichecompass_gp_names',
                         cat_covariates_keys=batch_keys,
                         active_gp_names_key='nichecompass_active_gp_names',
                         gp_targets_mask_key='nichecompass_gp_targets',
                         gp_targets_categories_mask_key='nichecompass_gp_targets_categories',
                         gp_sources_mask_key='nichecompass_gp_sources',
                         gp_sources_categories_mask_key='nichecompass_gp_sources_categories',
                         latent_key='nichecompass_latent',
                         conv_layer_encoder=conv_layer_encoder,
                         active_gp_thresh_ratio= 0.01,
                         seed=0)

    model.train(n_epochs=n_epochs, 
                n_epochs_all_gps=n_epochs_all_gps,
                lr=lr,
                lambda_edge_recon=lambda_edge_recon,
                lambda_gene_expr_recon=lambda_gene_expr_recon,
                lambda_l1_masked=lambda_l1_masked,
                edge_batch_size=edge_batch_size,
                n_sampled_neighbors=n_sampled_neighbors,
                use_cuda_if_available=True,
                latent_dtype=latent_dtype,
                verbose=False)

    return model



### Spatialdata utilities ------------------------------------------------------------------------------------------------------------

def export_xenium_explorer(adata, obs_key, file, cells=None):
    if cells is not None:
        df = pd.read_csv(cells)
        obs = adata.obs.loc[adata.obs.index.isin(df['cell_id'])]
    else:
        obs = adata.obs
    result = obs[[obs_key]].reset_index()
    result.rename(columns={'index': 'cell_id'}, inplace=True)
    result.rename(columns={obs_key: 'group'}, inplace=True)
    result.to_csv(file, index=False)
    

def load_samples(sample_dict, samplesdir, del_elements=[], n_jobs=1):
    
    def loader(sample, samplesdir, del_elements):
        sample_path = os.path.join(samplesdir, sample_dict[sample] + '.zarr')
        sdata = sd.read_zarr(sample_path)
        sdata['table'].obs.set_index('cell_id', drop=False, inplace=True)
        sdata['table'].obs['name'] = sample
        for elem in del_elements:
            del sdata[elem]
        sdata = prefix_elements(sdata, prefix=sample + '_', elements=['images', 'labels', 'shapes', 'points'])
        sdata = match_ids(sdata, [sample + '_cell_boundaries'], table_key='table')
        region_key = sdata['table'].uns['spatialdata_attrs']['region_key']
        sdata['table'].obs[region_key] = sample + '_cell_boundaries'
        sdata.set_table_annotates_spatialelement('table', region_key=region_key, region=sample + '_cell_boundaries')
        return sample, sdata

    samples = Parallel(n_jobs=n_jobs)(
        delayed(loader)(sample, samplesdir, del_elements) for sample in sample_dict
    )
    samples = {sample: sdata for sample, sdata in samples}
    return samples


def match_ids(sdata, elements, table_key='table'):

    from functools import reduce

    ix_table = sdata.tables[table_key].obs.index
    ix_elem = []
    for elem in elements:
        ix_elem.append(sdata[elem].index)
    ix_elem.append(ix_table)
    ix = reduce(set.intersection, map(set, ix_elem))
    ix = pd.Index(ix).drop_duplicates()
    sdata.tables[table_key] = sdata.tables[table_key][ix]
    
    for elem in elements:
        sdata[elem] = sdata[elem].loc[ix]

    return sdata


def concat_shapes(sdata, elem, samples):
    elems = {}
    for sample in samples:
        prefix = sample+'_'
        elems[sample] = sdata[prefix + elem]
    return pd.concat(elems)


def concat_points(sdata, elem, samples):
    import dask.dataframe as dd
    elems = {}
    for sample in samples:
        prefix = sample+'_'
        elems[sample] = sdata[prefix + elem]
    return dd.concat(list(elems.values()))


def concat_images(sdata, elem, samples, concat_dim='sample'):
    import xarray as xr
    elems = {}
    for sample in samples:
        prefix = sample+'_'
        elems[sample] = sdata[prefix + elem]
    merged = xr.merge(list(elems.values()), fill_value=0, compat='override')
    merged = merged[list(merged.data_vars)[0]]
    return merged
   

def apply_coord_transformation(sdata, elements, translation=None, rotation=None, target_coordinate_system='transformed', center=False, keep_prior=True, prior_coordinate_system='global', ref_elem=None):
    import math
    from spatialdata.transformations import (
        Affine,
        Identity,
        MapAxis,
        Scale,
        Sequence,
        Translation,
        get_transformation,
        get_transformation_between_coordinate_systems,
        set_transformation,
    )

    if translation is not None:
        tx, ty = translation
    else:
        tx = 0
        ty = 0

    if rotation is not None:
        theta = math.radians(rotation)
    else:
        theta = 0
    
    transformation = Affine(
        [
            [math.cos(theta), -math.sin(theta), tx],
            [math.sin(theta), math.cos(theta), ty],
            [0, 0, 1],
        ],
        input_axes=('x', 'y'),
        output_axes=('x', 'y'),
    )

    prior = Identity()
    if (translation is not None) or (rotation is not None):
        for elem in elements:
            if keep_prior is True:
                prior = get_transformation(sdata[elem], to_coordinate_system=prior_coordinate_system)
            set_transformation(sdata[elem], transformation=Sequence([prior, transformation]), to_coordinate_system=target_coordinate_system)
            sdata[elem] = sd.transform(sdata[elem], to_coordinate_system=target_coordinate_system)
        
    if center is True:
        extent = sd.get_extent(sdata[ref_elem], coordinate_system=target_coordinate_system)
        x_min = -extent['x'][0]
        y_min = -extent['y'][0]
        shift = Translation([x_min, y_min], axes=('x', 'y'))

        for elem in elements:
            print('Centering ' + elem + ' by [' + str(x_min) + ', ' + str(y_min) +']')
            prior = get_transformation(sdata[elem], to_coordinate_system=target_coordinate_system)
            set_transformation(sdata[elem], transformation=Sequence([prior, shift]), to_coordinate_system=target_coordinate_system)
            sdata[elem] = sd.transform(sdata[elem], to_coordinate_system=target_coordinate_system)
            
    return sdata


def par_apply_coord_transformation(samples, elements, rotations=None, translations=None, target_coordinate_system='transformed', center=False, keep_prior=True, prior_coordinate_system='global', ref_elem='morphology_focus', n_jobs=1):

    def par_apply_fun(sample_key, samples, elements, rotations, translations, target_coordinate_system, center, keep_prior, prior_coordinate_system, ref_elem):
        print(sample_key)
        prefix = sample_key + '_'
        elems = [prefix + elem for elem in elements]
        res = apply_coord_transformation(
            sdata = samples[sample_key],
            elements = elems,
            translation = translations[sample_key] if translations is not None else None,
            rotation = rotations[sample_key] if rotations is not None else None,
            target_coordinate_system=target_coordinate_system,
            center=center,
            keep_prior=keep_prior,
            prior_coordinate_system=prior_coordinate_system,
            ref_elem=prefix + 'morphology_focus',
        )
        return sample_key, res
    
    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(par_apply_fun)(sample_key, samples, elements, rotations, translations, target_coordinate_system, center, keep_prior, prior_coordinate_system, ref_elem)
        for sample_key in samples.keys()
    )
    return {key: value for key, value in results}


def compute_transformations(layout, sample_dims, spacing=(0.05, 0.05)):

    all_widths = [dims[0] for dims in sample_dims.values()]
    all_heights = [dims[1] for dims in sample_dims.values()]
    avg_width = sum(all_widths) / len(all_widths)
    avg_height = sum(all_heights) / len(all_heights)
    x_spacing = spacing[0] * avg_width
    y_spacing = spacing[1] * avg_height

    transformations = {}
    global_x_min = float('inf')
    global_y_min = float('inf')
    current_y = 0 

    for row in layout:
        row_width = 0
        row_heights = []
        for sample_name in row:
            if sample_name:
                if sample_name not in sample_dims:
                    raise KeyError(f"Sample '{sample_name}' not found in dict")
                width, height = sample_dims[sample_name]
            else:
                width, height = avg_width, avg_height
            row_width += width
            row_heights.append(height)
        row_width += (len(row) - 1) * x_spacing
        row_height = max(row_heights) if row_heights else 0
        current_x = -row_width / 2

        for sample_name in row:
            if sample_name:
                width, height = sample_dims[sample_name]
            else:
                width, height = avg_width, avg_height

            vertical_offset = (row_height - height) / 2
            tx = current_x
            ty = current_y + vertical_offset

            if sample_name:
                transformations[sample_name] = (tx, ty)
                global_x_min = min(global_x_min, tx)
                global_y_min = min(global_y_min, ty)
            current_x += width + x_spacing
        current_y += row_height + y_spacing

    shift_x = -global_x_min if global_x_min < 0 else 0
    shift_y = -global_y_min if global_y_min < 0 else 0
    for sample in transformations:
        tx, ty = transformations[sample]
        transformations[sample] = (tx + shift_x, ty + shift_y)

    return transformations


def shift_image_coords(elem, ref_elem, coordinate_system=None):
    ext = sd.get_extent(ref_elem, coordinate_system=coordinate_system)
    return elem.assign_coords(x=elem.coords['x'] + ext['x'][0], y=elem.coords['y'] + ext['y'][0])


def rotate_element_coords(sdata, elem, angle=0, target_coordinate_system='rotated', center=True):
    
    import math
    from spatialdata.transformations import (
        Affine,
        Identity,
        MapAxis,
        Scale,
        Sequence,
        Translation,
        get_transformation,
        get_transformation_between_coordinate_systems,
        set_transformation,
    )

    theta = math.radians(angle)

    if center is True:
        x_center = sdata[elem].shape[1]/2
        y_center = sdata[elem].shape[1]/2
    else:
        x_center = 0
        y_center = 0
    
    rotation = Affine(
        [
            [math.cos(theta), -math.sin(theta), x_center * (1 - math.cos(theta)) + y_center * math.sin(theta)],
            [math.sin(theta), math.cos(theta), y_center * (1 - math.cos(theta)) - x_center * math.sin(theta)],
            [0, 0, 1],
        ],
        input_axes=('x', 'y'),
        output_axes=('x', 'y'),
    )

    set_transformation(sdata[elem], transformation=rotation, to_coordinate_system=target_coordinate_system) 
    new_elem = sd.transform(sdata[elem], to_coordinate_system=target_coordinate_system)

    if center is True:
        extent = sd.get_extent(new_elem, coordinate_system=target_coordinate_system)
        x_min = -extent['x'][0]
        y_min = -extent['y'][0]

        shift = Translation([x_min, y_min], axes=('x', 'y'))
        set_transformation(sdata[elem], transformation=Sequence([rotation, shift]), to_coordinate_system=target_coordinate_system) 
        new_elem = sd.transform(sdata[elem], to_coordinate_system=target_coordinate_system)

    return new_elem


def transform_geodf(gdf, rotation_angle=0, at_origin=True):
    
    from shapely.affinity import rotate, translate

    if at_origin:
        bounds = gdf.total_bounds
        gdf['geometry'] = gdf['geometry'].apply(
            lambda geom: translate(geom, xoff=-bounds[0], yoff=-bounds[1])
        )

    if rotation_angle != 0:
        # gdf_valid = gdf['geometry'].apply(lambda geom: geom.buffer(0))
        gdf['geometry'] = gdf['geometry'].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)
        cent = gdf['geometry'].union_all().centroid
        gdf['geometry'] = gdf['geometry'].apply(
            lambda geom: rotate(geom, rotation_angle, origin=[cent.x, cent.y], use_radians=False)
    )

    return gdf


def transform_xenium_coords(sdata, elements_px=['morphology_focus', 'cell_labels', 'nucleus_labels', 'sample_area'], elements_µm=['cell_circles','transcripts','cell_boundaries','nucleus_boundaries'], coordinate_system_px='global', coordinate_system_µm='global_µm', pixel_size=0.2125):
    
    from spatialdata.transformations import (
        Affine,
        Identity,
        MapAxis,
        Scale,
        Sequence,
        Translation,
        get_transformation,
        get_transformation_between_coordinate_systems,
        set_transformation,
    )

    scale_px_to_µm = Scale([pixel_size, pixel_size], axes=('x','y'))
    scale_µm_to_px = Scale([1/pixel_size, 1/pixel_size], axes=('x','y'))

    for elem in elements_px:
        set_transformation(sdata[elem], transformation=Identity(), to_coordinate_system=coordinate_system_px) # set
        set_transformation(sdata[elem], transformation=scale_px_to_µm, to_coordinate_system=coordinate_system_µm) # transform to µm

    for elem in elements_µm:
        set_transformation(sdata[elem], transformation=Identity(), to_coordinate_system=coordinate_system_µm) # add
        set_transformation(sdata[elem], transformation=scale_µm_to_px, to_coordinate_system=coordinate_system_px) # transform to px

    return sdata


def get_enclosing_rect(sdata, name='enclosing', coordinate_system='global'):
    
    from shapely.geometry import Polygon
    from spatialdata.models import ShapesModel

    ext = sd.get_extent(sdata, coordinate_system=coordinate_system)
    xmin = ext['x'][0]
    xmax = ext['x'][1]
    ymin = ext['y'][0]
    ymax = ext['y'][1]
    
    rect = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
    gdf = gpd.GeoDataFrame(pd.DataFrame({"name": [name]}), geometry=[rect], crs="EPSG:4326")
    
    return ShapesModel.parse(gdf)


def shapes_to_cells(sdata, shapes, key='ROI', pixel_size=0.2125, n_cores=1):

    from shapely.geometry import Point

    cell_coords = sdata['table'].obsm['spatial'].copy()
    if isinstance(cell_coords, np.ndarray):
        cell_coords = pd.DataFrame(cell_coords, columns=["x", "y"])
    cell_coords = cell_coords / pixel_size
    cell_coords.reset_index(drop=True, inplace=True)

    ROI_shapes = sdata.shapes[shapes]
    ROIs = ROI_shapes['name'].unique()
    
    sdata['table'].obs[key] = None
    coord_chunks = np.array_split(cell_coords, n_cores*2)

    def is_within_polygon(x, y, polygons):
        point = Point(x, y)
        return any(point.within(polygon))
        
    def process_chunk(chunk, polygon):
        chunk_res = chunk.apply(lambda row: is_within_polygon(row['x'], row['y'], polygon), axis=1)
        return chunk_res
        
    for ROI in ROIs:
        polygon = ROI_shapes[ROI_shapes['name'] == ROI].geometry
        res_parts = Parallel(n_jobs=n_cores)(delayed(process_chunk)(chunk, polygon) for chunk in coord_chunks)
        res = pd.concat(res_parts).sort_index()

        check_values = sdata['table'].obs.loc[res, key]
        if check_values.notna().any():
            raise ValueError(f"Cannot assign multiple values to column '{key}'.")
        
        sdata['table'].obs.loc[res, key] = ROI
    
    return sdata

    
def subset_shapes(sdata, shapes, combined=False, coordinate_system='global', n_jobs=1):

    from shapely import unary_union
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.affinity import scale
    
    ROI_shapes = sdata.shapes[shapes]

    if combined == True:
        scaling_factor= 10
        ROI_shapes = [scale(polygon, xfact=scaling_factor, yfact=scaling_factor, origin='centroid') for polygon in ROI_shapes['geometry'].tolist()]
        multi_polygon = unary_union(ROI_shapes)
        ROIs_sdata = sd.polygon_query(sdata, polygon=multi_polygon, target_coordinate_system=coordinate_system)

    else:
        ROIs = ROI_shapes['name'].unique()
        ROIs_sdata = {}
        def process_ROI(ROI, ROI_shapes, sdata, coordinate_system):
            polygon = ROI_shapes[ROI_shapes['name'] == ROI].geometry.iloc[0]
            query = sd.polygon_query(sdata, polygon=polygon, target_coordinate_system=coordinate_system)
            return ROI, query
        results = Parallel(n_jobs=n_jobs)(delayed(process_ROI)(ROI, ROI_shapes, sdata, coordinate_system) for ROI in ROIs)
        ROIs_sdata = {ROI: query for ROI, query in results}
    
    return ROIs_sdata


def save_subset_shapes(sdata, shapes, samplesdir, coordinate_system='global', n_jobs=1):

    ROI_shapes = sdata.shapes[shapes]
    ROIs = ROI_shapes['name'].unique()
    # ROIs_sdata = {}

    def process_ROI(ROI, ROI_shapes, sdata, coordinate_system):
        polygon = ROI_shapes[ROI_shapes['name'] == ROI].geometry.iloc[0]
        ROI_sdata = sd.polygon_query(sdata, polygon=polygon, target_coordinate_system=coordinate_system)
        ROI_sdata.write(os.path.join(samplesdir, ROI + '.zarr'), overwrite=True)
        del ROI_sdata
        return ROI
    
    results = Parallel(n_jobs=n_jobs, prefer='threads')(delayed(process_ROI)(ROI, ROI_shapes, sdata, coordinate_system) for ROI in ROIs)


def import_shapes(csvfile, pixel_size=0.2125, coordinate_system='global'):
    # NOTE: The following reader can also be used, but doesn't support named FOVs
    # import spatialdata_io as sdio
    # polygons = sdio.xenium_explorer_selection(csvfile, pixel_size = pixel_size, return_list = True)
    # gdf = GeoDataFrame(geometry=polygons)
    # ROIs = ShapesModel.parse(gdf, transformations={coordinate_system: Identity()})
    
    from shapely.geometry import Polygon
    from geopandas import GeoDataFrame
    from spatialdata.models import ShapesModel
    from spatialdata.transformations import Identity
    
    def numpy_to_shapely(x: np.array) -> Polygon:
        return Polygon(list(map(tuple, x)))

    df = pd.read_csv(csvfile, skiprows=[0, 1]) # in µm
    ROIs = [(name, numpy_to_shapely(group[['X', 'Y']].to_numpy() / pixel_size)) for name, group in df.groupby('Selection')]
    ROIs = GeoDataFrame({"name": [name for name, _ in ROIs], "geometry": [geom for _, geom in ROIs]})
    ROIs = ShapesModel.parse(ROIs, transformations={coordinate_system: Identity()})

    return ROIs


def get_sample_dim(sdata, elem):
    xdim = sdata[elem].shape[1]
    ydim = sdata[elem].shape[2]
    return xdim, ydim
    

def sd_rename(sdata, oldname, newname):
    sdata[newname] = sdata[oldname].copy()
    del sdata[oldname]


def prefix_elements(sdata, elements = ['images', 'points', 'labels', 'shapes', 'tables'], prefix='', suffix=''):
    if 'images' in elements and hasattr(sdata, 'images'):
        for elem in list(sdata.images):
            sd_rename(sdata, elem, prefix + elem + suffix)
    if 'points' in elements and hasattr(sdata, 'points'):
        for elem in list(sdata.points):
            sd_rename(sdata, elem, prefix + elem + suffix)
    if 'labels' in elements and hasattr(sdata, 'labels'):
        for elem in list(sdata.labels):
            sd_rename(sdata, elem, prefix + elem + suffix)
    if 'shapes' in elements and hasattr(sdata, 'shapes'):
        for elem in list(sdata.shapes):
            sd_rename(sdata, elem, prefix + elem + suffix)
    if 'tables' in elements and hasattr(sdata, 'tables'):
        for elem in list(sdata.tables):
            sd_rename(sdata, elem, prefix + elem + suffix)

    return sdata


def fix_cells_without_nuclei(sdata):
    
    # bugfix from: https://github.com/scverse/spatialdata/discussions/657
    
    CELL_CIRCLES = "cell_circles"
    TABLE = "table"
    CELL_ID = "cell_id"
    
    assert CELL_CIRCLES in sdata.shapes, f"Expected {CELL_CIRCLES} in the spatial data."
    assert TABLE in sdata.tables, f"Expected {TABLE} in the spatial data."
    
    circles = sdata[CELL_CIRCLES]
    table = sdata[TABLE]
    
    radii = circles.radius
    assert circles.radius.isna().sum() > 0, "There is no NaN value in the radii column that we can correct."
    
    assert np.array_equal(
        circles.index.to_numpy(), table.obs[CELL_ID].to_numpy()
    ), "The indices of the circles and the table do not match, please adjust to your data."
    
    table.obs.set_index(CELL_ID, inplace=True, drop=False)
    
    original_cell_radii = (table.obs.cell_area / np.pi) ** 0.5
    original_nucleus_radii = (table.obs.nucleus_area / np.pi) ** 0.5
    
    nan_mask = circles.radius.isna()
    
    assert np.allclose(
        circles.radius[~nan_mask], original_nucleus_radii.iloc[np.where(~nan_mask)[0]]
    ), "The non-NaN values in the radii column do not match the nucleus radii as it would be expected."
    circles.radius = original_cell_radii
    
    CELL_CIRCLES_CORRECTED = f"{CELL_CIRCLES}_corrected"
    sdata[CELL_CIRCLES_CORRECTED] = circles

    return sdata

