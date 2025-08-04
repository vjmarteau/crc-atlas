# Functions for visualizing spatial data

import pandas as pd
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import spatialdata as sd
import spatialdata_plot

def spatialplot(sdata,
                shape=None,
                points=None,
                image=None,
                image_cmap=None,
                image_palette=None,
                image_alpha=1.0,
                image_channel=None,
                obs_key=None,
                feature=None,
                layer=None,
                table_key='table',
                title=None,
                outline_width=0.5,
                outline_color='#ffffff',
                outline_alpha=0.0,
                fill_alpha=1,
                coordinate_system='global',
                palette=None,
                points_palette=None,
                points_size=0.2,
                points_color='feature_name',
                points_key='transcripts',
                points_cmap=None,
                na_color='#cccccc',
                groups=None,
                rectangle=None,
                polygon=None,
                cmap=None, 
                cmap_q=None,
                cmap_clip=None,
                ax=None,
                xlabel=None,
                ylabel=None,
                ax_linewidth=None,
                ax_anchor='W',
                figsize=(None,8),
                legend_loc=None,
                legend_spacing=0.3,
                fontsize=30,
                legend_markerscale=3,
                bbox_to_anchor=None,
                legend_prop={'size': 15},
                dpi=None,
                rasterize_shapes=False,
                save=None,
                method='matplotlib',
                **kwargs):
    """
    Spatial plots of sdata objects. 
    """
    
    if figsize[0] is None and figsize[1] is not None:
        size = get_range(sd_dims(sdata, shape))
        figsize = (figsize[1] * size['x'] / size['y'], figsize[1])
    
    if obs_key is not None or feature is not None:
        region_key = 'plot_region'
        sdata.tables[table_key].obs[region_key] = shape
        sdata.set_table_annotates_spatialelement(table_name = table_key, region = shape, region_key = region_key, instance_key = 'cell_id')

        if feature is not None:
            obs_key = feature
            if layer is None:
                expr = np.array(sdata.tables[table_key][:, feature].X.todense())
            else:
                expr = np.array(sdata.tables[table_key][:, feature].layers[layer].todense())                   
            sdata.tables[table_key].obs[obs_key] = expr

        if obs_key is not None:
            if sdata[table_key].obs[obs_key].isnull().any():
                sdata[table_key].obs[obs_key] = sdata[table_key].obs[obs_key].astype('str')
                sdata[table_key].obs[obs_key].fillna('None')
            if palette is None:
                palette_dict = get_obs_colors(sdata[table_key], obs_key=obs_key, colors_key='colors')
                if palette_dict is not None:
                    palette=list(palette_dict.values())
                    if groups is None:
                        groups=list(palette_dict)
            if cmap is None and palette is None:
                cmap = get_cmap(sdata[table_key].obs[obs_key])
    
    obs_key_plot = obs_key
    obs_orig = sdata[table_key].obs.copy()
    
    if cmap_q is not None:
        q_val = np.quantile(sdata[table_key].obs[obs_key_plot], cmap_q)
        obs_val_clipped = np.clip(sdata[table_key].obs[obs_key_plot], a_min=0, a_max=q_val)
        obs_key_plot = '_' + obs_key + '_clipped_'
        sdata[table_key].obs[obs_key_plot] = obs_val_clipped
        if method != 'matplotlib':
            raise ValueError(f"Colorbar limits currently only work with method='matplotlib'")

    if cmap_clip is not None:
        obs_val_clipped = np.clip(sdata[table_key].obs[obs_key_plot], a_min=0, a_max=cmap_clip)
        obs_key_plot = '_' + obs_key + '_clipped_'
        sdata[table_key].obs[obs_key_plot] = obs_val_clipped
        if method != 'matplotlib':
            raise ValueError(f"Colorbar limits currently only work with method='matplotlib'")
    
    if title is None:
        title=obs_key

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.tight_layout(pad=0.5)

    if ax_linewidth is None:
        ax_linewidth = figsize[1]/5

    with mpl.rc_context({
        'font.size': fontsize,
        'legend.fontsize': fontsize,
        'axes.titlesize': fontsize,
        'axes.labelsize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'figure.titlesize': fontsize,
        'legend.markerscale': legend_markerscale,
        'axes.linewidth': ax_linewidth,
        'xtick.major.width': ax_linewidth,
        'ytick.major.width': ax_linewidth,
        'legend.labelspacing': legend_spacing
    }):
        
        if image is not None:
            sdata = sdata.pl.render_images(image,
                                           channel=image_channel,
                                           cmap=image_cmap,
                                           norm=None,
                                           na_color='default',
                                           palette=image_palette,
                                           alpha=image_alpha,
                                           scale='full')
        
        if shape is not None:
            sdata = sdata.pl.render_shapes(shape,
                                           color=obs_key_plot,
                                           coordinate_systems=coordinate_system,
                                           table_name=table_key,
                                           palette=palette,
                                           groups=groups,
                                           cmap=cmap,
                                           method=method,
                                           na_color=na_color,
                                           fill_alpha=fill_alpha,
                                           outline_width=outline_width,
                                           outline_color=outline_color,
                                           outline_alpha=outline_alpha,
                                           rasterize=rasterize_shapes,
                                           **kwargs)

        if points is not None:
            if points is True:
                points = None
            if points_palette is None:
                if points is not None:
                    cmap = mpl.cm.get_cmap(get_cmap(pd.Series(points)))
                    points_palette = [cmap(i / len(points)) for i in range(len(points))]
                    points_palette = [mpl.colors.rgb2hex(color) for color in points_palette]
            sdata = sdata.pl.render_points(element=points_key,
                                           color=points_color,
                                           size=points_size,
                                           palette=points_palette,
                                           groups=points,
                                           cmap=points_cmap,
                                           marker='.',
                                           method='matplotlib')
 
        sdata.pl.show(ax=ax, title=title, coordinate_systems=coordinate_system, dpi=dpi, legend_fontsize=fontsize)
        
        if legend_loc is not None:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc=legend_loc, prop=legend_prop, bbox_to_anchor=bbox_to_anchor,
                      alignment='left', frameon=False, fontsize=fontsize*0.8)
        
        ax.tick_params(axis='both', which='major', labelsize=fontsize*0.8)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize*0.8)
        ax.set_xlabel(xlabel, fontsize=fontsize*0.8)
        ax.set_ylabel(ylabel, fontsize=fontsize*0.8)
        ax.grid(False)
        ax.set_anchor(ax_anchor)
        ax.spines['top'].set_linewidth(ax_linewidth)
        ax.spines['right'].set_linewidth(ax_linewidth)
        ax.spines['bottom'].set_linewidth(ax_linewidth)
        ax.spines['left'].set_linewidth(ax_linewidth)
        ax.tick_params(axis='both', length=ax_linewidth*3, width=ax_linewidth)
        
        if (obs_key_plot not in obs_orig.columns) and (obs_key_plot in sdata[table_key].obs.columns):
            del sdata[table_key].obs[obs_key_plot]
    
        if feature is not None:
            del sdata.tables[table_key].obs[obs_key]
    
        if save is not None:
            fig.savefig(save, bbox_inches='tight', pad_inches=0, dpi=dpi)
        else:
            return ax


def get_cmap(data, cmap_continuous=None, cmap_cat10='tab10', cmap_cat20='tab20'):
    
    unique_values = data.nunique()
    if cmap_continuous is None:
        cmap_continuous = LinearSegmentedColormap.from_list("grey_to_blue", ["lightgrey", "mediumblue"])
    
    if pd.api.types.is_numeric_dtype(data):
        if unique_values <= 10:
            return cmap_cat10 
        else:
            return cmap_continuous
    else:
        if unique_values <= 10:
            return cmap_cat10
        elif unique_values <= 20:
            return cmap_cat20
        else:
            return cmap_continuous


def get_obs_colors(adata, obs_key='celltype', colors_key='colors'):
    colors = None
    if colors_key in adata.uns.keys():
        if obs_key in adata.uns[colors_key].keys():
            colors = adata.uns[colors_key][obs_key]
    return colors

    
def get_range(dims):
    xrange = dims['xmax'] - dims['xmin']
    yrange = dims['ymax'] - dims['ymin']
    
    if dims['zmax'] is not None and dims['zmin'] is not None:
        zrange = dims['zmax'] - dims['zmin']
    else:
        zrange = None
    
    return {'x': xrange, 'y': yrange, 'z': zrange}


def sd_dims(sdata, element='morphology_focus'):

    xmin = 0
    ymin = 0
    zmin = None
    xmax = None
    ymax = None
    zmax = None
    
    if element in list(sdata.images):
        # always starting at 0,0?
        xmax = sdata.images[element]['scale0'].dims['x']
        ymax = sdata.images[element]['scale0'].dims['y']
    
    if element in list(sdata.shapes):
        # usually no z-level
        xmin, ymin, xmax, ymax = sdata.shapes[element].total_bounds

    if element in list(sdata.points):
        xmin = sdata.points[element]['x'].min().compute()
        xmax = sdata.points[element]['x'].max().compute()
        ymin = sdata.points[element]['y'].min().compute()
        ymax = sdata.points[element]['y'].max().compute()
        zmin = sdata.points[element]['z'].min().compute()
        zmax = sdata.points[element]['z'].max().compute()
        
    return {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'zmin': zmin, 'zmax': zmax}

