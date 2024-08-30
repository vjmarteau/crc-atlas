import altair as alt
import pandas as pd
from natsort import natsorted

def set_scale_anndata(adata, column, palette=None):
    if palette is None:
        palette = column

    adata._sanitize()

    tmp_cols = getattr(COLORS, palette)
    adata.uns[f"{column}_colors"] = [
        tmp_cols[cat] for cat in adata.obs[column].cat.categories
    ]


def altair_scale(variable, *, data=None, data_col=None, **kwargs):
    """
    Discrete color scale for altair based on our color definitions.

    Parameters:
    -----------
    variable
        name of the color scale
    data
        Data frame used for the chart. If specified, will only show values that actually occur in the data.
    data_col
        If specified, check this column in `data` instead of `variable`

    Returns
    -------
    Altair color scale
    """
    tmp_colors = getattr(COLORS, variable)
    if data is not None:
        data_col = variable if data_col is None else data_col
        tmp_colors = {k: tmp_colors[k] for k in sorted(data[data_col].unique())}

    return alt.Scale(
        domain=list(tmp_colors.keys()),
        range=list(tmp_colors.values()),
        **kwargs,
    )


def altair_scale_mpl(scheme, **kwargs):
    """
    Use a continuous color scheme from mpl with altair
    """
    from matplotlib import cm
    from matplotlib.colors import to_hex

    return alt.Scale(
        range=[to_hex(x) for x in cm.get_cmap(scheme, 1000)(range(1000))], **kwargs
    )


def plot_palette(variable):
    """Display a palette"""
    tmp_cols = getattr(COLORS, variable)
    return (
        alt.Chart(
            pd.DataFrame.from_dict(tmp_cols, orient="index", columns=["color"])
            .reset_index()
            .rename(columns={"index": variable})
        )
        .mark_rect(height=40, width=30)
        .encode(
            x=alt.X(variable),
            color=alt.Color(variable, scale=altair_scale(variable), legend=None),
        )
    )


def plot_all_palettes():
    return alt.vconcat(
        *[plot_palette(v) for v in COLORS.__dict__.keys() if not v.startswith("_")]
    ).resolve_scale(color="independent")

class COLORS:

    sample_type = {
        "tumor": "#ff7f0e",  # Orange
        "normal": "#1f77b4",  # Blue
        "blood": "#aa40fc",  # Purple
        "metastasis": "#d62728",  # Red
        "polyp": "#f7b500",  # Yellow-orange
        "lymph node": "#279e68",  # Green
    }
    cancer_type = {
        "COAD": "#1f77b4",
        "COAD polyp": "#aa40fc",
        "READ": "#ff7f0e",
        "READ polyp": "#d62728",
        "normal": "#279e68",
        "unknown": "#dddddd",
    }

    sex = {
        "male": "#1f77b4",
        "female": "#ff7f0e",
        "unknown": "#dddddd",
    }
    age = {
        ">50": "#1f77b4",
        "<50": "#ff7f0e",
        "unknown": "#dddddd",
    }
    microsatellite_status = {
        "MSI": "#ff7f0e",
        "MSS": "#1f77b4",
        "unknown": "#dddddd",
    }
    anatomic_region = {
        "proximal colon": "#aa40fc",  # Purple
        "distal colon": "#1f77b4",  # Blue
        "blood": "#279e68",  # Green
        "liver": "#ff7f0e",  # Orange
        "mesenteric lymph nodes": "#d62728",  # Red
        "unknown": "#dddddd",  # Light gray
    }
    mismatch_repair_deficiency_status = {
        "dMMR": "#80b1d3",
        "pMMR": "#e41a1c",
        "unknown": "#dddddd",
    }
    tumor_stage = {
        "early": "#998ec3",
        "advanced": "#f1a340",
        "unknown": "#dddddd",
    }
    tumor_stage_verbose = {
        "early (I/II)": "#998ec3",
        "advanced (III/IV)": "#f1a340",
        "unknown": "#dddddd",
    }
    immune_type = {
        "B/CD4": "#d5eae7",
        "CD8": "#1f78b4",
        "M": "#33a02c",
        "B/T": "#ff7f00",
        "desert": "#999999",
        # "n/a": "#dddddd",
    }
    immune_type2 = {
        "T": "#1f78b4",
        "M": "#33a02c",
        "B": "#ff7f00",
        "desert": "#999999",
        # "n/a": "#dddddd",
    }
    histological_type = {
        "adenocarcinoma": "#ff7f00",
        "mucinous adenocarcinoma": "#1f78b4",
        "medullary carcinoma": "#33a02c",
        "unknown": "#999999",
        # "n/a": "#dddddd",
    }
    platform = {
        "BD Rhapsody": "#279e68",  # Green
        "10x 3p": "#1f77b4",  # Blue
        "10x 5p": "#add8e6",  # Light Blue
        "TruDrop": "#ff7f0e",  # Orange
        "DNBelab C4": "#aa40fc",  # Purple
        "GEXSCOPE Singleron": "#8c564b",  # Brown
        "Smart-seq2": "#d62728",  # Red
        "SMARTer (C1)": "#b5bd61",  # Olive green
        "scTrio-seq2": "#17becf",  # Cyan
    }
    tumor_grade = {
        "G1": "#279e68",  # Green
        "G2": "#1f77b4",  # Blue
        "G3": "#ff7f0e",  # Orange
        "G4": "#d62728",  # Red
        "unknown": "#dddddd",  # Light gray
    }
    dataset = {
        "Che_2021": "#003f5c",  # Dark navy blue
        "Chen_2024": "#2c7bb6",  # Dark blue
        "Guo_2022": "#41b6c4",  # Medium blue
        "Huang_2024": "#7fcdbb",  # Light blue
        "Ji_2024_scopeV1": "#003f5c",  # Very dark navy blue
        "Ji_2024_scopeV2": "#d0e0e6",  # Very light blue
        "Joanito_2022_KUL5": "#005b96",  # Standard blue
        "Joanito_2022_SG1": "#08306b",  # Deep navy blue
        "Joanito_2022_SG2": "#ccece6",  # Very soft blue
        "Khaliq_2022": "#9ecae1",  # Light cyan blue
        "Lee_2020_KUL3": "#08519c",  # Deep blue
        "Lee_2020_SMC": "#4292c6",  # Bright sky blue
        "Lee_2020_SMC_test_setup_fresh": "#2171b5",  # Medium blue
        "Lee_2020_SMC_test_setup_frozen": "#6baed6",  # Sky blue
        "Li_2023_10Xv2": "#e5f5f9",  # Pale soft blue
        "Li_2023_10Xv3": "#e0f3f8",  # Soft blue
        "MUI_Innsbruck": "#004c6d",  # Navy blue
        "Pelka_2021_10Xv2": "#002d72",  # Dark navy blue
        "Pelka_2021_10Xv3": "#a8d4e8",  # Pale blue
        "Qi_2022": "#004a8c",  # Rich blue
        "Qian_2020": "#f7f7f7",  # Faint blue
        "Qin_2023": "#0072b2",  # Bright blue
        "Terekhanova_2023_HTAN": "#0094d1",  # Light sky blue
        "Uhlitz_2021": "#00b2e2",  # Bright cyan blue
        "VUMC_HTAN_CRC": "#5cb1e8",  # Pale cyan blue
        "WUSTL_HTAN": "#bfd3e6",  # Very light blue
        "Wang_2021": "#7fc9d4",  # Soft blue
        "Yang_2023": "#a0d4e4",  # Very pale blue
        "Zheng_2022": "#d9eaf5",  # Extremely light blue
    }
    treatment_status_before_resection = {
        "naive": "#998ec3",
        "treated": "#f1a340",
        "unknown": "#dddddd",
    }
    KRAS_status = {
        "wt": "#998ec3",
        "mut": "#f1a340",
        "unknown": "#dddddd",
    }
    TP53_status = {
        "wt": "#998ec3",
        "mut": "#f1a340",
        "unknown": "#dddddd",
    }
    APC_status = {
        "wt": "#998ec3",
        "mut": "#f1a340",
        "unknown": "#dddddd",
    }
    BRAF_status = {
        "wt": "#998ec3",
        "mut": "#f1a340",
        "unknown": "#dddddd",
    }
    PIK3CA_status = {
        "wt": "#998ec3",
        "mut": "#f1a340",
        "unknown": "#dddddd",
    }
    driver_mutation_status = {
        "wt": "#ff7f0e",
        "mut": "#1f77b4",
        "unknown": "#dddddd",  # Light gray
    }
    cell_type_coarse = dict(
        natsorted(
            {
                "Myeloid cell": "#c51b7d",  # Magenta-pink
                "Neutrophil": "#ff6f00",  # Vivid orange
                "Mast cell": "#86594c",  # Earthy brown
                "B cell": "#f7b500",  # Warm yellow-orange
                "Plasma cell": "#91bfdb",  # Light blue
                "T cell": "#4575b4",  # Cool blue
                "NK": "#d95f0e",  # Reddish orange
                "Cancer cell": "#d73027",  # Deep red
                "Epithelial cell": "#4d9221",  # Rich green
                "Stromal cell": "#17becf",  # Bright cyan
                "Schwann cell": "#d1a85c",  # Earthy yellow-brown
            }.items()
        )
    )
    cell_type_immune_infiltartion = dict(
        natsorted(
            {
                "Cancer cell": "#d73027",
                "B cell": "#f7b500",
                "Plasma IgA": "#91bfdb",
                "Plasma IgG": "#fdae6b",
                "Monocyte": "#006837",
                "Macrophage": "#78c679",
                "Dendritic cell": "#ff6f00",
                "Mast cell": "#86594c",
                "T cell CD4": "#cc4c02",
                "T cell CD8": "#662506",
                "Treg": "#fe9929",
                "NK": "#d95f0e",
                "Endothelial cell": "#4d9221",
                "Fibroblast": "#17becf",
                "unknown": "#dddddd",
            }.items()
        )
    )
    cell_type_fine = dict(
        natsorted(
            {
                'Monocyte': "#17becf",
                'Macrophage': "#b5bd61",
                'Macrophage cycling': "#ff7f0e",
                'cDC progenitor': "#d62728",
                'cDC1': "#aa40fc",
                'cDC2': "#8c564b",
                'DC3': "#e377c2",
                'pDC': "#f7b500",
                'DC mature': "#ff7f0e",
                'Granulocyte progenitor': "#aec7e8",
                'Neutrophil': "#ff6f00",
                'Eosinophil': "#98df8a",
                'Mast cell': "#86594c",
                'Erythroid cell': "#c5b0d5",
                'Platelet': "#c51b7d",
                'GC B cell': "#31a354",
                'B cell naive': "#fed976",
                'B cell activated naive': "#fed976",
                'B cell activated': "#f7b500",
                'B cell memory': "#f03b20",
                'Plasma IgA': "#91bfdb",
                'Plasma IgG': "#4575b4",
                'Plasma IgM': "#ff7f0e",
                'Plasmablast': "#d1a85c",
                "CD4": "#1f77b4",
                "Treg": "#ff7f0e",
                "CD8": "#279e68",
                "NK": "#d62728",
                "ILC": "#ff6f00",
                "gamma-delta": "#aa40fc",
                "NK": "#8c564b",
                "NKT": "#e377c2",
                "CD4 naive": "#17becf",
                "CD8 naive": "#b5bd61",
                "CD4 stem-like": "#17becf",
                "CD8 stem-like": "#b5bd61",
                "CD4 cycling": "#1f77b4",
                "CD8 cycling": "#279e68",
                "NK cycling": "#d62728",
            }.items()
        )
    )
    
    