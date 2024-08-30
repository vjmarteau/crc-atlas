from typing import Callable, Optional, Union

import altair as alt
import decoupler as dc
import numpy as np
import pandas as pd

from ._fdr import fdr_correction


def plot_lm_result_altair(
    df: pd.DataFrame,
    p_cutoff: float = 0.1,
    p_col: str = "fdr",
    x: str = "variable",
    y: str = "group",
    color: str = "coef",
    title: str = "heatmap",
    cluster: bool = False,
    value_max: Optional[float] = None,
    configure: Callable[[alt.Chart], alt.Chart] = lambda x: x.configure_mark(opacity=1),
    cmap: str = "redblue",
    reverse: bool = True,
    domain: Callable[[float], list[float]] = lambda x: [-x, x],
    order: Union[str, np.ndarray, None] = None,
) -> alt.Chart:
    """
    Plot a results data frame of a comparison as a heatmap

    Parameters
    ----------
    df
        The input DataFrame containing the comparison results.
    p_cutoff
        The p-value cutoff for filtering the data. Default is 0.1.
    p_col
        The column name containing the p-values. Default is "fdr".
    x
        The column name for the x-axis values. Default is "variable".
    y
        The column name for the y-axis values. Default is "group".
    color
        The column name for the color values. Default is "coef".
    title
        The title of the heatmap. Default is "heatmap".
    cluster
        If True, perform clustering. Default is False.
    value_max
        The maximum value for color scale normalization. If None, it is determined from the data. Default is None.
    configure
        A function to configure the Altair chart. Default is a function that sets opacity to 1.
    cmap
        The color map for the heatmap. Default is "redblue".
    reverse
        If True, reverse the color scale. Default is True.
    domain
        A function to set the color scale domain. Default is a function that takes the negation of the maximum value.
    order
        The order of x-axis values. If None, it is determined based on clustering. Default is None.

    """
    df_filtered = df.loc[lambda _: _[p_col] < p_cutoff, :]
    df_subset = df.loc[
        lambda _: _[x].isin(df_filtered[x].unique()) & _[y].isin(df[y].unique())
    ]
    if not df_subset.shape[0]:
        print("No values to plot")
        return

    if order is None:
        order = "ascending"
        if cluster:
            from scipy.cluster.hierarchy import leaves_list, linkage

            values_df = df_subset.pivot(index=y, columns=x, values=color)
            order = values_df.columns.values[
                leaves_list(
                    linkage(values_df.values.T, method="average", metric="euclidean")
                )
            ]

    def _get_significance(fdr):
        if fdr < 0.001:
            return "< 0.001"
        elif fdr < 0.01:
            return "< 0.01"
        elif fdr < 0.1:
            return "< 0.1"
        else:
            return np.nan

    df_subset["FDR"] = pd.Categorical([_get_significance(x) for x in df_subset[p_col]])

    if value_max is None:
        value_max = max(
            abs(np.nanmin(df_subset[color])), abs(np.nanmax(df_subset[color]))
        )
    # just setting the domain in altair will lead to "black" fields. Therefore, we constrain the values themselves.
    df_subset[color] = np.clip(df_subset[color], *domain(value_max))
    return configure(
        alt.Chart(df_subset, title=title)
        .mark_rect()
        .encode(
            x=alt.X(x, sort=order),
            y=y,
            color=alt.Color(
                color,
                scale=alt.Scale(scheme=cmap, reverse=reverse, domain=domain(value_max)),
            ),
        )
        + alt.Chart(df_subset.loc[lambda x: ~x["FDR"].isnull()])
        .mark_point(color="white", filled=True, stroke="black", strokeWidth=0)
        .encode(
            x=alt.X(x, sort=order, axis=alt.Axis(labelLimit=1000)),
            y=y,
            size=alt.Size(
                "FDR:N",
                scale=alt.Scale(
                    domain=["< 0.001", "< 0.01", "< 0.1"],
                    range=4 ** np.array([3, 2, 1]),
                ),
            ),
        )
    )


def format_decoupler_results(tf_acts, tf_pvals, name="variable", contrast="contrast"):
    return (
        dc.format_contrast_results(tf_acts, tf_pvals)
        .drop(columns="adj_pvals")
        .rename(
            columns={
                "logFCs": "act_score",
                "pvals": "pvalue",
                "name": name,
                "contrast": contrast,
            }
        )
        .assign(
            act_score=lambda x: x["act_score"].fillna(0),
            pvalue=lambda x: x["pvalue"].fillna(1),
        )
        .pipe(fdr_correction)
    )
