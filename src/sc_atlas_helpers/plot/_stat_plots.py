from typing import List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def swarm_box_corr(
    res: pd.DataFrame,
    *,
    symbol: str,
    cell_type: str,
    comparison: str,
    color_by: str,
    title: str,
    use_log: bool = False,
    ms: int = 5,
    legend: bool = False,
    legend_title: str = "",
    cat_order: List[str] = ["tumor", "normal"],
    save_plot: Optional[bool] = False,
    prefix: str = "",
    save_dir: Optional[str] = None,
    file_ext: str = "pdf",
    display: bool = True,
) -> plt.Figure:

    df = res.loc[
        (res["cell_type_fine"] == cell_type) & (res["symbol"] == symbol)
    ].copy()

    pd_padj = df["padj"]
    pd_padj.index = df.comparison

    df[comparison] = pd.Categorical(df[comparison], categories=cat_order, ordered=True)
    # Create the strip plot with jitter
    log = use_log
    y = "norm_counts"
    plt.figure(figsize=(2.5, 4))
    ax = plt.gca()
    sns.swarmplot(
        x=comparison,
        y=y,
        data=df,  # jitter=True,
        # palette=['C0','C1','C2'], # colors for hue ...
        hue=color_by,  # variable after what you want to color
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
        zorder=0,
        size=ms,
        ax=ax,
        legend=legend,
    )
    if legend:
        han, lab = ax.get_legend_handles_labels()
        ax.legend(
            han,
            lab,
            title=legend_title,
            fontsize=10,
            title_fontsize=10,
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )
    ax.set_xlabel("")
    ax.set_ylabel(title, weight="bold")

    for j, c in enumerate(cat_order):
        n_sample_type = df["sample_type"].value_counts()[c]
        ax.text(
            j,
            ax.get_ylim()[0] * 0.99,
            f"n={n_sample_type}",
            ha="center",
            color="grey",  # weight='bold',
            va="bottom",
            fontsize=9,
            zorder=10,
        )
    # plt.title('HLA-DR MFI')
    if log:
        ax.set_ylim([10, 10001])
        ax.set_yscale("log")
        ax.set_yticks([10, 100, 1000, 10000])

    ### potential keyword arguments
    alpha = 0.9
    delta_iqr = 0.15
    delta_med = 0.3

    if log:
        factor_h = 2.2
        factor_v = 0.08
    else:
        factor_h = 1.1
        factor_v = 0.04
    #####

    for j, x in enumerate(df[comparison].cat.categories):
        q25, q50, q75 = (
            df.loc[df[comparison] == x, y].quantile([0.25, 0.5, 0.75]).values
        )
        ax.vlines(x=j, ymin=q25, ymax=q75, lw=1, color="black", zorder=10, alpha=alpha)
        for q in [q25, q75]:
            ax.hlines(
                y=q,
                xmin=j - delta_iqr,
                xmax=j + delta_iqr,
                lw=1,
                color="black",
                zorder=10,
                alpha=alpha,
            )
        ax.hlines(
            y=q50,
            xmin=j - delta_med,
            xmax=j + delta_med,
            lw=2.5,
            color="black",
            zorder=10,
            alpha=alpha,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", which="major", width=1.5, length=5)

    # Increase the width of the left and bottom spines (axes)
    ax.spines["left"].set_linewidth(1.5)  # Adjust the left axis width
    ax.spines["bottom"].set_linewidth(1.5)  # Adjust the bottom axis width

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")  # Set bold font weight

    y_height_l = df[y].max() * factor_h

    for combi in pd_padj.index:
        padj = pd_padj.loc[combi].unique().squeeze()  # .values
        if padj < 0.1:
            c1, _, c2 = combi.split("_")
            for c in [c1, c2]:
                if cat_order.index(c2) - cat_order.index(c1) > 1.5:
                    if log:
                        ymin = y_height_l * 1.2
                        yvline = y_height_l * 1.6
                    else:
                        ymin = y_height_l * 1.05
                        yvline = y_height_l * 1.1
                else:
                    yvline = y_height_l
                    ymin = df.loc[df[comparison] == c][y].max() + factor_v * y_height_l
                ax.vlines(
                    x=cat_order.index(c), ymin=ymin, ymax=yvline, color="black", lw=0.8
                )
            ax.hlines(
                y=yvline,
                xmin=cat_order.index(c1),
                xmax=cat_order.index(c2),
                color="black",
                lw=0.8,
            )
            if padj < 0.1:
                t = "."
            if padj < 0.05:
                t = "*"
            if padj < 0.01:
                t = "**"  ### ...todo ...
            if padj < 0.001:
                t = "***"
            ax.text(
                cat_order.index(c1) + 0.5,
                yvline * 0.99,
                t,
                weight="bold",
                fontsize=12,
                horizontalalignment="center",
                verticalalignment="bottom",
            )
    ax.set_title(cell_type + ": " + r"$\mathit{" + symbol + "}$")

    if save_plot:
        plt.savefig(
            f"{save_dir}/{prefix}-swarm_box_corr_plot_{cell_type}_{symbol}.{file_ext}",
            bbox_inches="tight",
        )
        if display:
            plt.show()
        plt.close()
