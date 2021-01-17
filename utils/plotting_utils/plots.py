import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np

import sys
sys.path.append("../..")
from utils import utils

def prepare_sns(use_tex=True):
    sns.set()
    sns.set_style("ticks")
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)
    plt.rc("axes", labelsize=16)
    plt.rcParams["text.usetex"] = use_tex

def get_group_colors(as_dict=False):
    if as_dict:
        return {"Fusional": "#95c78f",
                "Isolating": "#f79d97",
                "Agglutinative": "#abaff5",
                "Introflexive": "#fffecc"}
    else:
        return ["#95c78f", "#f79d97", "#abaff5", "#fffecc"]

def get_dual_bar_colors(as_dict=False):
    if as_dict:
        return {"Same": "#870c85", "Others": "#ff1cfb"}
    else:
        return ["#870c85", "#ff1cfb"]

def add_titles(title, xaxis_title, yaxis_title, fontsize):
    plt.ylabel(yaxis_title, fontsize=8 + fontsize, labelpad=20)
    plt.xlabel(xaxis_title, fontsize=8 +  fontsize)
    plt.title(title, fontsize=12 + fontsize, pad=20)
    plt.tick_params(labelsize=fontsize)

def add_legend(title, fontsize, coords):
    plt.legend(title=title, title_fontsize=2 + fontsize, loc="upper left",
               bbox_to_anchor=coords, fontsize=fontsize, facecolor="lightgrey",
               framealpha=1, edgecolor="black", labelspacing=0.7)

def scatter(x, y, data, style=None, kind="lmplot", log_x=False, log_y=False, extra_fontsize=0,
            fit_reg=False, exclude=[], exclude_reg=[], xlim=None, ylim=None, custom_offsets={},
            title="", xaxis_title="", yaxis_title="", legend_coords=(1, 0.5), use_tex=True,
            show=False, remove_labels=False):
    lang_col = utils.find_lang_column(data)
    data = data[~data[lang_col].isin(exclude)] # Exclude languages
    data = utils.add_lang_groups(data, colname="Group")

    if x not in data.columns:
        path, colname = utils.find_table(x, by="colname")
        x = colname
        data = utils.merge_tables(data, pd.read_excel(path), how="left", cols_table2=[colname])
    if y not in data.columns:
        path, colname = utils.find_table(y, by="colname")
        y = colname
        data = utils.merge_tables(data, pd.read_excel(path), how="left", cols_table2=[colname])
    # Transform to log
    if log_x:
        data["log_" + x] = np.log(data[x])
        x = "log_" + x
    if log_y:
        data["log_" + y] = np.log(data[y])
        y = "log_" + y
    xmax = data[x].max()
    xmin = data[x].min()
    ymax = data[y].max()
    ymin = data[y].min()

    # Prepare seaborn
    prepare_sns(use_tex)

    # Plot parameters
    colors = get_group_colors(as_dict=True)
    colors = {k: sns.saturate(v) for k,v in colors.items()}
    vertical_offset = (ymax - ymin) * 0.03
    custom_offsets = {k: (v[0] * (xmax - xmin), v[1] * (ymax - ymin)) for k, v in custom_offsets.items()}
    offsets = dict.fromkeys(data[lang_col].values, (0, vertical_offset))
    offsets.update(custom_offsets)

    # Main plot
    if kind == "lmplot":
        g = sns.lmplot(x=x, y=y, data=data, hue="Group", palette=colors, fit_reg=False, legend=False,
                       height=6, aspect=1.5, scatter_kws={"s": 150, "edgecolors": "black"})
        # Add legend
        add_legend(title="Language Group", fontsize=20 + extra_fontsize, coords=legend_coords)
    elif kind == "relplot":
        g = sns.relplot(x=x, y=y, data=data, hue="Group", style=style, palette=colors, legend="full",
                        height=8, aspect=1.25, edgecolor="black", s=150)
        # Add legend
        leg = g._legend
        leg.set_visible(False)
        leg = plt.legend(loc="upper left", bbox_to_anchor=legend_coords, fontsize=20 + extra_fontsize,
                         facecolor="lightgrey", framealpha=1, edgecolor="black", labelspacing=0.4)
        # Fix legend handles (dots, etc)
        for handle in leg.legendHandles:
            handle.set_sizes([150])
            handle.set_lw(0.5)
            handle.set_edgecolor("black")
        # Fix legend subtitles
        leg.texts[0].set_text("Language Group")
        leg.texts[0].set_position((-50, 0))
        if style:
            leg.texts[5].set_text(style.replace("_", " ").title())
            leg.texts[5].set_position((-50, 0))
    else:
        raise Exception("'{}' is not a valid kind of scatter plot.".format(kind))
    if fit_reg:
        sns.regplot(x=x, y=y, data=data[~data[lang_col].isin(exclude_reg)], scatter=False, color="grey", ci=None)

    # Language names
    if not remove_labels:
        for i, row in data.iterrows():
            plt.text(row[x] + offsets[row[lang_col]][0],
                     row[y] + offsets[row[lang_col]][1],
                     row[lang_col], fontsize=14 + extra_fontsize,
                     horizontalalignment="center")

    # Titles, xlim, fontsize, etc
    if xlim is None:
        xlim = (xmin * 0.95, xmax * 1.05)
    if ylim is None:
        ylim = (ymin * 0.95, ymax * 1.05)
    plt.xlim(xlim)
    plt.ylim(ylim)
    add_titles(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title, fontsize=16 + extra_fontsize)

    if show:
        plt.show()
        plt.close()
    else:
        return g