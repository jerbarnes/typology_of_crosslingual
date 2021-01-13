import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import seaborn as sns
from tqdm.notebook import tqdm
from collections import Counter
from functools import reduce
import sys
sys.path.extend(["..", "../.."])
from data_preparation.data_preparation_pos import read_conll
from utils import utils, pos_utils, plotting_utils

### MULTIWORDS
def count_multiwords(info, table, included_langs):
    file_path = info["file_path"]
    lang_name = info["lang_name"]
    dataset = info["dataset"]

    if lang_name in included_langs:
        conllu_data = read_conll(file_path)
        multiwords = []
        total_tags = []
        for taglist in conllu_data[2]:
            multiwords.append(taglist.count("_"))
            total_tags.append(len(taglist))
        multiwords = sum(multiwords)
        total_tags = sum(total_tags)

        if lang_name in table["language"].values.tolist():
            table.loc[table["language"] == lang_name, [dataset, dataset + " (%)"]] = multiwords, multiwords/total_tags * 100
        else:
            table.loc[table.shape[0], ["language", dataset, dataset + " (%)"]] = lang_name, multiwords, multiwords/total_tags * 100
    return table

def build_multiwords_table(experiment, save_to=None):
    included_langs = utils.get_langs(experiment)
    data_path = utils.find_relative_path_to_root() + "data/ud/"
    table = pd.DataFrame(dict.fromkeys(["language", "train", "train (%)", "dev", "dev (%)", "test", "test (%)"], []))
    table = utils.run_through_data(data_path, count_multiwords, table, included_langs=included_langs)
    table = utils.order_table(table, experiment)
    table = table.astype(dict.fromkeys(["train", "dev", "test"], pd.Int64Dtype())) # Convert to int
    if save_to:
        table.to_excel(save_to, index=False)
    return table

### TAG DISTRIBUTION
# Tag tables
def multi_merge(left, right, r):
    left_cols = ["Tag"] + list(filter(r.match, left.columns))
    right_cols = ["Tag"] + list(filter(r.match, right.columns))
    return pd.merge(left[left_cols], right[right_cols], on="Tag", suffixes=(None, "_{}".format(len(left_cols)-1)))

def calculate_total_tag_table(dfs):
    total = reduce(lambda left, right: multi_merge(left, right, re.compile("^Count($|_)")), dfs)
    total["Count"] = total.iloc[:,1:].apply(np.sum, axis=1)
    total = total.sort_values("Count", ascending=False).reset_index(drop=True)
    total = total.loc[:, ["Tag", "Count"]]
    total["Count(%)"] = total["Count"] / total["Count"].sum() * 100
    total["Cumulative(%)"] = total["Count(%)"].cumsum()
    return total

def tag_freq(info, output, lang_to_group, included_langs):
    lang_name = info["lang_name"]
    file_path = info["file_path"]
    dataset = info["dataset"]
    group = lang_to_group[lang_name]

    if lang_name in included_langs:
        conll_data = read_conll(file_path)
        tags = []
        for tag_list in conll_data[2]:
            tags.extend(tag_list)
        df = pd.DataFrame(list(Counter(tags).items()), columns=["Tag", "Count"])
        df = df.sort_values("Count", ascending=False).reset_index(drop=True)
        df["Count(%)"] = df["Count"] / df["Count"].sum() * 100
        df["Cumulative(%)"] = df["Count(%)"].cumsum()

        # Add missing tags
        tagset = pos_utils.get_ud_tags()
        missing_tags = set(tagset) ^ set(df["Tag"])
        missing_data = {col: [100]*len(missing_tags) if "Cumulative" in col else [0]*len(missing_tags) for col in df.columns[1:]}
        missing_rows = pd.DataFrame({"Tag": list(missing_tags), **missing_data},
                                    index=range(df.shape[0], df.shape[0] + len(missing_tags)))
        df = pd.concat([df, missing_rows])

        if lang_name not in output[group].keys():
            output[group][lang_name] = {}
        output[group][lang_name][dataset] = df

        # Calculate totals if all datasets are done
        if len(output[group][lang_name].keys()) == 3:
            total = calculate_total_tag_table(output[group][lang_name].values())
            output[group][lang_name]["total"] = total

    return output

def build_tag_tables(experiment):
    data_path = utils.find_relative_path_to_root() + "data/ud/"
    included_langs = utils.get_langs(experiment)
    table = {x: {} for x in ["Fusional", "Isolating", "Agglutinative", "Introflexive"]} # Setup dict per group
    tag_tables = utils.run_through_data(data_path, tag_freq, lang_to_group=utils.lang_to_group,
                                        table=table, included_langs=included_langs)
    return tag_tables

# Output
def calculate_group_avg_tables(group_tables):
    group_avgs = {}

    for dataset in ["train", "dev", "test", "total"]:
        table = reduce(lambda left, right: multi_merge(left, right, re.compile("^Count\(%\)")),
                       [group_tables[lang][dataset] for lang in group_tables.keys() if dataset in group_tables[lang].keys()])
        table = pd.DataFrame({"Tag": table["Tag"],
                              "MeanCount(%)": table.iloc[:,1:].apply(np.mean, axis=1)})
        table = table.sort_values("MeanCount(%)", ascending=False).reset_index(drop=True)
        table["Cumulative(%)"] = table["MeanCount(%)"].cumsum()
        group_avgs[dataset] = table

    return group_avgs

def export_tag_tables(tag_tables, experiment, output_path, img_path):
    group_to_color = plotting_utils.plots.get_group_colors(as_dict=True)

    for group in tag_tables.keys():
        writer = pd.ExcelWriter(output_path + "tag_stats_{}.xlsx".format(group.lower()))
        workbook  = writer.book

        # Formats
        percentage_format = workbook.add_format({"num_format": "0.00\%"})
        merge_format = workbook.add_format({
            "bold": 1,
            "border": 1,
            "align": "center",
            "fg_color": group_to_color[group],
            "font_size": 14
        })

        # Sheet for every language
        langs = utils.order_table(pd.DataFrame(tag_tables[group].keys(), columns=["Language"]),
                                  experiment=experiment).iloc[:,0].values # Order as usual

        for lang in langs:
            if len(tag_tables[group][lang].keys()) > 1:
                for i, dataset in enumerate(["train", "dev", "test", "total"]):
                    dcol = tag_tables[group][lang][dataset].shape[1] + 1
                    tag_tables[group][lang][dataset].to_excel(writer, index=False, sheet_name=lang,
                                                              startcol=i * dcol, startrow=1)
                    worksheet = writer.sheets[lang]
                    worksheet.merge_range(0, i * dcol, 0, (i + 1) * dcol - 2, dataset.upper(), merge_format)
                    worksheet.set_column((i * dcol) + 2, (i * dcol) + 3, 15, percentage_format)
            else:
                i = 0
                dataset = list(tag_tables[group][lang].keys())[0]
                dcol = tag_tables[group][lang][dataset].shape[1] + 1
                tag_tables[group][lang][dataset].to_excel(writer, index=False, sheet_name=lang, startrow=1)
                worksheet = writer.sheets[lang]
                worksheet.merge_range(0, i * dcol, 0, (i + 1) * dcol - 2, dataset.upper(), merge_format)
                worksheet.set_column((i * dcol) + 2, (i * dcol) + 3, 15, percentage_format)

            # Insert plot
            worksheet.insert_image(tag_tables[group][lang][dataset].shape[0] + 2, 0,
                                   img_path + "langs/pos_tags_plot_{}.png".format(lang.lower()),
                                   {"x_scale": 0.75, "y_scale": 0.75})

        # Group average sheet
        group_avgs = calculate_group_avg_tables(tag_tables[group])

        for i, dataset in enumerate(["train", "dev", "test", "total"]):
            dcol = group_avgs[dataset].shape[1] + 1
            group_avgs[dataset].to_excel(writer, index=False, sheet_name=group,
                                         startcol=i * dcol, startrow=1)
            worksheet = writer.sheets[group]
            worksheet.merge_range(0, i * dcol, 0, (i + 1) * dcol - 2, dataset.upper(), merge_format)
            worksheet.set_column((i * dcol) + 1, (i * dcol) + 2, 15, percentage_format)

        worksheet.set_tab_color(group_to_color[group]) # Special color for group sheet
        # Insert plot
        worksheet.insert_image(group_avgs[dataset].shape[0] + 2, 0,
                               img_path + "groups/pos_tags_plot_{}.png".format(group.lower()),
                               {"x_scale": 0.75, "y_scale": 0.75})

        writer.close()

def make_plot_table(lang_tables, n, nested=True, include_total=True):
    def get_tag_order(lang_tables, n):
        tag_order = []
        for row_tags in zip(*[df.loc[:n, "Tag"] for df in lang_tables.values()]):
            for tag in np.array(Counter(row_tags).most_common(None))[:,0].tolist():
                if tag not in tag_order:
                    tag_order.append(tag)
        return tag_order[:n]

    def find_freq_col(table):
        r = re.compile(".*Count\(%\)")
        return list(filter(r.match, table.columns))[0]

    tag_order = get_tag_order(lang_tables, n)
    tag_order.append("OTHERS")
    plot_table = pd.DataFrame({"Tag": tag_order})
    freq_col = find_freq_col(list(lang_tables.values())[0])

    if nested:
        if len(lang_tables.keys()) > 1:
            datasets = include_total * ["total"] + ["test", "dev", "train"]
        else:
            datasets = lang_tables.keys()
        for dataset in datasets:
            df = lang_tables[dataset]
            selected = df.set_index("Tag").loc[tag_order[:-1]]
            plot_table[dataset.capitalize()] = [*selected[freq_col], 100 - selected[freq_col].sum()]
    else:
        langs = lang_tables.keys()
        for lang in langs:
            df = lang_tables[lang]
            selected = df.set_index("Tag").loc[tag_order[:-1]]
            freq_col = find_freq_col(df)
            plot_table[lang] = [*selected[freq_col], 100 - selected[freq_col].sum()]

    plot_table = plot_table.replace("_", "MULTI")
    return plot_table

def make_plot(table_plot, lang=None, path=None, extra_ysize=0, ax=None, xlabel=None, title=None,
              width=0.5, adjust_size=False, use_tex=False, save=False, labels_dy=0):
    sns.set()
    sns.set_style("ticks")
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)
    plt.rc("axes", labelsize=16)
    if use_tex:
        plt.rcParams["text.usetex"] = True

    if adjust_size:
        figsize = (18, table_plot.shape[1] + extra_ysize)
    else:
        figsize = None

    g = table_plot.set_index("Tag").rename(columns={"Total": "TOTAL"}).T.plot(kind="barh", stacked=True, colormap="crest_r",
                                                                              figsize=figsize,
                                                                              xlim=(0, 100), ax=ax, width=width)
    for i, row in table_plot.iterrows():
        cumulative = table_plot.iloc[:i, 1:].sum().values
        current = table_plot.iloc[i, 1:].values
        for p, y in zip(row.iloc[1:], range(len(current))):
            if row[y+1] >= 1:
                x = cumulative[y] + current[y] / 2
                v_align = "center"
                rotation = None
                dx1, dy1, dx2, dy2 = (0,) * 4
                if current[y] < 3 and table_plot.iloc[i-1, y+1] < 4 and table_plot.iloc[i-1, y+1] > 0:
                    v_align = "bottom"
                    rotation = 30
                    dx1 = 1.5
                    dy1 = -0.15
                g.text(x=x+dx1, y=y+0.375+dy1+labels_dy, s=row["Tag"], fontsize=16, fontstretch="condensed",
                       horizontalalignment="center", verticalalignment=v_align, rotation=rotation)
                if current[y] < 3:
                    dy2 = -0.375
                    dx2 = 0.2
                g.text(x=x+dx2, y=y+dy2, s="{:.1f}".format(current[y]) + use_tex*"\\" + "%",
                       fontsize=16, fontstretch="condensed",
                       horizontalalignment="center", verticalalignment="center",
                       bbox=dict(boxstyle="round, pad=0.15",
                                 fc=(1, 1, 1, 0.5),
                                 ec="none"))
    g.set_xticks(range(0, 101, 10))
    if xlabel is not None:
        g.set(xlabel = xlabel)
    g.legend().remove()
    if title is not None:
        g.set_title(title, fontsize=28, pad=10, color="grey")
    sns.despine(ax=g)

    if save:
        g.figure.savefig(path + "pos_tags_plot_{}.png".format(lang.lower()), dpi=400)
        plt.close(g.figure)

    return g

def export_all_plots(tag_tables, path):
    for group in tqdm(tag_tables.keys()):
        for lang in tag_tables[group].keys():
            table_plot = make_plot_table(tag_tables[group][lang], 5)
            make_plot(table_plot, lang, path + "langs/", xlabel="Cumulative Frequency (\%)",
                      title=lang, adjust_size=True, use_tex=True, save=True)

        group_tables = calculate_group_avg_tables(tag_tables[group])
        table_plot = make_plot_table(group_tables, 5)
        make_plot(table_plot, group, path + "groups/", xlabel="Cumulative Frequency (\%)",
                  title=group, adjust_size=True, use_tex=True, save=True)