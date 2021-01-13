import numpy as np
import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import sys
sys.path.append("..")
import utils.utils as utils
from data_preparation.data_preparation_pos import read_conll
from utils.plotting_utils import plots

def find_training_langs(table):
    """Return list of training languages from a table."""
    return [col_name for col_name in table.columns if (
                table[col_name].apply(lambda x: isinstance(x, (np.floating, float))).all()
            )]

def reorder_columns(table):
    """Reorder training languages (columns) to match the order of testing languages (rows)."""
    lang_column = utils.find_lang_column(table)
    training_langs = find_training_langs(table)
    training_langs.sort() # Sort so that we can compare both
    testing_langs = table[lang_column].values.tolist()
    testing_langs.sort()
    assert training_langs == testing_langs, "Training language columns are missing"
    return table[[lang_column] + table[lang_column].values.tolist()]

def fill_missing_columns(table):
    """Fill missing training language columns with NaNs."""
    training_langs = find_training_langs(table)
    missing_langs = np.setdiff1d(table[utils.find_lang_column(table)], training_langs)
    table[missing_langs] = pd.DataFrame([[np.nan] * len(missing_langs)], index=table.index)
    return table

def mean_exclude_by_group(table):
    """Make table with average metrics per testing group (excluding the training language)."""
    table_by_test_group = pd.DataFrame({"Group": ["Fusional", "Isolating", "Agglutinative", "Introflexive"]})

    for train_lang in find_training_langs(table):
        metric_avgs = []
        for lang_group in table_by_test_group["Group"]:
            avg = table[(table["Group"] == lang_group) & (table["Language"] != train_lang)][train_lang].mean()
            metric_avgs.append(avg)
        table_by_test_group[train_lang] = metric_avgs

    return table_by_test_group

def mean_exclude(table):
    """Return mean metrics per testing language (excluding its own model)."""
    lang_cols = table.columns[1:]
    means = []
    for i, row in table.iterrows():
        row_mean = row[[col for col in lang_cols if col != row.iloc[0]]].mean()
        means.append(row_mean)
    return means

def retrieve_results(file_path, skip=3):
    """Return dictionary where every entry corresponds to a metric and contains its corresponding tables."""
    results = pd.read_excel(file_path, sheet_name=None, header=None)
    output = {}

    for metric, df in results.items():
        table_names = [
            "langvlang",
            "langvgroup",
            "groupvgroup",
        ]

        tables = {}
        start = 0
        end = df.shape[1] - 1
        for name in table_names:
            temp = df.loc[start:end]
            temp.columns = temp.iloc[0].values
            temp = temp.drop(temp.index[0])
            start = end + skip + 1
            end = start + 6
            tables[name] = temp.reset_index(drop=True)
        output[metric] = tables
    return output

class Postprocessor:
    def __init__(self, short_model_name, task, experiment):
        self.short_model_name = short_model_name
        self.task = task
        self.experiment = experiment

        # Paths and files
        self.results_path = utils.find_relative_path_to_root() + "results/{}/{}/".format(experiment, short_model_name)
        self.results = pd.read_excel(self.results_path + "results_{}.xlsx".format(task), sheet_name=None)
        baselines_path = utils.find_relative_path_to_root() + "results/{}/".format(experiment)
        self.baselines = pd.read_excel(baselines_path + "baselines_{}.xlsx".format(task), sheet_name=None)

        # Workbook
        self.workbook = xlsxwriter.Workbook(self.results_path + "results_{}_postprocessed.xlsx".format(task))
        self.space = 6

    def __call__(self):
        for sheet_name, df in self.results.items():
            worksheet = self.workbook.add_worksheet(sheet_name)

            df = utils.order_table(df, self.experiment)
            # Add empty column for missing training languages
            df = fill_missing_columns(df)
            # Reorder columns so that they match the order of testing languages
            df = reorder_columns(df)
            # Add language groups
            df = utils.add_lang_groups(df, "Group")
            # Add baseline
            df["Baseline"] = self.baselines[sheet_name]["Baseline"]

            # Change language column name
            output1 = df.rename(columns={utils.find_lang_column(df): "Test\Train"})
            output1 = output1.fillna("-")

            # Write to sheet
            worksheet.set_column(0, 1, 16) # Column width
            worksheet.set_column(1, output1.shape[1], 12)
            worksheet = self.write_to_sheet(output1, worksheet, start=0)

            # Mean of train languages by test language group
            df_by_test_group = mean_exclude_by_group(df).set_index("Group")

            output2 = df_by_test_group.copy()
            output2 = output2.fillna("-").rename_axis("Test\Train").reset_index()
            output2.insert(loc=1, column=None, value=[None]*output2.shape[0])

            # Write to sheet
            worksheet = self.write_to_sheet(output2, worksheet, start=df.shape[0] + self.space)

            # Mean of previous means by train language group
            df_by_both_group = df_by_test_group.drop("Baseline", axis=1)
            df_by_both_group = df_by_both_group.transpose().reset_index().rename(columns={"index": "Train_langs"})
            df_by_both_group = utils.add_lang_groups(df_by_both_group, "Train Group")
            df_by_both_group = df_by_both_group.groupby(["Train Group"]).mean()
            df_by_both_group = df_by_both_group.reindex(["Fusional", "Isolating",
                                                         "Agglutinative", "Introflexive"]).transpose()

            output3 = df_by_both_group.rename_axis("Test\Train")
            output3 = output3.reset_index()
            output3.insert(loc=1, column=None, value=[None]*output3.shape[0])

            # Write to sheet
            worksheet = self.write_to_sheet(output3, worksheet, start=df.shape[0] + df_by_test_group.shape[0] + self.space * 2)

            # Mean over others for every column
            start = 1 - self.space
            for i, table in enumerate([output1, output2, output3]):
                start += table.shape[0] + self.space
                worksheet = self.write_to_sheet(self.calc_mean_over_others(table), worksheet,
                                                start=start, header=False)

        self.workbook.close()
        print("Saved file to " + self.results_path + "results_{}_postprocessed.xlsx".format(self.task))

    def write_to_sheet(self, table, worksheet, start, header=True):
        max_locs = self.row_maxs(table)

        # Column names
        if header:
            for coln in range(table.shape[1]):
                worksheet.write(start, coln, table.columns[coln],
                                self.make_format(cell_value=table.columns[coln], coln=coln))

        # Values
        for rown in range(start + 1, table.shape[0] + start + 1):
            i = rown - start - 1
            for coln in range(table.shape[1]):
                cell_value = table.values[i, coln]
                worksheet.write(rown, coln, cell_value, self.make_format(cell_value=cell_value, coln=coln,
                                                                         row_max=table.iloc[i, max_locs[i]]))

        return worksheet

    def row_maxs(self, table):
        return table.loc[:, find_training_langs(table)].apply(
            lambda x: table.columns.tolist().index(x.astype(float).idxmax()), axis=1
        ).values

    def make_format(self, cell_value, coln, row_max=None):
        color_dict = {
            "Fusional": "#95c78f",
            "Isolating": "#f79d97",
            "Agglutinative": "#abaff5",
            "Introflexive": "#fffecc"
        }
        grey = "#d1d1d1"

        # Default values
        bold = False
        underline = False
        color = 0
        border = 0

        # Alignment
        if coln < 2:
            align = "left"
        else:
            align = "right"

        # String, numeric or NaN
        if isinstance(cell_value, str) and cell_value != "-":
            bold = True
            border = 1
            # Pick color
            if cell_value in color_dict.keys():
                color = color_dict[cell_value]
            elif cell_value in utils.lang_to_group.keys():
                color = color_dict[utils.lang_to_group[cell_value]]
            else:
                color = grey
        elif cell_value == row_max:
            underline = True
            bold = True

        return self.workbook.add_format({"bold": bold, "underline": underline, "align": align,
                                         "num_format": "0.000", "bg_color": color, "border": border})

    def calc_mean_over_others(self, table):
        def mean(x, table):
            if table.columns.get_loc(x.name) == 0:
                return "Mean over others"
            elif (x == "-").all():
                return "-"
            elif x.apply(lambda y: isinstance(y, float)).all():
                return (x[table["Test\Train"] != x.name]).mean()

        return table.apply(lambda x: mean(x, table)).to_frame().T

def pos_baseline(info, baselines, included_langs):
    file_path = info["file_path"]
    lang_name = info["lang_name"]
    dataset = info["dataset"]

    if lang_name in included_langs and dataset == "test":
        conllu_data = read_conll(file_path)
        tags = [tag for taglist in conllu_data[2] for tag in taglist]
        # Accuracy will be the relative frequency of the majority tag
        acc = tags.count(max(set(tags), key=tags.count)) / len(tags)
        baselines.append((lang_name, acc))
    return baselines # Return even if it's unaltered, otherwise it gets overwritten by None

def sentiment_baseline(info, baselines, included_langs):
    file_path = info["file_path"]
    lang_name = info["lang_name"]
    dataset = info["dataset"]
    if lang_name in included_langs and dataset == "test":
        data = pd.read_csv(file_path, header=None)
        data.columns = ["sentiment", "review"]
        # Prediction will be the majority class
        y_true = data["sentiment"].values
        y_pred = [data["sentiment"].mode()[0]] * len(y_true)
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")
        baselines.append((lang_name, acc, precision, recall, f1))
    return baselines

def calc_baselines(task, experiment, save=False):
    path_to_root = utils.find_relative_path_to_root()
    output_filepath = path_to_root + "results/{}/baselines_{}.xlsx".format(experiment, task)
    included_langs = utils.get_langs(experiment)

    if task == "pos":
        baselines = utils.run_through_data(path_to_root + "data/ud/",
                                           pos_baseline,
                                           table=[],
                                           included_langs=included_langs)
        columns = ["Language", "Accuracy"]
    elif task == "sentiment":
        baselines = utils.run_through_data(path_to_root + "data/sentiment/",
                                           sentiment_baseline,
                                           table=[],
                                           included_langs=included_langs)
        columns = ["Language", "Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1"]
    else:
        raise Exception("Invalid task")

    # Prepare table
    baselines = pd.DataFrame(np.array(baselines), columns=columns)
    baselines.iloc[:, 1:] = baselines.iloc[:, 1:].astype(float)
    baselines = utils.order_table(baselines, experiment)

    # Output file
    if save:
        with pd.ExcelWriter(output_filepath) as writer:
            for metric in baselines.columns[1:]:
                baselines[["Language", metric]].rename(
                    columns={metric: "Baseline"}
                ).to_excel(writer, index=False, sheet_name=metric)

    return baselines

def remove_untrained_cols(table):
    return table.drop(table.columns[(table == "-").max()].tolist(), axis=1)

def remove_untrained_rows(table):
    return table[(table != "-").all(axis=1)].reset_index(drop=True)

def group(df, grouped_train, grouped_test):
    if grouped_train and not grouped_test:
        df = df.groupby(["Train-Group", "Test-Language"], as_index=False, sort=False).mean()
    elif not grouped_train and grouped_test:
        df = df.groupby(["Train-Language", "Test-Group"], as_index=False, sort=False).mean()
    elif grouped_train and grouped_test:
        df = df.groupby(["Train-Group", "Test-Group"], as_index=False, sort=False).mean()
    return df

class Metrics:
    def __init__(self, results_dir, experiment, short_model_name, task, metric, skip=3):
        results_filepath = results_dir + "{}/{}/results_{}_postprocessed.xlsx".format(experiment,
                                                                                      short_model_name,
                                                                                      task)
        results = retrieve_results(results_filepath, skip)
        langvlang = results[metric]["langvlang"]
        n_rows = langvlang.iloc[:,0].isnull().argmax()
        self.langvlang = langvlang.iloc[:n_rows, :-1]
        langvgroup = results[metric]["langvgroup"]
        self.langvgroup = langvgroup.iloc[:4, [True] + [False] + [True]*n_rows + [False]]

    def within_score(self, grouped=False):
        df = pd.DataFrame({"Train-Group": self.langvlang.iloc[:, 0], # Bc train and test langs are in the same order
                           "Train-Language": self.langvlang.columns[2:],
                           "Within-Score": np.diagonal(self.langvlang.iloc[:, 2:].values)})
        df = remove_untrained_rows(df)
        df["Within-Score"] = df["Within-Score"].astype(float)

        if grouped:
            # Mean within-score per training group
            df = df.groupby("Train-Group", as_index=False, sort=False).mean()

        return df

    def cross_score(self, grouped_train=False, grouped_test=False):
        df = pd.melt(self.langvlang.rename(columns={"Test\Train": "Test-Language"}),
                     id_vars="Test-Language",
                     value_vars=self.langvlang.columns[2:],
                     var_name="Train-Language",
                     value_name="Cross-Score")
        df = df[["Train-Language", "Test-Language", "Cross-Score"]]
        df = df[df["Train-Language"] != df["Test-Language"]] # Remove within
        df = remove_untrained_rows(df)
        df["Cross-Score"] = df["Cross-Score"].astype(float)
        # Add train and test groups
        df.insert(loc=0, column="Train-Group", value=df["Train-Language"].apply(lambda x: utils.lang_to_group[x]))
        df.insert(loc=2, column="Test-Group", value=df["Test-Language"].apply(lambda x: utils.lang_to_group[x]))

        df = group(df, grouped_train, grouped_test)

        return df

    def transfer_loss(self, grouped_train=False, grouped_test=False):
        df_within = self.within_score()
        df_cross = self.cross_score()
        df = pd.merge(df_within, df_cross.iloc[:, 1:], how="left", on="Train-Language")
        df["Transfer-Loss"] = df["Within-Score"] - df["Cross-Score"]

        df = group(df, grouped_train, grouped_test)

        return df

class Transfer:
    def __init__(self, results_path, skip=3):
        self.results_path = results_path
        self.skip = skip
        self.transfer_loss = {"pos": {}, "sentiment": {}}

    def prepare_tables(self, task, metric, as_percent=False):
        p = 1 + 99 * as_percent

        # Retrieve postprocessed results tables
        results = retrieve_results(self.results_path + "results_{}_postprocessed.xlsx".format(task), self.skip)
        langvlang = results[metric]["langvlang"]
        langvgroup = results[metric]["langvgroup"]

        # Build dataframes
        n_rows = langvlang.iloc[:,0].isnull().argmax()
        df = pd.DataFrame({"Train-Group": langvlang.iloc[:n_rows, 0], "Train-Language": langvlang.columns[2:-1],
                           "Within-Score": np.diagonal(langvlang.iloc[:n_rows, 2:-1].values)})
        df_group = langvgroup.iloc[:4, [True] + [False] + [True]*n_rows + [False]]
        df_group = pd.melt(df_group.rename(columns={"Test\Train": "Test-Group"}), id_vars="Test-Group",
                                           value_vars=df_group.columns[1:],
                                           var_name="Train-Language", value_name="Cross-Score")
        df = remove_untrained_rows(df)
        n_rows = df.shape[0]
        df = pd.merge(df, df_group, on="Train-Language")
        df["Cross-Score"] = df["Cross-Score"].astype(float) * p
        df["Within-Score"] = df["Within-Score"].astype(float) * p

        return df, n_rows

    def calc_transfer_loss(self, task, metric, as_percent=False):
        transfer, n_rows = self.prepare_tables(task, metric, as_percent)

        # Calculate transfer loss
        transfer["Transfer-Loss"] = transfer["Within-Score"] - transfer["Cross-Score"]

        # Make same/other division
        final = transfer[transfer["Test-Group"] != transfer["Train-Group"]].groupby(by=["Train-Group", "Train-Language"],
                                                                                    as_index=False, sort=False).mean()
        final["Test-Group"] = "Others"
        temp = transfer[transfer["Test-Group"] == transfer["Train-Group"]].copy()
        temp.loc[:, "Test-Group"] = "Same"
        final = pd.concat([final, temp], ignore_index=True)
        final["sort"] = np.concatenate((np.arange(1, n_rows * 2, 2), np.arange(0, n_rows * 2, 2)))
        final = final.sort_values("sort").reset_index(drop=True).drop("sort", axis=1)
        final["Relative-Transfer-Loss"] = final["Transfer-Loss"] / final["Within-Score"]
        final_avg = final.groupby(by=["Train-Group", "Test-Group"], as_index=False, sort=False).mean()

        self.transfer_loss[task][metric] = {"all": final, "groups": final_avg}

    def plot_transfer(self, task, metric, grouped=False, extra_fontsize=0, legend_coords=(1, 0.5), xlim=None,
                      yaxis_title="Train", xaxis_title="Transfer Loss",
                      title="Transfer Loss in the Same vs Other Language Groups"):
        if metric not in self.transfer_loss[task].keys():
            raise Exception("Transfer loss has not been calculated for this task and metric.")

        # Prepare seaborn
        plots.prepare_sns()

        # Set parameters
        colors = plots.get_group_colors()
        bar_colors = plots.get_dual_bar_colors(as_dict=True)

        if not grouped:
            df = self.transfer_loss[task][metric]["all"]
            y = "Train-Language"
            height = 12
            aspect = 1.5
            train_groups = df["Train-Group"]
            group_counts = train_groups.value_counts()[train_groups.unique()].values / 2
            label_colors = np.array([np.repeat(color, times).tolist() for color, times in zip(colors, group_counts)]).sum()
        else:
            df = self.transfer_loss[task][metric]["groups"]
            y = "Train-Group"
            height = 4
            aspect = 3.5
            label_colors = colors

        scale = 1 + 99 * (df["Transfer-Loss"] > 1).any()

        # Main plot
        g = sns.catplot(
            data=df, kind="bar", x="Transfer-Loss", y=y, hue="Test-Group",
            height=height, aspect=aspect, palette=bar_colors, saturation=0.3, legend=False
        )

        # Add text
        y_labels = df[y].unique().tolist()

        for y_label in y_labels:
            values = df.loc[(df[y] == y_label), "Transfer-Loss"].values
            dy = [-0.2, 0.2]
            for i, idx in enumerate(df.index[df[y] == y_label]):
                if scale == 100:
                    p = "{:.1f}".format(values[i])
                else:
                    p = "{:.3f}".format(values[i])
                extra_dy = 0.025
                g.ax.text(values[i] + 0.005 * scale, y_labels.index(y_label) + dy[i] + extra_dy, p,
                          verticalalignment="center", horizontalalignment="left",
                          fontsize=18 + extra_fontsize)

            # Difference bar
            bbar = patches.Rectangle((values[0], y_labels.index(y_label) + dy[i] - 0.2), values[1] - values[0], -0.4,
                                      fill=True, color="#a3a3a3", alpha=0.7, ec=None)
            g.ax.add_patch(bbar)
            x = values.mean()
            align = "center"
            diff = values[1] - values[0]
            color = "black"
            if np.abs(diff) <= 0.03 * scale:
                align = "left"
                x = values[0] + 0.03 * scale
            if diff < 0:
                color = "red"
            if scale == 100:
                diff_text = r"\textbf{{{:.1f}}}".format(diff)
            else:
                diff_text = r"\textbf{{{:.3f}}}".format(diff)
            g.ax.text(x, y_labels.index(y_label) + dy[i] + extra_dy - 0.4, diff_text,
                      verticalalignment="center", horizontalalignment=align,
                      fontsize=18 + extra_fontsize, color=color,
                      bbox=dict(boxstyle="round, pad=0.15",
                                         fc=(1, 1, 1, 0.5),
                                         ec="none"))

        # Add y axis labels
        self.add_yaxis_labels(g, label_colors)

        # Add legend
        plots.add_legend(title="Test Language Group", fontsize=20 + extra_fontsize, coords=legend_coords)

        # Titles, xlim, fontsize, etc
        if xlim is None:
            xmax = df["Transfer-Loss"].max()
            xlim = (0, xmax + xmax * 0.05)
        plt.xlim(xlim)
        plt.xticks(np.arange(xlim[0], xlim[1], 0.05 * scale))
        plots.add_titles(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title, fontsize=16 + extra_fontsize)

        plt.show()
        plt.close()

    def add_yaxis_labels(self, fig, label_colors):
        for i, label in enumerate(fig.ax.yaxis.get_ticklabels()):
            label.set_bbox(dict(boxstyle="round,pad=0.85",
                                fc=label_colors[i]))