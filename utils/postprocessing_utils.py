import numpy as np
import pandas as pd
import xlsxwriter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import sys
sys.path.append("..")
import utils.utils as utils
from data_preparation.data_preparation_pos import read_conll

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

            df = utils.order_table(df)
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
            worksheet = self.write_to_sheet(output2, worksheet, start=df.shape[0] + space)

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
            worksheet = self.write_to_sheet(output3, worksheet, start=df.shape[0] + df_by_test_group.shape[0] + space * 2)

            # Mean over others for every column
            start = 1 - space
            for i, table in enumerate([output1, output2, output3]):
                start += table.shape[0] + space
                worksheet = self.write_to_sheet(self.calc_mean_over_others(table), worksheet,
                                                start=start, header=False)

        self.workbook.close()

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