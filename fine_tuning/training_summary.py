import numpy as np
import pandas as pd

import sys
sys.path.append("..")
from utils import utils, plotting_utils, find_all_colnames
import fine_tuning

def make_train_summary_table(checkpoints_path, task, experiment, save_to=None):
    # Get scores and other info
    table = fine_tuning.get_fine_tune_scores(task, checkpoints_path)
    table = utils.order_table(table, experiment)

    # Merge other useful columns
    find_all_colnames.run()
    files = ["../data_exploration/{}/tables/basic_stats_{}_mbert.xlsx".format(experiment, task)]
    columns = ["train_examples"]

    if task == "sentiment":
        files.extend(["../data_exploration/{}/tables/sentiment_balance.xlsx".format(experiment),
                      "../fine_tuning/class_weights_{}.xlsx".format(experiment),
                      "../fine_tuning/class_weights_{}.xlsx".format(experiment),
                      "../data_exploration/{}/tables/sentiment_balance.xlsx".format(experiment)])
        columns.extend(["Ratio", "Negative", "Positive", "Balanced_total"])

    for path, col in zip(files, columns):
        aux_table = pd.read_excel(path)
        table = utils.merge_tables(table, aux_table, how="left", cols_table2=[col])

    # Model-specific info
    basic_stats_mbert = pd.read_excel(
        "../data_exploration/{}/tables/basic_stats_{}_mbert.xlsx".format(experiment, task)
    )
    basic_stats_xlm_roberta = pd.read_excel(
        "../data_exploration/{}/tables/basic_stats_{}_xlm-roberta.xlsx".format(experiment, task)
    )
    basic_stats_mbert["model_name"] = "bert-base-multilingual-cased"
    basic_stats_xlm_roberta["model_name"] = "tf-xlm-roberta-base"
    basic_stats = pd.concat([basic_stats_mbert, basic_stats_xlm_roberta])
    table = pd.merge(table,
                     basic_stats[["language", "model_name", "train_avg_tokens"]],
                     how="left",
                     left_on=["training_lang", "model_name"],
                     right_on=["language", "model_name"]).drop("language", axis=1)

    # Make column for actual number of training examples used
    def get_real_train_size(x):
        return x["train_examples"] if x["use_class_weights"] else x["Balanced_total"]
    table["real_train_size"] = table[["use_class_weights", "train_examples", "Balanced_total"]].apply(
        get_real_train_size, axis=1
    )

    # Rename columns and save
    table = table.rename(columns={"Ratio": "pos_ratio",
                                  "Negative": "neg_class_weight",
                                  "Positive": "pos_class_weight",
                                  "Balanced_total": "balanced_train_examples"})
    if save_to:
        table.to_excel(save_to, index=False)
    return table
