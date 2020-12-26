import numpy as np
import pandas as pd
import glob
import re
from tqdm.notebook import tqdm

import sys
sys.path.append("..")
from utils import utils, plotting_utils, find_all_colnames, pos_utils
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

def eval_all_dev(checkpoints_path, task, experiment, load_from=None, num_models=2):
    # Setup
    langs = utils.get_langs(experiment)
    data_path = utils.find_relative_path_to_root()
    if task == "pos":
        data_path += "data/ud/"
    elif task == "sentiment":
        data_path += "data/sentiment/"
    tagset = pos_utils.get_ud_tags()
    # Most parameters do not matter, but we need them to build the model
    params = {"max_length": 256,
              "train_batch_size": 8,
              "learning_rate": 2e-5,
              "epochs": 30,
              "tagset": tagset,
              "num_labels": len(tagset),
              "eval_batch_size": 64}
    if load_from:
        output = pd.read_excel(load_from).to_dict(orient="list")
    else:
        output = {"Language": [], "Model": [], "Dev_score": []}

    for lang_name in tqdm(langs):
        # Check if the language is already done
        if output["Language"].count(lang_name) == num_models:
            print("Already evaluated", lang_name)
            continue

        training_lang = utils.name_to_code[lang_name]
        weight_files = glob.glob(checkpoints_path + "{}/*{}.hdf5".format(training_lang, task))
        for file in weight_files:
            model_name = re.search(r"[^a-z]([a-z-]+)_[a-z]+\.", file).group(1)
            if model_name == "bert-base-multilingual-cased":
                short_model_name = "mbert"
            elif model_name == "tf-xlm-roberta-base":
                short_model_name = "xlm-roberta"
            # Check if this model has already been tested for the language
            if (len(output["Language"]) > 0) and (output["Language"][-1] == lang_name) and (output["Model"][-1] == model_name):
                print("Already evaluated {} for {}".format(model_name, lang_name))
                continue

            trainer = fine_tuning.Trainer(training_lang, data_path, task, short_model_name)
            trainer.build_model(**params)
            trainer.prepare_data()
            trainer.model.load_weights(file)
            preds = trainer.handle_oom(trainer.manual_predict, trainer.dev_data,
                                       batch_size=trainer.eval_batch_size)
            dev_score = trainer.metric(preds, trainer.dev_data, "dev")
            trainer.model = None
            output["Language"].append(lang_name)
            output["Model"].append(model_name)
            output["Dev_score"].append(dev_score)
            pd.DataFrame(output).to_excel("eval_temp.xlsx", index=False) # Save progress
    return utils.order_table(pd.DataFrame(output), experiment=experiment)