import numpy as np
import pandas as pd
import sys
sys.path.extend(["..", "../.."])
from utils import utils

def balance(info, d, included_langs):
    file_path = info["file_path"]
    lang_name = info["lang_name"]
    dataset = info["dataset"]

    if lang_name in included_langs and dataset == "train":
        df = pd.read_csv(file_path, header=None)
        df.columns = ["sentiment", "review"]
        total = df.shape[0]
        positives = df["sentiment"].sum()
        negatives = total - positives
        ratio = positives / total
        balanced_total = min(positives, negatives) * 2
        d[lang_name] = {"Positive": positives,
                        "Negative": negatives,
                        "Total": total,
                        "Ratio": ratio,
                        "Balanced_total": balanced_total}
    return d

def build_balance_table(experiment, save=False):
    included_langs = utils.get_langs(experiment)
    table = utils.run_through_data(utils.find_relative_path_to_root() + "data/sentiment/",
                                   balance, table={}, included_langs=included_langs)
    table = pd.DataFrame(table).T.rename_axis("Language").reset_index()
    table = table.astype(dict.fromkeys(["Positive", "Negative", "Total", "Balanced_total"], int))
    table = utils.order_table(table, experiment=experiment)
    if save:
        table.to_excel(input("Save to: "), index=False)
    return table