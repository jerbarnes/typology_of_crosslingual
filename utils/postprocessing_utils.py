import numpy as np
import pandas as pd
import utils.utils as utils

def find_training_langs(table):
    return [col_name for col_name in table.columns if (table[col_name].apply(lambda x: isinstance(x, (np.floating, float))).all())]

def reorder_columns(table):
    lang_column = utils.find_lang_column(table)
    training_langs = find_training_langs(table)
    training_langs.sort()
    testing_langs = table[lang_column].values.tolist()
    testing_langs.sort()
    assert training_langs == testing_langs, "Training language columns are missing"
    return table[[lang_column] + table[lang_column].values.tolist()]

def fill_missing_columns(table):
    training_langs = find_training_langs(table)
    missing_langs = np.setdiff1d(table[utils.find_lang_column(table)], training_langs)
    table[missing_langs] = pd.DataFrame([[np.nan] * len(missing_langs)], index=table.index)
    return table

def mean_exclude_by_group(table):
    table_by_test_group = pd.DataFrame({"Group": ["Fusional", "Isolating", "Agglutinative", "Introflexive"]})

    for train_lang in find_training_langs(table):
        metric_avgs = []
        for lang_group in table_by_test_group["Group"]:
            avg = table[(table["Group"] == lang_group) & (table["Language"] != train_lang)][train_lang].mean()
            metric_avgs.append(avg)
        table_by_test_group[train_lang] = metric_avgs

    return table_by_test_group
