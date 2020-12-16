import numpy as np
import pandas as pd
import os
import itertools
from collections import Counter

import sys
sys.path.append("..")
from utils import utils, model_utils
from data_preparation.data_preparation_pos import read_conll

def pos_stats(info, table, tokenizer, included_langs):
    file_path = info["file_path"]
    lang_name = info["lang_name"]
    dataset = info["dataset"]
    d = {}

    if lang_name in included_langs:
        conllu_data = read_conll(file_path)
        examples = [{"id": sent_id, "tokens": tokens, "tags": tags} for sent_id, tokens, tags in zip(conllu_data[0],
                                                                                                     conllu_data[1],
                                                                                                     conllu_data[2])]

        # Number of examples in dataset
        if table["language"].isna().all() or lang_name not in table["language"].values:
            d["language"] = lang_name
            index = table.index[table["language"].isna()][0]
        else:
            index = table.index[table["language"] == lang_name][0]
        d[dataset + "_examples"] = [len(examples)]

        # Avg tokens
        tokens, lengths = [], []
        for e in examples:
            tokenized = tokenizer.subword_tokenize(e["tokens"], e["tags"])[0]
            tokens.extend(tokenized)
            lengths.append(len(tokenized))
        d[dataset + "_avg_tokens"] = [np.array(lengths).mean()]

        # Hapaxes
        counts = np.array(list(Counter(tokens).items()))
        hapaxes = counts[counts[:,1] == "1"][:,0]
        d[dataset + "_hapaxes"] = [len(hapaxes)]
        d[dataset + "_hapaxes(%)"] = [len(hapaxes) / len(tokens) * 100]

        # Unknown
        unk = (np.array(tokens) == "[UNK]").sum()
        d[dataset + "_unknown"] = [unk]
        d[dataset + "_unknown(%)"] = [unk / len(tokens) * 100]

        table.update(pd.DataFrame(d, index=[index]))

    return table

def sentiment_stats(info, table, tokenizer, included_langs):
    file_path = info["file_path"]
    lang_name = info["lang_name"]
    dataset = info["dataset"]
    d = {}

    if lang_name in included_langs:
        data = pd.read_csv(file_path, header=None)
        data.columns = ["sentiment", "review"]

        # Number of examples in dataset
        if table["language"].isna().all() or lang_name not in table["language"].values:
            d["language"] = lang_name
            index = table.index[table["language"].isna()][0]
        else:
            index = table.index[table["language"] == lang_name][0]
        d[dataset + "_examples"] = [data.shape[0]]

        # Avg tokens
        tokens, lengths = [], []
        for e in data["review"]:
            tokenized = tokenizer.encode(e)
            tokens.extend(tokenized)
            lengths.append(len(tokenized))
        d[dataset + "_avg_tokens"] = [np.array(lengths).mean()]

        # Hapaxes
        counts = np.array(list(Counter(tokens).items()))
        hapaxes = counts[counts[:,1] == 1][:,0]
        d[dataset + "_hapaxes"] = [len(hapaxes)]
        d[dataset + "_hapaxes(%)"] = [len(hapaxes) / len(tokens) * 100]

        # Unknown
        unk = (np.array(tokens) == 100).sum()
        d[dataset + "_unknown"] = [unk]
        d[dataset + "_unknown(%)"] = [unk / len(tokens) * 100]

        table.update(pd.DataFrame(d, index=[index]))

    return table

def build_stats_table(task, included_langs, tokenizer):
    funcs = {"pos": pos_stats, "sentiment": sentiment_stats}
    data_path = {"pos": "data/ud/", "sentiment": "data/sentiment/"}
    path_to_root = utils.find_relative_path_to_root()
    names = ["train", "dev", "test"]
    names_examples = (np.array(names, dtype=object) + "_examples").tolist()
    names_avg = (np.array(names, dtype=object) + "_avg_tokens").tolist()
    names_hapaxes = np.array(list(itertools.product(names, ["_hapaxes", "_hapaxes(%)"])), dtype=object)
    names_hapaxes = (names_hapaxes[:,0] + names_hapaxes[:,1]).tolist()
    names_unknown = np.array(list(itertools.product(names, ["_unknown", "_unknown(%)"])), dtype=object)
    names_unknown = (names_unknown[:,0] + names_unknown[:,1]).tolist()
    colnames = ["language"] + names_examples + names_avg + names_hapaxes + names_unknown
    values = np.empty((len(included_langs), len(colnames)))
    values[:] = np.nan

    table = utils.run_through_data(path_to_root + data_path[task],
                                   funcs[task],
                                   pd.DataFrame(values, columns=colnames),
                                   tokenizer=tokenizer,
                                   included_langs=included_langs)

    table = utils.order_table(table, experiment="acl")
    table = table.astype(dict.fromkeys([col for col in table.columns[1:] if "%" not in col and "avg" not in col],
                                        pd.Int64Dtype())) # Convert to int
    return table