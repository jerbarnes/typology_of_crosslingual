import numpy as np
import pandas as pd

import sys
sys.path.extend(["..", "../.."])
from utils import utils, model_utils
from data_preparation import data_preparation_pos

def sentiment_examples_per_maxlength(info, d, included_langs, tokenizer, lengths=[64, 128, 256, 512]):
    file_path = info["file_path"]
    lang_name = info["lang_name"]
    dataset = info["dataset"]

    # Only for train set
    if lang_name in included_langs and dataset == "train":
        df = pd.read_csv(file_path, header=None)
        df.columns = ["sentiment", "review"]
        d[lang_name] = {}
        d[lang_name]["total"] = df.shape[0] # Total examples
        for l in lengths:
            # How many examples below limit
            n = (df["review"].apply(lambda x: len(tokenizer.encode(x))) <= l).sum()
            d[lang_name][l] = n
    return d

def pos_examples_per_maxlength(info, d, included_langs, tokenizer, lengths=[64, 128, 256, 512]):
    file_path = info["file_path"]
    lang_name = info["lang_name"]
    dataset = info["dataset"]

    # Only for train set
    if lang_name in included_langs and dataset == "train":
        data = data_preparation_pos.read_conll(file_path)
        d[lang_name] = {}
        d[lang_name]["total"] = len(data[0])
        for l in lengths:
            examples = [{"id": sent_id, "tokens": tokens, "tags": tags} for sent_id, tokens, tags in zip(data[0],
                                                                                                         data[1],
                                                                                                         data[2])]
            examples = [example for example in examples if len(tokenizer.subword_tokenize(example["tokens"],
                                                                                          example["tags"])[0]) <= l]
            d[lang_name][l] = len(examples)
    return d

def build_lengths_table(task, short_model_name, experiment, lengths=[64, 128, 256, 512], save_to=None):
    data_path = utils.find_relative_path_to_root()
    if task == "sentiment":
        data_path += "data/sentiment/"
        f = sentiment_examples_per_maxlength
    elif task == "pos":
        data_path += "data/ud/"
        f = pos_examples_per_maxlength
    included_langs = utils.get_langs(experiment)
    tokenizer = model_utils.get_tokenizer(short_model_name, task)

    results = utils.run_through_data(data_path, f, {}, included_langs=included_langs,
                                     tokenizer=tokenizer, lengths=lengths)
    table = pd.DataFrame(results).T.rename_axis("language").reset_index()
    table = utils.order_table(table, "acl")
    if save_to:
        table.to_excel(save_to, index=False)
    return table