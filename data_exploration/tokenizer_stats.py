import numpy as np
import pandas as pd
import string

import sys
sys.path.extend(["..", "../.."])
from utils import utils, model_utils
from data_preparation.data_preparation_pos import read_conll

def count_mbert(s, tokenizer=None):
    words = 0
    tokens = 0
    split_words = 0
    chars = 0
    if tokenizer is None:
        tokenizer = model_utils.get_tokenizer("mbert", "pos")
    tokenized_string = tokenizer.tokenize(s)
    previous_token = ""

    for token in tokenized_string:
        tokens += 1
        chars += len(token)
        if token.startswith("##"):
            chars -= 2 # Discount the "##" marker
            if not previous_token.startswith("##"):
                split_words += 1
        elif not token.startswith("##"):
            words += 1
        previous_token = token

    return words, tokens, split_words, chars

def count_xlm_roberta(s, tokenizer=None):
    words = 0
    tokens = 0
    split_words = 0
    chars = 0
    if tokenizer is None:
        tokenizer = model_utils.get_tokenizer("xlm-roberta", "pos")
    tokenized_string = tokenizer.tokenize(s)
    previous_token = ""

    for token in tokenized_string:
        tokens += 1
        chars += len(token)
        if token not in string.punctuation and not token.startswith("▁") and previous_token.startswith("▁"):
            split_words += 1
        elif token.startswith("▁"):
            words += 1
            chars -= 1 # Discount "▁" marker
        elif token in string.punctuation:
            words += 1
        previous_token = token

    return words, tokens, split_words, chars

def pos_tokenizer_stats(info, d, tokenizer, included_langs, short_model_name):
    file_path = info["file_path"]
    lang_name = info["lang_name"]
    dataset = info["dataset"]

    if lang_name in included_langs:
        if lang_name not in d.keys():
            d[lang_name] = {}
        conllu_data = read_conll(file_path)
        conllu_token_lists = conllu_data[1]
        words = 0
        tokens = 0
        split_words = 0
        chars = 0

        # Count totals
        for token_list in conllu_token_lists:
            for t in token_list:
                if short_model_name == "mbert":
                    temp_words, temp_tokens, temp_split_words, temp_chars = count_mbert(t, tokenizer)
                elif short_model_name == "xlm-roberta":
                    temp_words, temp_tokens, temp_split_words, temp_chars = count_xlm_roberta(t, tokenizer)
                words += temp_words
                tokens += temp_tokens
                split_words += temp_split_words
                chars += temp_chars

        # Make averages
        d[lang_name]["avg_words_{}".format(dataset)] = words / len(conllu_token_lists)
        d[lang_name]["avg_split_words_{}(%)".format(dataset)] = split_words / words * 100
        d[lang_name]["avg_subwords_per_word_{}".format(dataset)] = tokens / words
        d[lang_name]["avg_chars_{}".format(dataset)] = chars / len(conllu_token_lists)
        d[lang_name]["avg_chars_per_word_{}".format(dataset)] = chars / words

    return d

def sentiment_tokenizer_stats(info, d, tokenizer, included_langs, short_model_name):
    file_path = info["file_path"]
    lang_name = info["lang_name"]
    dataset = info["dataset"]

    if lang_name in included_langs:
        if lang_name not in d.keys():
            d[lang_name] = {}
        data = pd.read_csv(file_path, header=None)
        data.columns = ["sentiment", "review"]
        words = 0
        tokens = 0
        split_words = 0
        chars = 0

        # Count totals
        for text in data["review"]:
            if short_model_name == "mbert":
                temp_words, temp_tokens, temp_split_words, temp_chars = count_mbert(text, tokenizer)
            elif short_model_name == "xlm-roberta":
                temp_words, temp_tokens, temp_split_words, temp_chars = count_xlm_roberta(text, tokenizer)
            words += temp_words
            tokens += temp_tokens
            split_words += temp_split_words
            chars += temp_chars

        # Make averages
        d[lang_name]["avg_words_{}".format(dataset)] = words / data.shape[0]
        d[lang_name]["avg_split_words_{}(%)".format(dataset)] = split_words / words * 100
        d[lang_name]["avg_subwords_per_word_{}".format(dataset)] = tokens / words
        d[lang_name]["avg_chars_{}".format(dataset)] = chars / data.shape[0]
        d[lang_name]["avg_chars_per_word_{}".format(dataset)] = chars / words

    return d

def build_tokenizer_stats_table(task, short_model_name, included_langs, tokenizer):
    funcs = {"pos": pos_tokenizer_stats, "sentiment": sentiment_tokenizer_stats}
    data_path = {"pos": "data/ud/", "sentiment": "data/sentiment/"}
    path_to_root = utils.find_relative_path_to_root()

    d = utils.run_through_data(path_to_root + data_path[task],
                               funcs[task],
                               {},
                               short_model_name=short_model_name,
                               tokenizer=tokenizer,
                               included_langs=included_langs)

    table = pd.DataFrame(d).T.rename_axis(index="language").reset_index()
    return table