import basic_stats, lengths, sentiment_balance, tag_stats
import logging

import sys
sys.path.extend(["..", "../.."])
from utils import utils, model_utils

def calc_stats(task, experiment):
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR) # Avoid max length warning
    included_langs = utils.get_langs(experiment)
    models = ["mbert", "xlm-roberta"]

    for short_model_name in models:
        tokenizer = model_utils.get_tokenizer(short_model_name, task)
        # Basic stats
        print("Calculating basic stats for {}...".format(short_model_name))
        basic_stats_table = basic_stats.build_stats_table(task, included_langs, tokenizer)
        basic_stats_table.to_excel(input("Save to: "), index=False)
        # Lengths
        print("Calculating lengths for {}...".format(short_model_name))
        lengths_table = lengths.build_lengths_table(task, short_model_name, experiment, save=True)

    if task == "sentiment":
        # Positive/negative balance
        print("Calculating sentiment balance...")
        balance_table = sentiment_balance.build_balance_table(experiment, save=True)
    elif task == "pos":
        # Tag stats
        print("Calculating tag stats...")
        tag_tables = tag_stats.build_tag_tables(experiment)
        img_path = input("Save plots to: ")
        tag_stats.export_all_plots(tag_tables, img_path)
        tag_stats.export_tag_tables(tag_tables, experiment, input("Save file to: "), img_path)
