import logging

import sys
sys.path.extend(["..", "../.."])
from utils import utils, model_utils
from data_exploration import basic_stats, lengths, sentiment_balance, tag_stats, tokenizer_stats

def calc_stats(task, experiment, save_to):
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR) # Avoid max length warning
    included_langs = utils.get_langs(experiment)
    models = ["mbert", "xlm-roberta"]

    for short_model_name in models:
        tokenizer = model_utils.get_tokenizer(short_model_name, task)
        # Basic stats
        print("Calculating basic stats for {}...".format(short_model_name))
        basic_stats_table = basic_stats.build_stats_table(task, included_langs, tokenizer, experiment)
        basic_stats_table.to_excel(save_to["basic_stats_" + short_model_name], index=False)
        # Lengths
        print("Calculating lengths for {}...".format(short_model_name))
        lengths_table = lengths.build_lengths_table(task, short_model_name, experiment,
                                                    save_to=save_to["lengths_" + short_model_name])
        # Tokenizer stats
        print("Calculating tokenizer stats for {}...".format(short_model_name))
        toke_stats_table = tokenizer_stats.build_tokenizer_stats_table(task, short_model_name, included_langs, tokenizer)
        toke_stats_table.to_excel(save_to["tokenizer_stats_" + short_model_name], index=False)

    if task == "sentiment":
        # Positive/negative balance
        print("Calculating sentiment balance...")
        balance_table = sentiment_balance.build_balance_table(experiment, save_to=save_to["balance"])
    elif task == "pos":
        # Multiwords
        print("Calculating multiword stats...")
        multiwords_table = tag_stats.build_multiwords_table(experiment, save_to=save_to["multiwords"])

        # Tag stats
        print("Calculating tag stats...")
        tag_tables = tag_stats.build_tag_tables(experiment)
        tag_stats.export_all_plots(tag_tables, path=save_to["tag_stats_img_path"])
        tag_stats.export_tag_tables(tag_tables, experiment, output_path=save_to["tag_stats"],
                                    img_path=save_to["tag_stats_img_path"])
