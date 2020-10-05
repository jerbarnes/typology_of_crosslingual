import pandas as pd
from pathlib import Path

def make_lang_code_dicts(lang_codes_path):
    lang_codes = pd.read_excel(lang_codes_path, header=0)
    return {"code_to_name": pd.Series(lang_codes["English name of Language"].values,index=lang_codes["ISO 639-1 Code"]).to_dict(),
            "name_to_code": pd.Series(lang_codes["ISO 639-1 Code"].values,index=lang_codes["English name of Language"]).to_dict()}

def order_table(table):
    # Make sure the path is correct even when importing this function from somewhere else
    file_path = Path(__file__).parent / "../data_exploration/pos_table.txt"
    file = open(file_path, "r")
    lang_order = [line.split("&")[1].strip() for line in file.readlines()]
    table["sort"] = table[table.columns[0]].apply(lambda x: lang_order.index(x))
    table = table.sort_values(by=["sort"]).drop("sort", axis=1).reset_index(drop=True)
    return table