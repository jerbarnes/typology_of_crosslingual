import pandas as pd
import glob
import re
from tqdm.notebook import tqdm
from pathlib import Path

def make_lang_code_dicts():
    file_path = Path(__file__).parent / "../utils/lang_codes.xlsx"
    lang_codes = pd.read_excel(file_path, header=0)
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

def convert_table_to_latex(table):
    table = order_table(table) # In case it's not already in correct order
    
    # Retrieve language groups in correct order and add them to table
    file_path = Path(__file__).parent / "../data_exploration/pos_table.txt"
    file = open(file_path, "r")
    lang_groups = [line.split("&")[0].strip() for line in file.readlines()]
    table.insert(loc=0, column="group", value=lang_groups)
    
    # Latex output
    print("\n".join([" & ".join(line) + r"\\" for line in table.astype(str).values]))
    
    # Pandas output
    return table

def run_through_data(data_path, f, table=None):
    code_dicts = make_lang_code_dicts()
    code_to_name = code_dicts["code_to_name"]
    name_to_code = code_dicts["name_to_code"]
    # Find all data files in path
    data_files = glob.glob(data_path + "*/*.csv") + glob.glob(data_path + "*/*.conllu")
    task = data_path.split("/")[-2]
    assert task in ["ud", "sentiment"], data_path + " is not a valid data path."

    for file_path in tqdm(data_files):
        lang_code = file_path.split("\\")[1]
        lang_name = code_to_name[lang_code]
        dataset = re.findall(r"[a-z]+\.", file_path)[0][:-1]  
        table = f(file_path, lang_name, lang_code, dataset, table)
    return table