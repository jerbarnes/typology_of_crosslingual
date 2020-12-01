import pandas as pd
import numpy as np
import glob
import re
from tqdm.notebook import tqdm
from pathlib import Path

def make_lang_code_dicts():
    file_path = Path(__file__).parent / "../utils/lang_codes.xlsx"
    lang_codes = pd.read_excel(file_path, header=0)
    return {"code_to_name": pd.Series(lang_codes["English name of Language"].values,index=lang_codes["ISO 639-1 Code"]).to_dict(),
            "name_to_code": pd.Series(lang_codes["ISO 639-1 Code"].values,index=lang_codes["English name of Language"]).to_dict()}

def make_lang_group_dict():
    file_path = Path(__file__).parent / "../data_exploration/pos_table.txt"
    file = open(file_path, "r")
    lang_to_group = {}
    for line in file.readlines():
        group_latex = line.split("&")[0].strip()
        lang = line.split("&")[1].strip()
        match = re.search("{(.*)}", group_latex)
        lang_to_group[lang] = match.group(1)
    return lang_to_group

def order_table(table):
    # Make sure the path is correct even when importing this function from somewhere else
    file_path = Path(__file__).parent / "../data_exploration/pos_table.txt"
    file = open(file_path, "r")
    lang_colname = find_lang_column(table)
    current_langs = table[lang_colname].values
    lang_order = []
    for line in file.readlines():
        lang = line.split("&")[1].strip()
        if lang in current_langs: # Not all languages need to be present
            lang_order.append(lang)
    if isinstance(table.columns, pd.MultiIndex): # Check for hierarchical columns
        level = 0
    else:
        level = None
    table.insert(0, "sort", table[lang_colname].apply(lambda x: lang_order.index(x)))
    table = table.sort_values(by=["sort"]).drop("sort", axis=1, level=level).reset_index(drop=True)
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

def run_through_data(data_path, f, table=None, **kwargs):
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
        match = re.findall(r"[a-z]+\.", file_path)
        if match and match[0][:-1] in ["train", "dev", "test"]:
            dataset = match[0][:-1]
        else:
            print(file_path, "is not a valid data path, skipping")
            continue
        table = f({"file_path": file_path,
                   "lang_name": lang_name,
                   "lang_code": lang_code,
                   "dataset": dataset},
                   table, **kwargs)
    return table

def find_lang_column(table):
    r = re.compile(r".*[lL]ang")
    if isinstance(table.columns, pd.MultiIndex): # Check for hierarchical columns
        matches = list(filter(r.match, table.columns.levels[0])) # Check in top level
    else:
        matches = list(filter(r.match, table.columns))
    if matches:
        return matches[0]
    else:
        return None

def add_lang_groups(table, colname="group"):
    table = order_table(table) # In case it's not already in correct order

    # Retrieve language groups in correct order and add them to table in human readable format
    lang_to_group = make_lang_group_dict()
    lang_colname = find_lang_column(table)
    table.insert(loc=0, column=colname, value=table[lang_colname].map(lang_to_group))
    return table

def find_table(r, task="", by="colname"):
    possible_tasks = ["", "pos", "sentiment"]
    possible_by = ["colname", "path"]
    assert task in possible_tasks, "Task must be one of " + str(possible_tasks)
    assert by in possible_by, "'by' must be one of " + str(possible_by)
    all_colnames = pd.read_csv(Path(__file__).parent / "../utils/all_colnames.tsv", sep="\t")

    r = re.compile(r)
    matches = list(filter(r.match, all_colnames.loc[all_colnames["path"].apply(lambda x: task in x), by]))
    if len(matches) == 0:
        raise Exception("No match.")
    if by == "colname":
        paths = all_colnames.loc[all_colnames["path"].apply(lambda x: task in x) & all_colnames["colname"].isin(matches), "path"].values
        if len(paths) == 0:
            raise Exception("No match.")
        print("\nMatched pairs: ", *enumerate(zip(paths, matches)), sep="\n")
        i = int(input("Choose pair: "))
        path = paths[i]
        colname = matches[i]
    else:
        print("\nMatched paths", *enumerate(np.unique(matches)), sep="\n")
        i = int(input("Choose path: "))
        path = matches[i]
        cols = pd.read_excel(path).columns
        print("\nPossible columns", *enumerate(pd.read_excel(path).columns), sep="\n")
        i = int(input("Choose column: "))
        colname = cols[i]
    return path, colname

def get_langs(experiment):
    assert experiment in ["tfm", "acl"], "Only possible experiments are 'tfm' and 'acl'"
    file_path = Path(__file__).parent / "{}_langs.tsv".format(experiment)
    return pd.read_csv(file_path, sep="\t", header=None).values.flatten().tolist()
