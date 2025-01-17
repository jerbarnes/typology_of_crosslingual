import pandas as pd
import numpy as np
import glob
import re
import tkinter as tk
from tkinter import filedialog
from tqdm.notebook import tqdm
from pathlib import Path
from . import find_all_colnames

def make_lang_code_dicts():
    """Return a dictionary containing code_to_name (ISO code to language name) and name_to_code dicts."""
    file_path = Path(__file__).parent / "../utils/lang_codes.xlsx"
    lang_codes = pd.read_excel(file_path, header=0)
    return {"code_to_name": pd.Series(lang_codes["English name of Language"].values,
                                      index=lang_codes["ISO 639-1 Code"]).to_dict(),
            "name_to_code": pd.Series(lang_codes["ISO 639-1 Code"].values,
                                      index=lang_codes["English name of Language"]).to_dict()}

def make_lang_group_dict():
    """Return a dictionary that maps language to morphological group."""
    file_path = Path(__file__).parent / "../utils/lang_groups.xlsx"
    return pd.read_excel(file_path).set_index("Language").to_dict()["Group"]

# Make lang-code and lang-group dicts available
code_dicts = make_lang_code_dicts()
code_to_name = code_dicts["code_to_name"]
name_to_code = code_dicts["name_to_code"]

lang_to_group = make_lang_group_dict()

def find_relative_path_to_root():
    """Find path from current working directory to 'typology of crosslingual' dir."""
    parts = Path().cwd().parts
    rel_path = (len(parts) - parts.index("typology_of_crosslingual") - 1) * "../"
    return rel_path

def get_langs(experiment):
    """Return a list of all languages included in the experiment."""
    # Make sure the path is correct even when importing this function from somewhere else
    file_path = Path(__file__).parent / "{}_langs.tsv".format(experiment)
    return pd.read_csv(file_path, sep="\t", header=None).values.flatten().tolist()

def order_table(table, experiment):
    """Order table according to the language order defined in the experiment."""
    all_langs = get_langs(experiment)
    lang_colname = find_lang_column(table)
    lang_order = [lang for lang in all_langs if lang in table[lang_colname].values]
    if isinstance(table.columns, pd.MultiIndex): # Check for hierarchical columns
        level = 0
    else:
        level = None
    new_table = table.copy() # Make a copy so the original does not get modified
    new_table.insert(0, "sort", table[lang_colname].apply(lambda x: lang_order.index(x)))
    new_table = new_table.sort_values(by=["sort"]).drop("sort", axis=1, level=level).reset_index(drop=True)
    return new_table

def convert_table_to_latex(table, experiment, add_color=False, add_group=True):
    """Print table in latex format, also output dataframe."""
    table = order_table(table, experiment) # In case it's not already in correct order

    # Retrieve language groups in correct order and add them to table
    if add_group:
        table = add_lang_groups(table, "group")

    if add_color:
        # Add color to each group
        table.iloc[:, 0] = table.iloc[:, 0].apply(lambda x: r"\{}{{{}}}".format(x.lower(), x))

    # Latex output
    print("\n".join([" & ".join(line) + r"\\" for line in table.astype(str).values]))

    # Pandas output
    return table

def find_lang_column(table):
    """Find the name of the column where languages are kept."""
    r = re.compile(r".*[lL]ang")
    if isinstance(table.columns, pd.MultiIndex): # Check for hierarchical columns
        matches = list(filter(r.match, table.columns.levels[0])) # Check in top level
    else:
        matches = list(filter(r.match, [col for col in table.columns if isinstance(col, str)]))
    if matches:
        return matches[0]
    else:
        return None

def add_lang_groups(table, colname="group"):
    """Add a column containing the morphological group of each language."""
    lang_colname = find_lang_column(table)
    new_table = table.copy() # Make a copy so the original does not get modified
    new_table.insert(loc=0, column=colname, value=table[lang_colname].map(lang_to_group))
    return new_table

def find_table(r, task="", by="colname", update=False):
    """
    Search for a table file's path given a column or file name.

    Parameters:
    r: Can be a regular string or a regex.
    task: Useful when the same table exists for both tasks.
    by: Whether to search by file name ('path') or column name ('colname', default).
    update: If True, update the file that contains all column names.

    Returns:
    Selected path and column name.
    """
    possible_tasks = ["", "pos", "sentiment"]
    possible_by = ["colname", "path"]
    assert task in possible_tasks, "Task must be one of " + str(possible_tasks)
    assert by in possible_by, "'by' must be one of " + str(possible_by)

    if update:
        find_all_colnames.run()

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
        path = find_relative_path_to_root() + paths[i]
        colname = matches[i]
    else:
        matches = np.unique(matches)
        print("\nMatched paths", *enumerate(matches), sep="\n")
        i = int(input("Choose path: "))
        path = find_relative_path_to_root() + matches[i]
        cols = pd.read_excel(path).columns
        print("\nPossible columns", *enumerate(cols), sep="\n")
        i = int(input("Choose column: "))
        colname = cols[i]

    return path, colname

def merge_tables(table1, table2, how, cols_table2=None):
    """
    Merge two pandas DataFrames on the language column.

    'cols_table2' can be used to select which columns from table2 will be merged (eg. ['col_name1', 'col_name2']).
    When set to None, it will use all columns.
    """
    left_lang_col = find_lang_column(table1)
    right_lang_col = find_lang_column(table2)
    if cols_table2 is None:
        cols_table2 = table2.columns
    else:
        cols_table2 = [right_lang_col] + cols_table2

    new_table = pd.merge(table1, table2[cols_table2], how=how, left_on=left_lang_col, right_on=right_lang_col)
    if left_lang_col != right_lang_col:
        new_table = new_table.drop(right_lang_col, axis=1)
    return new_table

def run_through_data(data_path, f, table=None, **kwargs):
    """
    Apply a function to all data. Extra keyword arguments will be passed to f. Provides the function
    with the following dictionary:
        {"file_path": Path to data file,
         "lang_name": Full name of the language,
         "lang_code": ISO code for the language,
         "dataset": Name of the dataset (train, dev or test)}

    Note that 'table' is updated for every file considered, so if you are using this variable, the
    function should always return it, even if it hasn't been modified.

    Parameters:
    data_path: Data directory.
    f: function that will be applied to each data file. Must take an info (dict containing file info)
       and table parameters.
    table: Variable to store info that will be updated by f.

    Returns:
    table variable used to store results from the function
    """
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

def select_dir():
    root = tk.Tk()
    directory = filedialog.askdirectory(initialdir="..") + "/"
    root.withdraw()
    return directory