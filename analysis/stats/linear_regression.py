import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import preprocessing

import sys
sys.path.extend(["..", "../.."])
from utils import utils

def load_data(task, short_model_name, experiment):
    path_to_root = utils.find_relative_path_to_root()
    path_to_table = "analysis/stat_tests/{}/tables/full_table_{}_{}.xlsx".format(experiment,
                                                                                 task,
                                                                                 short_model_name)
    full_path = path_to_root + path_to_table
    print("Loading from", full_path)
    df = pd.read_excel(full_path)
    df["Transfer-Type"] = (df["Transfer-Type"] == "Inter").astype(int)
    return df

def preprocess(df, columns_to_drop):
    temp = df.drop(columns_to_drop, axis=1)
    noncat = np.where(temp.apply(lambda x: len(np.unique(x))) != 2)[0] # Will need to improve this
    cat = list(set(np.arange(temp.shape[1])) - set(noncat))
    names = temp.columns.tolist()
    X = preprocessing.scale(temp.iloc[:, noncat], axis=0)
    X = np.hstack((X, temp.iloc[:, cat].values))
    y = df["Transfer-Loss"].values
    X = sm.add_constant(X)
    return X, y, names

def fit_model(X, y, names, output_filepath):
    results = sm.OLS(endog = y, exog = X).fit()
    pvalues = results.pvalues
    results_summary = results.summary(xname=["Constant"] + names)

    # Transform to DataFrame
    summary_table = pd.DataFrame(results_summary.tables[1]).drop(0)
    summary_table.iloc[:, 1:] = summary_table.iloc[:, 1:].applymap(lambda x: float(x.data))
    summary_table.columns = ["var", "coef", "std_err", "t", "p-value", "low", "high"]
    summary_table.to_csv(output_filepath, index=False, sep="\t")
    summary_table = pd.read_csv(output_filepath, sep="\t", dtype={"p-value": float})
    vifs = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    summary_table["vif"] = vifs

    return results_summary, summary_table, pvalues, vifs

def stepwise(task, short_model_name, experiment, p=0.01, check_vifs=False, v=5):
    df = load_data(task, short_model_name, experiment)
    columns_to_drop = ["Train-Language", "Train-Group", "Test-Language",
                       "Test-Group", "Transfer-Loss", "Cross-Score"]
    output = "" # String that will be saved to output file

    while True:
        X, y, names = preprocess(df, columns_to_drop)
        path_to_root = utils.find_relative_path_to_root()
        path_to_tables = path_to_root + "analysis/stat_tests/{}/tables/".format(experiment)
        results_filepath = path_to_tables + "regression_results_{}_{}.tsv".format(task, short_model_name)
        summary_filepath = path_to_tables + "regression_summary_{}_{}.txt".format(task, short_model_name)

        # Fit
        results_summary, summary_table, pvalues, vifs = fit_model(X, y, names, output_filepath=results_filepath)

        # Remove variables with high VIFs or high p-values
        # Ignore transfer type
        vif_cond = np.any(np.array(vifs)[:-1] > v)
        pvalue_cond = np.any(pvalues[:-1] > p)

        if vif_cond and check_vifs:
            idxmax = summary_table.drop(summary_table.shape[0] - 1)["vif"].idxmax()
            excluded = summary_table.at[idxmax, "var"]
            columns_to_drop.append(excluded)
            msg = "Dropping {}, VIF = {}".format(excluded, summary_table.at[idxmax, "vif"])
            output += msg + "\n"
            print(msg)
        elif pvalue_cond:
            idxmax = summary_table.drop(summary_table.shape[0] - 1)["p-value"].idxmax()
            excluded = summary_table.at[idxmax, "var"]
            columns_to_drop.append(excluded)
            msg = "Dropping {}, p-value = {}".format(excluded, summary_table.at[idxmax, "p-value"])
            output += msg + "\n"
            print(msg)
        # End stepwise regression
        else:
            output += "\n" + str(results_summary)
            with open(summary_filepath, "w") as file:
                file.write(output)
            print(results_summary)
            break