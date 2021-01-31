import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append("../..")
from utils import utils, postprocessing_utils as post

def prepare_table(task, short_model_name, metric, experiment, results_path):
    params = {
        "results_dir": results_path,
        "experiment": experiment,
        "short_model_name": short_model_name,
        "task": task,
        "metric": metric
    }

    M = post.Metrics(**params)
    df = M.transfer_loss()
    df["Transfer-Loss"] *= 100
    df_same = df[df["Train-Group"] == df["Test-Group"]]
    df_others = df[df["Train-Group"] != df["Test-Group"]]
    pd.options.mode.chained_assignment = None # Avoid warning here
    df_same.loc[:, "Test-Group"] = "Intra-Group"
    df_others.loc[:, "Test-Group"] = "Inter-Group"
    pd.options.mode.chained_assignment = "warn"
    final = pd.concat([df_same, df_others], ignore_index=True)
    return final

def make_groups(table):
    G1 = table.loc[table["Test-Group"] == "Intra-Group", "Transfer-Loss"].values
    G2 = table.loc[table["Test-Group"] == "Inter-Group", "Transfer-Loss"].values
    return G1, G2

def plot_distribution(table):
    sns.displot(
        x="Transfer-Loss",
        data=table,
        kind="kde",
        hue="Test-Group",
        bw_adjust=0.5,
        palette="crest",
        common_norm=False
    )
    plt.show()
    plt.close()

def one_way(task, short_model_name, metric, experiment, results_path, show_distribution=False, p=0.01, p_norm=0.05):
    table = prepare_table(task, short_model_name, metric, experiment, results_path)
    G1, G2 = make_groups(table)

    if show_distribution:
        plot_distribution(table)

    levene_test = stats.levene(G1, G2)
    levene_pvalue = levene_test.pvalue
    print("Levene (p-value):", levene_pvalue)
    if levene_pvalue < p:
        print("ANOVA condition not met, switching to Kruskal-Wallis")
        print(stats.kruskal(G1, G2))
    else:
        norm_intra = stats.normaltest(G1)
        norm_inter = stats.normaltest(G2)
        print("Normality intra-group (p-value):", norm_intra.pvalue)
        print("Normality inter-group (p-value):", norm_inter.pvalue)
        if norm_intra.pvalue > p_norm or norm_inter.pvalue > p_norm:
            print("ANOVA condition not met, switching to Kruskal-Wallis")
            print(stats.kruskal(G1, G2))
        else:
            print(stats.f_oneway(G1, G2))