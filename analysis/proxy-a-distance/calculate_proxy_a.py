import argparse # option parsing
from src.dataset import Dataset
from src.model import SVM
import random
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import itertools
from scipy.stats import pearsonr
from sklearn import linear_model

SMALL_SIZE = 14
MEDIUM_SIZE = 16

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : SMALL_SIZE}

plt.rc('font', **font)
#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('lines', linewidth=2)

def plot_heatmap(dataframe, outfile="figures/aggregate_over_graph_setups.pdf", cmap=plt.cm.Blues):
    fig = plt.figure(figsize=[15, 15])
    ax = fig.add_subplot(2, 2, 1)
    #
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    #
    im = ax.imshow(dataframe, cmap=cmap, vmin=0, vmax=2)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(dataframe.columns))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(dataframe.columns, rotation=45)
    ax.set_yticklabels(dataframe.index)
    #
    ax.set_ylabel("Source domain")
    ax.set_xlabel("Target domain")
    #
    data = df.to_numpy()
    #
    fmt = '.2f'
    thresh = 100
    #for i, j in itertools.product(range(data.shape[0]), range(data.shape[1])):
    #    ax.text(j, i, format(data[i, j], fmt),
    #             horizontalalignment="center",
    #             color="white" if data[i, j] > thresh else "black")
    plt.tight_layout()
    plt.show()


def easy_heatmap(dataframe,
                 outfile="figures/aggregate_over_graph_setups.pdf",
                 cmap=plt.cm.Blues):
    c = plt.pcolor(dataframe, cmap=plt.get_cmap(cmap))
    plt.yticks(np.arange(0.5, len(dataframe.index), 1), dataframe.index, )
    plt.xticks(np.arange(0.5, len(dataframe.columns), 1), dataframe.columns, rotation=45, ha='right')
    plt.colorbar(c)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.show()

def proxy_A_distance(domain_source, domain_target, vocab, batch_size):
    data_iterator = Dataset(domain_source,
                            domain_target,
                            vocab,
                            batch_size=batch_size)
    model = SVM(batch_size, data_iterator.get_vocab_size())
    model.fit(data_iterator)
    print('INFO: testing...')
    test_mae = model.test(data_iterator, mae=True)
    print('INFO: test MAE: {}'.format(test_mae))
    pad = 2. * (1. - 2. * test_mae)
    print('INFO: PAD value: {}'.format(2. * (1. - 2. * test_mae)))
    return pad

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_dir", default="../../data/translated_to_en")
    parser.add_argument("--task", default="sentiment")
    parser.add_argument("--langs", nargs="+", default=['de', 'eu', 'es', 'sk', 'ja', 'no', 'tr', 'fi', 'zh', 'he', 'ko', 'el', 'mt', 'zh_yue', 'th', 'id', 'vi', 'ar_dz', 'ar'])

    mapping = {"de": "German", "eu": "Basque", "ru": "Russian", "hr": "Croatian", "es": "Spanish", "ja": "Japanese", "no": "Norwegian", "tr": "Turkish", "fi": "Finnish", "zh": "Chinese", "he": "Hebrew", "ko": "Korean", "en": "English", "el": "Greek", "bg": "Bulgarian", "sk": "Slovak", "mt": "Maltese", "zh_yue": "Cantonese", "th": "Thai", "id": "Indonesian", "vi": "Vietnamese", "ar_dz": "Algerian", "ar": "Arabic"}

    grouping = {"de": "fusional", "eu": "agglutinative",  "es": "fusional", "ja": "agglutinative", "no": "fusional", "tr": "agglutinative", "fi": "agglutinative", "zh": "isolating", "he": "introflexive", "ko": "agglutinative", "en": "fusional", "el": "fusional", "bg": "Bulgarian", "sk": "fusional", "mt": "introflexive", "zh_yue": "isolating", "th": "isolating", "id": "isolating", "vi": "isolating", "ar_dz": "introflexive", "ar": "introflexive"}

    args = parser.parse_args()

    vocab = os.path.join(args.text_dir, args.task, "joint_vocab.txt")

    dists = {}
    for source in args.langs:
        source_file = os.path.join(args.text_dir,
                                   args.task,
                                   source,
                                   "text.txt")
        dists[mapping[source]] = {}
        for target in args.langs:
            target_file = os.path.join(args.text_dir,
                                       args.task,
                                       target,
                                       "text.txt")
            pad = proxy_A_distance(source_file, target_file, vocab, 32)
            dists[mapping[source]][mapping[target]] = pad

    df = pd.DataFrame.from_dict(dists)
    reverse_index = df.index[::-1]
    df = df.reindex(index=reverse_index)

    easy_heatmap(df,
                 outfile="../figures/proxy-a-dist-{}.pdf".format(args.task),
                 cmap=plt.cm.Blues)

    if args.task == "sentiment":
        mbert_results = pd.read_excel("../../results/acl/mbert/results_sentiment.xlsx", index_col=0)
        xlm_results = pd.read_excel("../../results/acl/xlm-roberta/results_sentiment.xlsx", index_col=0)
    else:
        mbert_results = pd.read_excel("../../results/acl/mbert/results_pos.xlsx", index_col=0)
        xlm_results = pd.read_excel("../../results/acl/xlm-roberta/results_pos.xlsx", index_col=0)

    print("Pearson Coefficient of Proxy A-distance and results")
    for model_name, model_results in [("mbert", mbert_results), ("xlm", xlm_results)]:
        apds_by_group = {"fusional": {"fusional": {},
                                      "isolating": {},
                                      "agglutinative": {},
                                      "introflexive": {}
                                      },
                         "isolating": {"fusional": {},
                                       "isolating": {},
                                       "agglutinative": {},
                                       "introflexive": {}
                                       },
                         "agglutinative": {"fusional": {},
                                           "isolating": {},
                                           "agglutinative": {},
                                           "introflexive": {}
                                           },
                         "introflexive": {"fusional": {},
                                          "isolating": {},
                                          "agglutinative": {},
                                          "introflexive": {}
                                          }
                         }
        fusional, isolating, agglutinative, introflexive = [], [], [], []
        fusr, isolr, agglr, intror = [], [], [], []
        target_group = []
        apds = []
        results = []
        for source in args.langs:
            for target in args.langs:
                if source != target:
                    try:
                        apd = df[mapping[source]][mapping[target]]
                        res = model_results[mapping[source]][mapping[target]]
                        apds_by_group[grouping[source]][grouping[target]]["{}-{}".format(source, target)] = (apd, res)
                        apds.append(apd)
                        results.append(res)
                        if grouping[target] == "fusional":
                            fusional.append(apd)
                            fusr.append(res)
                        elif grouping[target] == "isolating":
                            isolating.append(apd)
                            isolr.append(res)
                        elif grouping[target] == "agglutinative":
                            agglutinative.append(apd)
                            agglr.append(res)
                        elif grouping[target] == "introflexive":
                            introflexive.append(apd)
                            intror.append(res)
                    except:
                        pass

        coeff, p_value = pearsonr(apds, results)
        print("{0}: coeff: {1:.3f} p-value: {2:.3f}".format(model_name,
                                                            coeff,
                                                            p_value))

        apds = np.array(apds)
        regr = linear_model.LinearRegression()
        regr.fit(apds.reshape(-1, 1), results)
        pred = regr.predict(apds.reshape(-1, 1))

        plt.scatter(apds, results)
        plt.plot(apds, pred, color='black', linewidth=3)
        plt.xlabel("Proxy A-distance")
        if args.task == "sentiment":
            plt.ylabel("Macro $F_1$")
        else:
            plt.ylabel("Accuracy")
        plt.savefig("../figures/proxy-a-scatter-{0}-{1}.pdf".format(model_name,
                                                                    args.task))
        plt.show()

        plt.scatter(fusional, fusr, color="lightgreen", marker=".", label="fusional")
        plt.scatter(isolating, isolr, color="lightcoral", marker="x", label="isolating")
        plt.scatter(agglutinative, agglr, color="lightblue", marker="*", label="agglutinative")
        plt.scatter(introflexive, intror, color="gold", marker="^", label="introflexive")
        plt.plot(apds, pred, color='black', linewidth=2)
        ax = plt.gca()
        ax.set_facecolor("aliceblue")
        plt.legend(loc='upper center', bbox_to_anchor=(.5, 1.15), ncol=2, fancybox=True, shadow=True)
        plt.xlabel("Proxy A-distance")
        if args.task == "sentiment":
            plt.ylabel("Macro $F_1$")
        else:
            plt.ylabel("Accuracy")

        plt.savefig("../figures/proxy-a-scatter-groups-{0}-{1}.pdf".format(model_name, args.task))
        plt.show()
