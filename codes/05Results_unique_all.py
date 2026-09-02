#!/usr/bin/env python3


# %% PACKAGES
import os
import sys
from itertools import product
from os.path import join as pjoin
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parent.parent / "codes")

sys.path.append(os.getcwd())
from functions import load_models, get_gen_names


# MATPLOTLIB PARAMETERS
from pylab import rcParams

rcParams["figure.figsize"] = 8, 3
rcParams["figure.dpi"] = 400
plt.style.use("seaborn-dark-palette")
# plt.style.use('seaborn-v0_8-dark-palette')

# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# colors.append('#FF6347')

color_models = {
    "LSTMR": "#001C7F",
    "MLPR": "#017517",
    "MLPC": "#8C0900",
    "GBC": "#7600A1",
    "RFC": "#B8860B",
    "LSTMC": "#006374",
    "SVC": "#FF6347",
    "KNNC": "#000000",
    "NBC": "#5A6650",
}


""" 
05Results_unique_supervised_unsupervised.py :
    
    Compile all supervised & unsupervised ML results                                    

"""


# %% PATH MANAGER
os.chdir("..")  # Working directory is repo main

nets_dict = ["CH", "DE", "ES"]  # List with all nets

models_dict = load_models()  # Dictionary with all supervised models
models_dict["lstm"] = {}  # unsupervised from google colab

types_dict = ["injection", "generation"]  # List with all dataset type
# types_dict = ['generation']  # List with all dataset type

## Regression dataset types
sequence_lens = [0, 4, 24]  # 4 hours or 24 hours
contextual_lens = ["t", "hist"]  # W/O historical values for contextual variable


# %% LOAD SUPERVISED RESULTS
cartesian = product(
    nets_dict, models_dict.keys(), types_dict, sequence_lens, contextual_lens
)

columns_to_keep = [
    "f2_score",
    "tn",
    "fp",
    "fn",
    "tp",
    "net",
    "node",
    "model",
    "ds_type",
    "sequence",
    "contextual",
    "learning",
]
net_old = ""
results = list()
for net, model, ds_type, seq, contextual in cartesian:
    if net != net_old:  # Avoid doing same things
        net_old = net

        dir_dataset = pjoin("datasets", net)
        path_result = pjoin("results", "supervised", "single_node_attack", net)

        attacked_gens = pd.read_pickle(pjoin(dir_dataset, "attacked_gens.p"))

    # ANOMALY LOAD LOOP
    for i, node in enumerate(attacked_gens):
        dir_result = pjoin(
            path_result,
            f"{model}",
            ds_type,
            f"{node}",
            f"sequence_len-{seq}",
            f"contextual_{contextual}",
        )

        # LOAD RESULTS

        f_path = pjoin(dir_result, "confusion_df.p")
        if not os.path.isfile(f_path):
            continue

        df = pd.read_pickle(f_path)  # Load results
        df = df[df.train_size == 1]  # Keep only for train ratio 1

        df["net"] = net
        df["node"] = node
        df["model"] = model
        df["ds_type"] = ds_type
        df["sequence"] = seq
        df["contextual"] = contextual
        df["learning"] = "supervised"

        if model == "lstm":
            df["model"] = model + "c"  # lstm Classifier
            df["sequence"] = 4
            df["contextual"] = "hist"

        results.append(df[columns_to_keep])


results = pd.concat(results, ignore_index=True)

print("all supervised fits:", results.shape[0], "\n")

# %% LOAD UNSUPERVISED RESULTS
cartesian = product(
    nets_dict, models_dict.keys(), types_dict, sequence_lens, contextual_lens
)

net_old = ""
result_unsup = list()
for net, model, ds_type, seq, contextual in cartesian:
    if net != net_old:  # Avoid doing same things
        net_old = net

        dir_dataset = pjoin("datasets", net)
        path_result = pjoin("results", "unsupervised", net, "regression")

        attacked_gens = pd.read_pickle(pjoin(dir_dataset, "attacked_gens.p"))

    # [2.1] ANOMALY LOAD LOOP
    for i, node in enumerate(attacked_gens):
        dir_result = pjoin(
            path_result,
            f"{model}",
            ds_type,
            f"{node}",
            f"sequence_len-{seq}",
            f"contextual_{contextual}",
        )

        # LSTM NOT 6*4 seq or contectual_t
        # if not os.path.exists(dir_result): continue
        if not os.path.isfile(pjoin(dir_result, "confusion_test_set.p")):
            continue

        # [2.3] LOAD TEST RESULTS
        df = pd.read_pickle(pjoin(dir_result, "confusion_test_set.p"))

        df["net"] = net
        df["node"] = node
        df["model"] = model
        df["ds_type"] = ds_type
        df["sequence"] = seq
        df["contextual"] = contextual
        df["learning"] = "unsupervised"

        if model == "lstm":
            df["model"] = model + "r"  # lstm Regressor

        result_unsup.append(df[columns_to_keep])

result_unsup = pd.concat(result_unsup, ignore_index=True)

results = pd.concat([results, result_unsup], ignore_index=True)

print("all unsupervised fits:", result_unsup.shape[0], "\n")


# %% LOAD OTHER RESULTS
# result_ext_file = 'external_results/best_results_%s.csv' % case
# if os.path.isfile(result_ext_file):
#     result_ext = pd.read_csv(result_ext_file)
#     results = pd.concat([results, result_ext], ignore_index=True)
#
#     print('external results:', result_ext.shape[0], '\n')

# results['fn_rate'] = results['fn']/results['test_hacked']


# EXCLUDING ONE NODE
# results = results[results["node"] != 208]

# %% PROCESS RESULTS

# get gen names
all_gen_names = {}
gen_names_by_country = {}
for net in nets_dict:
    gen_names = get_gen_names(net)
    gen_names_by_country[net] = gen_names
    for gen, name in gen_names.items():
        all_gen_names[gen] = name
# rename nodes
results["name"] = results["node"].map(all_gen_names)

# sort by F2 score
results.sort_values(by="f2_score", ascending=False, inplace=True)

# results = results.round(3)

# rename models
model_names = {
    "lstmr": "LSTMR",
    "mlpr": "MLPR",
    "mlpc": "MLPC",
    "gbc": "GBC",
    "rf": "RFC",
    "lstmc": "LSTMC",
    "svc": "SVC",
    "knn": "KNNC",
    "nb": "NBC",
}
results.model = results.model.replace(model_names)

# select best result for each model and each node
best_results = (
    results.groupby(["model", "node"])
    .first()
    .sort_values(by="f2_score", ascending=False)
    .reset_index()
)

# record f2 score my model and node
f2_scores = pd.DataFrame(
    {
        model: df.set_index("name")["f2_score"]
        for model, df in best_results.groupby("model")
    }
)

# order models
ordered_models = ["LSTMR", "MLPR", "LSTMC", "GBC", "MLPC", "RFC", "SVC", "NBC", "KNNC"]
ordered_models = [m for m in ordered_models if m in f2_scores.columns]
f2_scores = f2_scores[ordered_models]

# order gens
f2_scores.reindex(f2_scores.max(axis=1).sort_values(ascending=False).index)

# %% BAR PLOT OF BEST MODEL BY NODE FOR EACH COUNTRY

colors = [color_models[m] for m in f2_scores.columns]

for net, names in gen_names_by_country.items():
    net_f2_scores = f2_scores.loc[names.values()]

    fig, ax = plt.subplots(figsize=(8, 3))
    net_f2_scores.plot.bar(ax=ax, color=colors)
    ax.set(ylabel="F\u2082", ylim=(0, 1))
    ax.legend(bbox_to_anchor=(1, 1))
    # ax.get_legend().remove()

    # ax.tick_params(axis='x', labelrotation=45)
    plt.xticks(rotation=45, ha="right")
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_y(label.get_position()[1] - (i % 2) * 0.075)
    fig.tight_layout()

    figname = f"{net}_unique_f2-score_all.pdf"
    plt.savefig(pjoin("figures", figname), dpi=600)
    plt.close(fig)


# %% BAR PLOT OF 5 BEST MODEL BY NODE FOR EACH COUNTRY
best_models = ordered_models[:5]

f2_scores_5best = f2_scores[best_models]

for net, names in gen_names_by_country.items():
    net_f2_scores = f2_scores_5best.loc[names.values()]

    fig, ax = plt.subplots(figsize=(8, 3))
    net_f2_scores.plot.bar(ax=ax, color=colors)
    ax.set(ylabel="F\u2082", ylim=(0.85, 1))
    ax.legend(bbox_to_anchor=(1, 1))
    # ax.get_legend().remove()

    # ax.tick_params(axis='x', labelrotation=45)
    plt.xticks(rotation=45, ha="right")
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_y(label.get_position()[1] - (i % 2) * 0.075)
    fig.tight_layout()

    figname = f"{net}_unique_f2-score_5bests.pdf"
    plt.savefig(pjoin("figures", figname), dpi=600)
    plt.close(fig)


# %%% BOX PLOT FOR EACH COUNTRY

classifier_models = ["LSTMC", "GBC", "MLPC", "RFC", "SVC", "NBC", "KNNC"]

f2_scores_classifiers = f2_scores[classifier_models]

for net, names in gen_names_by_country.items():
    net_f2_scores = f2_scores_classifiers.loc[names.values()]

    fig, ax = plt.subplots(figsize=(4, 2.5))  # f2_score comparison by model and node
    net_f2_scores.plot.box(ax=ax)
    ax.set(ylabel="F\u2082", ylim=(0, 1))  # , title=net)
    fig.tight_layout()

    plt.savefig(pjoin("figures", f"{net}_unique_f2-score_boxplot.pdf"), dpi=600)
    plt.close(fig)

# %%% BOX PLOT OF 5 BEST MODEL FOR EACH COUNTRY

for net, names in gen_names_by_country.items():
    net_f2_scores = f2_scores_5best.loc[names.values()]

    fig, ax = plt.subplots(figsize=(4, 2.5))  # f2_score comparison by model and node
    net_f2_scores.plot.box(ax=ax)
    ax.set(ylabel="F\u2082", ylim=(0.85, 1))  # , title=net)
    fig.tight_layout()

    plt.savefig(pjoin("figures", f"{net}_unique_f2-score_5best_boxplot.pdf"), dpi=600)
    plt.close(fig)


# %%% BOX PLOT OF 5 BEST MODEL

fig, ax = plt.subplots(figsize=(3.5, 4))  # f2_score comparison by model and node
f2_scores_5best.plot.box(ax=ax)
ax.set(ylabel="F\u2082", ylim=(0.85, 1))
fig.tight_layout()

plt.savefig(pjoin("figures", f"unique_f2-score_5best_boxplot.pdf"), dpi=600)
plt.close(fig)


# %%% PLOT PRECISION AGAINST RECALL

# compute precision and recall
best_results["precision"] = best_results["tp"] / (
    best_results["tp"] + best_results["fp"]
)
best_results["recall"] = best_results["tp"] / (best_results["tp"] + best_results["fn"])


fig, ax = plt.subplots(figsize=(6, 5))  # f2_score comparison by model and node

# for model in ['GBC', 'MLPC', 'LSTMC']:
# for model in ['MLPR', 'LSTMR']:
for model in ["GBC", "MLPC", "LSTMC", "MLPR", "LSTMR"]:
    model_results = best_results[best_results.model == model]
    model_results.plot.scatter(
        "precision",
        "recall",
        ax=ax,
        label=model,
        c=color_models[model],
        marker="+" if model in ["MLPR", "LSTMR"] else "o",
    )
# ax.set(xlabel='Precision', ylabel='Recall', xlim=(0.87, 1), ylim=(0.9, 1))
ax.set(xlabel="Precision", ylabel="Recall", xlim=(0.85, 1), ylim=(0.85, 1))
ax.legend(loc="lower left")
fig.tight_layout()

# plt.savefig(pjoin('figures', f'precision-recall-classifiers.pdf'), dpi=600)
plt.savefig(pjoin("figures", f"precision-recall.pdf"), dpi=600)
plt.close(fig)

# %%% LIST EXISTING AND MISSING RESULTS

nodes_by_country = {
    net: set(results_by_country.node.unique())
    for net, results_by_country in results.groupby(["net"])
}
nodes = set(results.node.unique())
nodes_count = len(nodes)
models = ["NBC", "KNNC", "SVC", "RFC", "GBC", "MLPC", "LSTMC", "MLPR", "LSTMR"]
contextual_lens = ["t", "hist"]
ds_types = ["generation", "injection"]

for model in models:
    print()
    print(model)
    results_by_model = results[results.model == model]
    for context in contextual_lens:
        if context == "t" and model in ["LSTMC", "LSTMR"]:
            continue
        print(f"\tContext: {context}")
        results_by_context = results_by_model[results_by_model.contextual == context]
        for ds_type in ds_types:
            results_by_ds_type = results_by_context[
                results_by_context.ds_type == ds_type
            ]
            complete = []
            incomplete = {}
            for seq, results_by_seq in results_by_ds_type.groupby("sequence"):
                if set(results_by_seq.node) == nodes:
                    complete.append(seq)
                else:
                    incomplete[seq] = nodes - set(results_by_seq.node)
            print(f"\t\t{ds_type} - sequences", complete)
            for seq, missing in incomplete.items():
                missing_string = "missing  "
                for country, country_nodes in nodes_by_country.items():
                    if country_nodes.issubset(missing):
                        missing_string += f"{country}  "
                    else:
                        country_missing = country_nodes.intersection(missing)
                        if country_missing:
                            missing_string += f"{country} {country_missing}  "
                print(f"\t\t\t[{seq:2}]:", missing_string)
