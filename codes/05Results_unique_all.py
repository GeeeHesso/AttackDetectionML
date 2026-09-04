#!/usr/bin/env python3
"""Compile all supervised & unsupervised ML results."""

# %% PACKAGES
import os
import sys
from itertools import product
from os.path import join as pjoin
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pylab import rcParams

os.chdir(Path(__file__).resolve().parent)

sys.path.append(os.getcwd())
from functions import get_gen_names, load_models

# MATPLOTLIB PARAMETERS

rcParams["figure.figsize"] = 8, 3
rcParams["figure.dpi"] = 400
plt.style.use("seaborn-dark-palette")

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
results = []
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
result_unsup = []
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
f2_scores_classifiers.to_csv(pjoin("results", "f2_scores_classifiers.csv"), index=False)

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

plt.savefig(pjoin("figures", "unique_f2-score_5best_boxplot.pdf"), dpi=600)
plt.close(fig)


# %%% PLOT PRECISION AGAINST RECALL, WITH MARGINAL BOX PLOTS

# compute precision and recall
best_results["precision"] = best_results["tp"] / (
    best_results["tp"] + best_results["fp"]
)
best_results["recall"] = best_results["tp"] / (best_results["tp"] + best_results["fn"])

plot_models = ["GBC", "MLPC", "LSTMC", "MLPR", "LSTMR"]
xlim = (0.85, 1)
ylim = (0.85, 1)

fig = plt.figure(figsize=(6, 6))
gs = fig.add_gridspec(
    2,
    2,
    width_ratios=(1, 4),
    height_ratios=(4, 1),
    wspace=0.05,
    hspace=0.05,
)
ax = fig.add_subplot(gs[0, 1])
ax_boxy = fig.add_subplot(gs[0, 0], sharey=ax)  # recall distribution, to the left
ax_boxx = fig.add_subplot(gs[1, 1], sharex=ax)  # precision distribution, below

for model in plot_models:
    model_results = best_results[best_results.model == model]
    marker = "+" if model in ["MLPR", "LSTMR"] else "o"
    ax.scatter(
        model_results["precision"],
        model_results["recall"],
        label=model,
        c=color_models[model],
        marker=marker,
    )

positions = range(len(plot_models))

# value axis (recall) is vertical, aligned with ax's y-axis
bp_y = ax_boxy.boxplot(
    [best_results.loc[best_results.model == m, "recall"] for m in plot_models],
    positions=positions,
    vert=True,
    widths=0.6,
    patch_artist=True,
)
# value axis (precision) is horizontal, aligned with ax's x-axis
bp_x = ax_boxx.boxplot(
    [best_results.loc[best_results.model == m, "precision"] for m in plot_models],
    positions=positions,
    vert=False,
    widths=0.6,
    patch_artist=True,
)

for bp in (bp_y, bp_x):
    for patch, model in zip(bp["boxes"], plot_models):
        patch.set_facecolor(color_models[model])
        patch.set_edgecolor(color_models[model])
        patch.set_alpha(0.5)
    for part in ("whiskers", "caps"):
        for line, model in zip(bp[part], [m for m in plot_models for _ in range(2)]):
            line.set_color(color_models[model])
    for line, model in zip(bp["medians"], plot_models):
        line.set_color(color_models[model])
        line.set_linewidth(1.5)
    for flier, model in zip(bp["fliers"], plot_models):
        flier.set_markeredgecolor(color_models[model])
        flier.set_marker("o")
        flier.set_markersize(3)

ax.set(xlim=xlim, ylim=ylim)
ax.tick_params(labelbottom=False, labelleft=False)
ax.legend(loc="lower left")

ax_boxx.set(xlabel="Precision", ylim=(-0.5, len(plot_models) - 0.5))
ax_boxx.tick_params(labelleft=False, left=False)

ax_boxy.set(ylabel="Recall", xlim=(-0.5, len(plot_models) - 0.5))
ax_boxy.tick_params(labelbottom=False, bottom=False)

fig.tight_layout()

plt.savefig(pjoin("figures", "precision-recall.pdf"), dpi=600)
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
