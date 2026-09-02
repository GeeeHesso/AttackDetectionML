#!/usr/bin/env python3
"""Compile all unsupervised ML results after classification."""

# %% PACKAGES
import os
import sys
from itertools import product
from os.path import join as pjoin
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pylab import rcParams

os.chdir(Path(__file__).resolve().parent.parent)

sys.path.append(os.getcwd())
from functions import get_gen_names, load_models

# MATPLOTLIB PARAMETERS

rcParams["figure.figsize"] = 8, 3
rcParams["figure.dpi"] = 400
plt.style.use("seaborn-dark-palette")
# plt.style.use('seaborn-v0_8-dark-palette')

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colors.append("#FF6347")


# %% PARAMETERS
os.chdir("..")  # Working directory is repo main

os.makedirs(pjoin("figures", "unsupervised_results_classification"), exist_ok=True)


case = "CH"

nets = [case]  # List with all nets
types = ["generation", "injection"]  # List with all dataset type
# types = ['generation']  # List with all dataset type

models = load_models()  # Dictionary with all regression models
models = {"mlpr": models["mlpr"]}  # Dictionary with all regression models
models["lstm"] = {}  # From google colab

sequences = [1 * 4, 6 * 4]  # 6 hours or 1 hour
# sequences = [1*4]  # 6 hours or 1 hour
# contextuals = ['t']  # W/O historical values for contectual variable
contextuals = ["t", "hist"]  # W/O historical values for contectual variable

# sorted_nodes = ['Innertkirchen', 'Sils', 'Löbbia', 'Tavanasa', 'Sedrun', 'Pradella',
#                 'Rothenbrunnen', 'Riddes', 'Stalden', 'Cavergno']


# %% LOAD RESULTS DF
cartesian = product(nets, models.keys(), types, sequences, contextuals)

net_old = ""
result_val = pd.DataFrame()
result = pd.DataFrame()
for net, model, ds_type, seq, contextual in cartesian:
    if net != net_old:  # Avoid doing same things
        net_old = net

        dir_dataset = pjoin("datasets", net)
        path_result = pjoin("results", "unsupervised", net, "regression")

        nodes = pd.read_pickle(pjoin(dir_dataset, "attacked_gens.p"))

    # ANOMALY LOAD LOOP
    for i, node in enumerate(nodes):
        dir_result = pjoin(
            path_result,
            model,
            ds_type,
            f"{node}",
            f"sequence_len-{seq}",
            f"contextual_{contextual}",
        )

        # if not os.path.exists(dir_result): continue  # LSTM NOT 6*4 seq or contectual_t
        if not os.path.isfile(pjoin(dir_result, "confusion_validation_set.p")):
            continue  # LSTM NOT 6*4 seq or contectual_t

        # LOAD VALIDATION RESULTS
        df = pd.read_pickle(pjoin(dir_result, "confusion_validation_set.p"))

        df["net"] = net
        df["node"] = node
        df["model"] = model
        df["ds_type"] = ds_type
        df["sequence"] = seq
        df["contextual"] = contextual

        result_val = result_val.append(df, ignore_index=True)

        # LOAD TEST RESULTS
        df = pd.read_pickle(pjoin(dir_result, "confusion_test_set.p"))

        df["net"] = net
        df["node"] = node
        df["model"] = model
        df["ds_type"] = ds_type
        df["sequence"] = seq
        df["contextual"] = contextual

        result = result.append(df, ignore_index=True)


print("all fits:", result_val.shape[0], "\n")

result_val.sort_values(by="f2_score", ascending=False, inplace=True)
result_val = result_val.round(3)

# Rename gen
result_val.node = result_val.node.replace(get_gen_names(case))
result.node = result.node.replace(get_gen_names(case))

type_names = {"injection": "injections", "generation": "productions"}
result.ds_type = result.ds_type.replace(type_names)
result_val.ds_type = result_val.ds_type.replace(type_names)


# %% ANALYSE METRICS BY DS_TYPE
result_net = result[result.net == case]  # Useless for the moment
for model, result_model in result_net.groupby("model"):
    for metric in ["f2_score"]:
        for param in ["ds_type", "sequence", "contextual"]:
            ds_param = pd.DataFrame()
            for col, df in result_model.groupby(param):
                # ds_param[col] = df[metric].values
                ser = df[metric]
                ser.name = col
                ds_param = pd.concat([ds_param, ser], axis=1)

            # print(ds_param.shape)
            fig, ax = plt.subplots(figsize=(8, 2.5))  # metric comparison by param
            ds_param.plot.box(ax=ax, vert=False)
            ax.set(title=model)
            ax.set(xlabel=metric)
            # if metric=='f2_score': ax.set(xlabel=metric, xlim=(.7,1))
            fig.tight_layout()
            fig.savefig(
                pjoin(
                    "figures",
                    "unsupervised_results_classification",
                    f"{case}_{model}_{metric}_{param}.pdf",
                ),
                dpi=600,
            )
            plt.close(fig)


# %% METRICS BY NODE
for model, result_model in result_net.groupby("model"):
    for metric in ["f2_score"]:
        df_node = pd.DataFrame()
        for col, df in result_model.groupby("node"):
            # df_node[col] = df[metric].values
            ser = df[metric]
            ser.name = col
            df_node = pd.concat([df_node, ser], axis=1)

        print(df_node.shape)
        fig, ax = plt.subplots()  # metric comparison by param
        df_node.plot.box(ax=ax, vert=False)
        ax.set(title=model)
        # ax.set(xlabel=metric)
        # if metric=='f2_score': ax.set(xlabel=metric)
        if metric == "f2_score":
            ax.set(xlabel=metric)
        fig.tight_layout()
        fig.savefig(
            pjoin(
                "figures",
                "unsupervised_results_classification",
                f"{case}_{model}_{metric}.pdf",
            ),
            dpi=600,
        )
        plt.close(fig)


# %% DEBUG
# result_mlpr = result[result.model == 'mlpr']
# result_mlpr = result_mlpr[result_mlpr.node == 'Rothenbrunnen']
# result_mlpr = result_mlpr[result_mlpr.sequence == 24]
# result_mlpr = result_mlpr[result_mlpr.contextual == 't']
# result_mlpr = result_mlpr.reset_index(drop=True)
# result_mlpr_unique = result_mlpr[['node', 'model', 'ds_type', 'sequence',
#                                   'contextual', 'f2_score', 'tn', 'fp', 'fn', 'tp']]

# %% LSTM RESULTS ANALYSIS
result_lstm = result[result.model == "lstm"]


# INJ VS GEN F2_SCORE
f2_score_lstm = pd.DataFrame()
for gtype, df in result_lstm.groupby("ds_type"):
    df.set_index("node", inplace=True)
    f2_score_lstm[gtype] = df.f2_score


# Reorder index as the fisrt plot
# f2_score_lstm = f2_score_lstm.reindex(sorted_nodes)

# ## BAR PLOT
# fig, ax = plt.subplots()  # f2_score comparison by node and dataset type
# f2_score_lstm.plot.bar(ax=ax)
# ax.set(xlabel='', ylabel='F\u2082', ylim=(.8,1))
# ax.tick_params(axis='x', labelrotation=20)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
#            ncol=len(f2_score_lstm.columns))
# fig.tight_layout()

## BOX PLOT
fig, ax = plt.subplots(figsize=(8, 2))  # f2_score comparison by model and node
f2_score_lstm.plot.box(ax=ax, vert=False)
ax.set(xlabel="F\u2082", xlim=(0.8, 1))
# ax.set_yticklabels(f2_score_lstm.columns, rotation=90, va='center')
fig.tight_layout()
fig.savefig(
    pjoin(
        "figures",
        "unsupervised_results_classification",
        f"{case}_lstmr_prod-inj_boxplot.pdf",
    ),
    dpi=600,
)
plt.close(fig)


# INJ VS GEN FN-FP
# tneg = pd.DataFrame()
fpos = pd.DataFrame()
fneg = pd.DataFrame()
tpos = pd.DataFrame()

for gtype, df in result_lstm.groupby("ds_type"):
    df.set_index("node", inplace=True)
    # tneg[gtype] = df.tn
    fpos[gtype] = df.fp
    fneg[gtype] = df.fn
    tpos[gtype] = df.tp

# Reorder index as the fisrt plot
# tneg = tneg.reindex(sorted_nodes)
# fpos = fpos.reindex(sorted_nodes)
# fneg = fneg.reindex(sorted_nodes)
# tpos = tpos.reindex(sorted_nodes)


# %%%% ABSOLUTE FN-FP
fig, ax = plt.subplots()  # fn comparison by node and dataset type
fneg.plot.bar(ax=ax)
ax.set(ylabel="fn")
ax.tick_params(axis="x", labelrotation=20)
fig.tight_layout()
fig.savefig(
    pjoin(
        "figures",
        "unsupervised_results_classification",
        f"{case}_lstmr_false_negatives.pdf",
    ),
    dpi=600,
)
plt.close(fig)

fig, ax = plt.subplots()  # fp comparison by node and dataset type
fpos.plot.bar(ax=ax)
ax.set(ylabel="fp")
ax.tick_params(axis="x", labelrotation=20)
fig.tight_layout()
fig.savefig(
    pjoin(
        "figures",
        "unsupervised_results_classification",
        f"{case}_lstmr_false_positives.pdf",
    ),
    dpi=600,
)
plt.close(fig)


# %%%% RECALL
recall = tpos / (tpos + fneg) * 100

fig, ax = plt.subplots(figsize=(8, 2.5))  # recall (sensitivity) by node & type
recall.plot.bar(ax=ax)
ax.set(ylabel="rappel [%]", xlabel="")
ax.set(ylim=(70, 100))
ax.tick_params(axis="x", labelrotation=0)
for i, label in enumerate(ax.get_xticklabels()):
    label.set_y(label.get_position()[1] - (i % 2) * 0.075)
fig.tight_layout()
fig.savefig(
    pjoin("figures", "unsupervised_results_classification", f"{case}_lstmr_recall.pdf"),
    dpi=600,
)
plt.close(fig)


# %%%% PRECISION
precision = tpos / (tpos + fpos) * 100

fig, ax = plt.subplots(
    figsize=(8, 2.5)
)  # precision (positive predictive) by node & type
precision.plot.bar(ax=ax)
ax.set(ylabel="précision [%]", xlabel="")
ax.set(ylim=(70, 100))
ax.tick_params(axis="x", labelrotation=0)
for i, label in enumerate(ax.get_xticklabels()):
    label.set_y(label.get_position()[1] - (i % 2) * 0.075)
fig.tight_layout()
fig.savefig(
    pjoin(
        "figures", "unsupervised_results_classification", f"{case}_lstmr_precision.pdf"
    ),
    dpi=600,
)
plt.close(fig)
