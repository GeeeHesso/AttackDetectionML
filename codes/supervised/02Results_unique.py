#!/usr/bin/env python3
"""Compile all supervised ML results."""

# %% [0] PACKAGES
import os
import pickle
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

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colors.append("#FF6347")


# %% [1] PATH MANAGER
os.chdir("..")  # Working directory is repo main

os.makedirs(pjoin("figures", "supervised_results_unique"), exist_ok=True)

case = "CH"

nets_dict = [case]  # List with all nets
models_dict = load_models()  # Dictionary with all models
models_dict["lstm"] = {}  # From google colab

types_dict = ["injection", "generation"]  # List with all dataset type

sequence_lens = [0, 4, 24]  # 4 hours or 24 hours
contextual_lens = ["t", "hist"]  # W/O historical values for contextual variable

# %%%  [1.1]  SPECIFIC PARAMETERS
# keys = ['nb', 'knn', 'svc', 'gbc'] # Select only certain models
# models_dict = {key: models_dict[key] for key in keys}
# models_dict.pop('mlpc')

# types_dict = ['injection'] # Select specific ds_type
# types_dict = ['generation'] # Select specific ds_type


# %% [2] LOAD CM
cartesian = product(
    nets_dict, models_dict.keys(), types_dict, sequence_lens, contextual_lens
)

net_key_old = ""
cm_all = pd.DataFrame()

hyper_idx = ["case", "model", "ds_type", "node", "params"]
hyper_all = pd.DataFrame(index=hyper_idx)
learning_idx = [
    "case",
    "model",
    "ds_type",
    "node",
    "learning",
    "iter (epochs)",
    "loss",
    "best_loss",
    "seq",
]
learning_mlpc = pd.DataFrame(index=learning_idx)
for net_key, model_key, ds_type, seq, contextual in cartesian:
    # if contextual == 'hist' and (ds_type == 'injection' or seq != 4):
    #     continue

    if net_key != net_key_old:  # Avoid doing same things
        net_key_old = net_key

        ds_path = pjoin("datasets", net_key)
        res_dir = pjoin("results", "supervised", "single_node_attack", net_key)

        attacked_gens = pd.read_pickle(pjoin(ds_path, "attacked_gens.p"))

    # [2.1] ANOMALY LOAD LOOP
    # attacked_gens = attacked_gens[:1]  ## DEBUG
    # attacked_gens = ['Riddes_gen']  ## DEBUG
    for i, attack_gen in enumerate(attacked_gens):
        res_path = pjoin(
            res_dir,
            f"{model_key}",
            ds_type,
            f"{attack_gen}",
            f"sequence_len-{seq}",
            f"contextual_{contextual}",
        )

        # [2.2] LOAD HYPERPARAMETERS
        if model_key != "lstm":
            f_path = pjoin(res_path, "gscv_best_params.p")
            if os.path.isfile(f_path):
                params = pickle.load(open(f_path, "rb"))

                hyper = pd.DataFrame(
                    [net_key, model_key, ds_type, attack_gen, params], index=hyper_idx
                )
                hyper_all = pd.concat([hyper_all, hyper], axis=1)

        # [2.3] TRAINING PARAMETERS
        if model_key == "mlpc":
            zipped = zip(
                [
                    "gscv",
                ],
                [
                    "gscv_trained.p",
                ],  # 2024 Only gscv models
            )
            for learner, learner_file in zipped:
                f_path = pjoin(res_path, learner_file)
                if not os.path.isfile(f_path):
                    continue
                estimator = pickle.load(open(f_path, "rb"))

                learning = pd.DataFrame(
                    [
                        net_key,
                        model_key,
                        ds_type,
                        attack_gen,
                        learner,
                        estimator.n_iter_,
                        estimator.loss_,
                        estimator.best_loss_,
                        seq,
                    ],
                    index=learning_idx,
                )
                learning_mlpc = pd.concat([learning_mlpc, learning], axis=1)

        # [2.3] LOAD RESULTS

        f_path = pjoin(res_path, "confusion_df.p")
        if not os.path.isfile(f_path):
            continue

        cm = pd.read_pickle(f_path)  # Load results
        # cm = cm[cm.train_size==1]  # Keep only for train ratio 1

        cm["net"] = net_key
        cm["attack_gen"] = attack_gen
        cm["model"] = model_key
        cm["ds_type"] = ds_type
        cm["seq"] = seq
        cm["contextual"] = contextual

        cm_all = cm_all.append(cm, ignore_index=True)


print(cm_all.columns)
print("all fits:", cm_all.shape[0], "\n")
cm_all["fn_rate"] = cm_all["fn"] / cm_all["test_hacked"]
cols = [
    "net",
    "attack_gen",
    "model",
    "ds_type",
    "seq",
    "contextual",
    "fn_rate",
    "f2_score",
    "f5_score",
    "tn",
    "fp",
    "fn",
    "tp",
    "test_hacked",
    "test_occ",
    "train_size",
    "train_sample",
    "train_hacked",
    "train_occ",
]

cm_all = cm_all[cols]
cm_all.sort_values(by="f2_score", ascending=False, inplace=True)
cm_all = cm_all.round(3)

## Rename gens
cm_all.attack_gen = cm_all.attack_gen.replace(get_gen_names(case))

# cm_all_all = cm_all.copy()  # Only train size == 1
# cm_all = cm_all_all[cm_all_all.train_size==1]

learning_mlpc = learning_mlpc.T
learning_mlpc["delta"] = learning_mlpc.loss - learning_mlpc.best_loss

hyper_all = hyper_all.T


# %% [4] MLPC LEARNING ANALYSIS
for est_type, result_type in learning_mlpc.groupby("learning"):
    epochs_by_seq = pd.DataFrame()
    for col, df in result_type.groupby("seq"):
        ser = df["iter (epochs)"].reset_index(drop=True)
        ser.name = col
        print(ser)
        epochs_by_seq = pd.concat([epochs_by_seq, ser], axis=1)

    fig, ax = plt.subplots(figsize=(8, 1.5))
    epochs_by_seq.plot.box(vert=False, ax=ax)
    ax.set(
        xlabel="Number of epochs",
        ylabel="Sequence length",
        # title=est_type,
        xlim=(0, 201),
    )
    fig.tight_layout()
    fig.savefig(
        pjoin(
            "figures",
            "supervised_results_unique",
            f"{case}_mlpc_learning_{est_type}.pdf",
        ),
        dpi=600,
    )
    plt.close(fig)


# %% [5] BEST MODEL BY NODE
cm_net = cm_all[cm_all.net == case]  # Useless for the moment

f2_scores = {key: list() for key in cm_net.model.unique()}
for node in cm_net.attack_gen.unique():
    cm_node = cm_net[cm_net.attack_gen == node]

    # for model in models_dict.keys():
    for model in f2_scores.keys():
        cm_model = cm_node[cm_node.model == model]

        f2_scores[model].append(cm_model.f2_score.iat[0])

f2_scores = pd.DataFrame(f2_scores, index=cm_net.attack_gen.unique())

fig, ax = plt.subplots(figsize=(8, 3.5))  # f2_score comparison by model and node
f2_scores.plot.bar(ax=ax, color=colors)
ax.set(ylabel="f2_score", ylim=(0, 1))
ax.tick_params(axis="x", labelrotation=20)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=len(f2_scores.columns))
# ax.legend(ncol=len(f2_scores.columns))
fig.tight_layout()
fig.savefig(
    pjoin("figures", "supervised_results_unique", f"{case}_best_model_by_node.pdf"),
    dpi=600,
)
plt.close(fig)


# %%% [5.1] LSTM VS MLPC VS GBC
gbc_mlpc = {key: list() for key in ["gbc", "mlpc", "lstm"]}
for node in cm_net.attack_gen.unique():
    cm_node = cm_net[cm_net.attack_gen == node]

    for model in gbc_mlpc.keys():
        cm_model = cm_node[cm_node.model == model]

        gbc_mlpc[model].append(cm_model.f2_score.iat[0])


gbc_mlpc = pd.DataFrame(gbc_mlpc, index=cm_net.attack_gen.unique())

fig, ax = plt.subplots(figsize=(8, 3.5))  # f2_score comparison by model and node
gbc_mlpc.plot.bar(ax=ax)
ax.set(ylabel="f2_score", ylim=(0.9, 1))
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=len(f2_scores.columns))
ax.tick_params(axis="x", labelrotation=20)
fig.tight_layout()
fig.savefig(
    pjoin("figures", "supervised_results_unique", f"{case}_comparison_1.pdf"), dpi=600
)
plt.close(fig)

fig, ax = plt.subplots()  # f2_score comparison by model and node
gbc_mlpc.plot.box(ax=ax, vert=False)
ax.set(xlabel="f2_score", xlim=(0.9, 1))
fig.tight_layout()
fig.savefig(
    pjoin("figures", "supervised_results_unique", f"{case}_comparison_2.pdf"), dpi=600
)
plt.close(fig)


# %%% [5.2] BEST MODEL BY NODE - F5_SCORER

cm_net = cm_all[cm_all.net == case]  # Useless for the moment
cm_net.sort_values(by="f5_score", ascending=False, inplace=True)

f5_scores = {key: list() for key in cm_net.model.unique()}
for node in cm_net.attack_gen.unique():
    cm_node = cm_net[cm_net.attack_gen == node]

    # for model in models_dict.keys():
    for model in f5_scores.keys():
        cm_model = cm_node[cm_node.model == model]

        f5_scores[model].append(cm_model.f5_score.iat[0])

f5_scores = pd.DataFrame(f5_scores, index=cm_net.attack_gen.unique())

fig, ax = plt.subplots(figsize=(8, 3.5))  # f5_score comparison by model and node
f5_scores.plot.bar(ax=ax, color=colors)
ax.set(ylabel="f5_score", ylim=(0, 1))
ax.tick_params(axis="x", labelrotation=20)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=len(f5_scores.columns))
# ax.legend(ncol=len(f5_scores.columns))
fig.tight_layout()
fig.savefig(
    pjoin("figures", "supervised_results_unique", f"{case}_f5_score.pdf"), dpi=600
)
plt.close(fig)


# %%% [5.2] DS_TYPE COMPARISON
cm_net = cm_all[cm_all.net == case]  # Useless for the moment

for model, result_model in cm_net.groupby("model"):
    ds_param = pd.DataFrame()
    for col, df in result_model.groupby("ds_type"):
        # ds_param[col] = df[metric].values
        ser = df["f2_score"]
        ser.name = col
        ds_param = pd.concat([ds_param, ser], axis=1)

    # print(ds_param.shape)
    fig, ax = plt.subplots(figsize=(8, 2.5))  # metric comparison by param
    ds_param.plot.box(ax=ax, vert=False)
    ax.set(title=model)
    ax.set(xlabel="F\u2082")
    # ax.set(xlabel=metric, xlim=(.7, 1))
    fig.tight_layout()
    fig.savefig(
        pjoin(
            "figures",
            "supervised_results_unique",
            f"{case}_{model}_generation-injection.pdf",
        ),
        dpi=600,
    )
    plt.close(fig)


# %%% [5.3] SEQUENCE LENGTH COMPARISON

for model, result_model in cm_net.groupby("model"):
    cm_seq = pd.DataFrame()
    for col, df in result_model.groupby("seq"):
        ser = df["f2_score"]
        ser.name = col
        cm_seq = pd.concat([cm_seq, ser], axis=1)

    fig, ax = plt.subplots(figsize=(8, 2.5))  # metric comparison by param
    cm_seq.plot.box(ax=ax, vert=False)
    ax.set(title=model)
    ax.set(xlabel="F\u2082")
    fig.tight_layout()
    fig.savefig(
        pjoin("figures", "supervised_results_unique", f"{case}_{model}_sequence.pdf"),
        dpi=600,
    )
    plt.close(fig)


# %%% [5.4] CONTEXT COMPARISON

for model, result_model in cm_net.groupby("model"):
    cm_seq = pd.DataFrame()
    for col, df in result_model.groupby("contextual"):
        ser = df["f2_score"]
        ser.name = col
        cm_seq = pd.concat([cm_seq, ser], axis=1)

    fig, ax = plt.subplots(figsize=(8, 2.5))  # metric comparison by param
    cm_seq.plot.box(ax=ax, vert=False)
    ax.set(title=model)
    ax.set(xlabel="F\u2082")
    fig.tight_layout()
    fig.savefig(
        pjoin("figures", "supervised_results_unique", f"{case}_{model}_context.pdf"),
        dpi=600,
    )
    plt.close(fig)


# %% [7] LSTM RESULTS ANALYSIS
cm_lstm = cm_net[cm_net.model == "lstm"]


# %%% [7.1] INJ VS GEN F2_SCORE
f2_score_lstm = pd.DataFrame()
for gtype, df in cm_lstm.groupby("ds_type"):
    df.set_index("attack_gen", inplace=True)
    f2_score_lstm[gtype] = df.f2_score

# Reorder index as the fisrt plot
f2_score_lstm = f2_score_lstm.reindex(f2_scores.index)

## BAR PLOT
fig, ax = plt.subplots()  # f2_score comparison by node and dataset type
f2_score_lstm.plot.bar(ax=ax)
# ax.set(xlabel='', ylabel='f2_score', ylim=(.5,1))
ax.set(xlabel="", ylabel="F\u2082", ylim=(0.9, 1))
ax.tick_params(axis="x", labelrotation=20)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=len(f2_scores.columns))
fig.tight_layout()
fig.savefig(
    pjoin(
        "figures",
        "supervised_results_unique",
        f"{case}_lstm_generation-injection_by_node.pdf",
    ),
    dpi=600,
)
plt.close(fig)

## BOX PLOT
# fig, ax = plt.subplots(figsize=(8,2.4))  # f2_score comparison by model and node
# f2_score_lstm.plot.box(ax=ax, vert=False)
# ax.set(xlabel='f2_score', xlim=(0.9,1))
# ax.set_yticklabels(f2_score_lstm.columns, rotation=90, va='center')
# fig.tight_layout()
# fig.savefig(pjoin('figures', 'supervised_results_unique', f'{case}_lstm_type.pdf'), dpi=600)
# plt.close(fig)


# %%% [7.2] INJ VS GEN FN-FP
# tneg = pd.DataFrame()
fpos = pd.DataFrame()
fneg = pd.DataFrame()
tpos = pd.DataFrame()

for gtype, df in cm_lstm.groupby("ds_type"):
    df.set_index("attack_gen", inplace=True)
    # tneg[gtype] = df.tn
    fpos[gtype] = df.fp
    fneg[gtype] = df.fn
    tpos[gtype] = df.tp

# Reorder index as the fisrt plot
# tneg = tneg.reindex(f2_scores.index)
fpos = fpos.reindex(f2_scores.index)
fneg = fneg.reindex(f2_scores.index)
tpos = tpos.reindex(f2_scores.index)


# %%%% [7.2.1] ABSOLUTE FN-FP
fig, ax = plt.subplots()  # fn comparison by node and dataset type
fneg.plot.bar(ax=ax)
ax.set(ylabel="fn")
ax.tick_params(axis="x", labelrotation=20)
fig.tight_layout()

fig, ax = plt.subplots()  # fp comparison by node and dataset type
fpos.plot.bar(ax=ax)
ax.set(ylabel="fp")
ax.tick_params(axis="x", labelrotation=20)
fig.tight_layout()


# %%%% [7.2.3] RECALL & PRECISION

recall = tpos / (tpos + fneg) * 100
precision = tpos / (tpos + fpos) * 100

fig, ax = plt.subplots()  # recall (sensitivity) by node & type
recall.plot.bar(ax=ax)
ax.set(ylabel="rappel [%]")
ax.set(ylim=(50, 100))
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=len(f2_scores.columns))
ax.tick_params(axis="x", labelrotation=20)
fig.tight_layout()

fig, ax = plt.subplots()  #  precision (positive predictive) by node & type
precision.plot.bar(ax=ax)
ax.set(ylabel="précision [%]")
ax.set(ylim=(70, 100))
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=len(f2_scores.columns))
ax.tick_params(axis="x", labelrotation=20)
fig.tight_layout()
