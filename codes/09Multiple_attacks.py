#!/usr/bin/env python3
"""Performance of the algorithm under multiple concurrent attacks."""

import os
import sys
from itertools import combinations
from os.path import join as pjoin
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import read_pickle as rpckl

# %% PATH/NAME MANAGER
os.chdir(Path(__file__).resolve().parent)


raw_folder = "raw_data"

sys.path.append(os.getcwd())

from functions import load_gen_data

# %% [1] PARAMETERS
os.chdir("..")  # Working directory is repo main

case = "CH"

seq = 24

# %% [2] LOAD DATA

ds_path = pjoin("datasets", case)
results_path = pjoin(
    "results", "unsupervised", case, "regression", "mlpr", "generation"
)

gen_p, _ = load_gen_data(case)


# %% SELECT VALIDATION SET

# selected_index = rpckl(pjoin(ds_path, 'regression_validation_timesteps.p')).to_list()
selected_index = rpckl(pjoin(ds_path, "test_timesteps.p"))[0].to_list()

attacked_gens = rpckl(pjoin(ds_path, "attacked_gens.p"))
attacked_index = rpckl(pjoin(ds_path, "attacked_timesteps.p"))[0].to_list()
attacked_index = [
    selected_index.index(t) for t in attacked_index if t in selected_index
]
regular_index = [t for t in range(len(selected_index)) if t not in attacked_index]


# %% LOOP FOR ATTACKED NODE

max_concurring_attacks = 3

absolute_errors = {i: [] for i in range(max_concurring_attacks + 1)}
relative_errors = {i: [] for i in range(max_concurring_attacks + 1)}
f2 = {i: [] for i in range(max_concurring_attacks + 1)}
scores = pd.DataFrame(columns=["attacked_gen", "tn", "fp", "fn", "tp"])

rated_power = gen_p.max().to_dict()


for attacked_gen in attacked_gens:
    # for attacked_gen in [923]:

    print("Attack on generator:", attacked_gen)

    # load models, threshold and scalers
    model_path = pjoin(
        results_path, str(attacked_gen), f"sequence_len-{seq}", "contextual_t"
    )
    estimator = rpckl(pjoin(model_path, "gscv_trained.p"))
    scaler_x = rpckl(pjoin(model_path, "scaler_x.p"))
    scaler_y = rpckl(pjoin(model_path, "scaler_y.p"))
    confusion = rpckl(pjoin(model_path, "confusion_validation_set.p"))
    threshold = confusion["threshold"].iat[0]

    # data
    # LOAD DATASET
    X_all = gen_p.copy()

    y = X_all[attacked_gen]
    X_context = X_all.drop(columns=attacked_gen)
    context_gens = X_context.columns
    X_context.columns = X_context.columns.astype(str) + "_t"

    X_all = X_all[[attacked_gen]]

    X_hist = pd.DataFrame()
    for i in range(seq):
        X_temp = np.reshape(X_all.values, (20, 8736, -1))
        X_temp = np.roll(X_temp, seq - i, axis=1)
        X_temp = pd.DataFrame(np.reshape(X_temp, (20 * 8736, -1)))
        X_temp.columns = X_all.columns.astype(str) + f"_t-{seq - i}"
        X_hist = pd.concat([X_hist, X_temp], axis=1)

    X = pd.concat([X_context, X_hist], axis=1)

    X = X.loc[selected_index]
    y = y[selected_index].values

    y_max = rated_power[attacked_gen]

    # perform estimation
    X_scaled = scaler_x.transform(X)
    y_scaled = estimator.predict(X_scaled).reshape(-1, 1)
    y_pred = scaler_y.inverse_transform(y_scaled).squeeze()

    # y_attacked = np.where(y > y_max / 2, 0., y_max)
    y_attacked = rpckl(pjoin(ds_path, f"{attacked_gen}_p_attacked.p"))[
        selected_index
    ].to_list()

    fp = (np.abs(y - y_pred)[regular_index] > threshold).sum()
    fn = (np.abs(y_attacked - y_pred)[attacked_index] < threshold).sum()
    tn = len(regular_index) - fp
    tp = len(attacked_index) - fn
    f2[0].append(5 * tp / (5 * tp + 4 * fn + fp))

    scores.loc[len(scores)] = [attacked_gen, tn, fp, fn, tp]

    absolute_errors[0].append(y_pred - y)
    relative_errors[0].append((y_pred - y) / y_max * 100)

    for i in range(1, max_concurring_attacks + 1):
        print("  Concurring attacks:", i)

        y_pred_all = []

        for concurring_attacks in combinations(context_gens, i):
            X_modified = X.copy()
            for gen in concurring_attacks:
                label = f"{gen}_t"
                gen_max = rated_power[gen]
                X_modified[label] = np.where(
                    X_modified[label] > gen_max / 2, 0.0, gen_max
                )
            X_scaled = scaler_x.transform(X_modified)
            y_scaled = estimator.predict(X_scaled).reshape(-1, 1)
            y_pred = scaler_y.inverse_transform(y_scaled).squeeze()
            y_pred_all.append(y_pred)

            fp = (np.abs(y - y_pred)[regular_index] > threshold).sum()
            fn = (np.abs(y_attacked - y_pred)[attacked_index] < threshold).sum()
            tn = len(regular_index) - fp
            tp = len(attacked_index) - fn
            f2[i].append(5 * tp / (5 * tp + 4 * fn + fp))

        y_pred_all = np.array(y_pred_all)
        y_pred_max = y_pred_all.max(axis=0)
        y_pred_min = y_pred_all.min(axis=0)
        y_diff = np.where(y_pred_max - y > y - y_pred_min, y_pred_max, y_pred_min) - y

        absolute_errors[i].append(y_diff)
        relative_errors[i].append(y_diff / y_max * 100)

absolute_errors = {i: np.array(e) for i, e in absolute_errors.items()}
relative_errors = {i: np.array(e) for i, e in relative_errors.items()}

total_score = scores[["tn", "fp", "fn", "tp"]].sum()
print(
    "\nGlobal F2 score:",
    5
    * total_score["tp"]
    / (5 * total_score["tp"] + 4 * total_score["fn"] + total_score["fp"]),
)

# %% F2 SCORE

fig = plt.figure(figsize=(6, 4))
plt.violinplot(
    [f2[i] for i in range(max_concurring_attacks + 1)], vert=False, showmeans=True
)
plt.yticks(
    [i + 1 for i in range(max_concurring_attacks + 1)],
    [i for i in range(max_concurring_attacks + 1)],
)
plt.ylim(0, max_concurring_attacks + 2)
plt.ylabel("Concurring attacks")
plt.xlim(0.85, 1)
plt.xlabel(r"$F_2$")

fig.tight_layout()
fig.savefig(pjoin("figures", "multiple_attacks", "f2.pdf"), dpi=300)
plt.close(fig)

# %% DISTRIBUTION OF ERRORS (ABSOLUTE)

hist_bin_width = 10

hist_range_min = hist_bin_width * np.floor(absolute_errors[0].min() / hist_bin_width)
hist_range_max = hist_bin_width * np.floor(absolute_errors[0].max() / hist_bin_width)

hist_range_min = max(hist_range_min, -hist_range_max)
hist_range_max = min(-hist_range_min, hist_range_max)

hist_range = (hist_range_min, hist_range_max)
hist_bins = int(hist_range_max - hist_range_min) // hist_bin_width

fig = plt.figure(figsize=(6, 4))
plt.hist(
    absolute_errors[0].reshape(-1),
    density=True,
    stacked=True,
    range=hist_range,
    bins=hist_bins,
    color="green",
    alpha=0.3,
    label="None",
)
plt.hist(
    absolute_errors[1].reshape(-1),
    density=True,
    stacked=True,
    range=hist_range,
    bins=hist_bins,
    color="purple",
    label="1",
    histtype="step",
)
if max_concurring_attacks >= 2:
    plt.hist(
        absolute_errors[2].reshape(-1),
        density=True,
        stacked=True,
        range=hist_range,
        bins=hist_bins,
        color="red",
        label="2",
        histtype="step",
    )
if max_concurring_attacks >= 3:
    plt.hist(
        absolute_errors[3].reshape(-1),
        density=True,
        stacked=True,
        range=hist_range,
        bins=hist_bins,
        color="orange",
        label="3",
        histtype="step",
    )
plt.xlabel("Error [MW]")
plt.xlim(hist_range)
plt.ylabel("Probability")
plt.yscale("log")
plt.ylim(1e-6, 1)
plt.legend(title="Concurring attacks", ncol=max_concurring_attacks + 1)

fig.tight_layout()
fig.savefig(pjoin("figures", "multiple_attacks", "absolute_errors.pdf"), dpi=300)
plt.close(fig)


# %% DISTRIBUTION OF ERRORS (RELATIVE)

hist_bin_width = 5

hist_range_min = hist_bin_width * np.floor(relative_errors[0].min() / hist_bin_width)
hist_range_max = hist_bin_width * np.floor(relative_errors[0].max() / hist_bin_width)

hist_range_min = max(hist_range_min, -hist_range_max)
hist_range_max = min(-hist_range_min, hist_range_max)

hist_range = (hist_range_min, hist_range_max)
hist_bins = int(hist_range_max - hist_range_min) // hist_bin_width

fig = plt.figure(figsize=(6, 4))
plt.hist(
    relative_errors[0].reshape(-1),
    density=True,
    stacked=True,
    range=hist_range,
    bins=hist_bins,
    color="green",
    alpha=0.3,
    label="None",
)
plt.hist(
    relative_errors[1].reshape(-1),
    density=True,
    stacked=True,
    range=hist_range,
    bins=hist_bins,
    color="purple",
    label="1",
    histtype="step",
)
if max_concurring_attacks >= 2:
    plt.hist(
        relative_errors[2].reshape(-1),
        density=True,
        stacked=True,
        range=hist_range,
        bins=hist_bins,
        color="red",
        label="2",
        histtype="step",
    )
if max_concurring_attacks >= 3:
    plt.hist(
        relative_errors[3].reshape(-1),
        density=True,
        stacked=True,
        range=hist_range,
        bins=hist_bins,
        color="orange",
        label="3",
        histtype="step",
    )
plt.xlabel("Error relative to rated power [%]")
plt.xlim(hist_range)
plt.ylabel("Probability")
plt.yscale("log")
plt.ylim(2e-5, 1)
plt.legend(title="Concurring attacks", ncol=max_concurring_attacks + 1)

fig.tight_layout()
fig.savefig(pjoin("figures", "multiple_attacks", "relative_errors.pdf"), dpi=300)
plt.close(fig)
