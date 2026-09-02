#!/usr/bin/env python3
"""Distribution of anomalies."""

import os
import sys
from os.path import join as pjoin
from pathlib import Path

# MATPLOTLIB PARAMETERS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import read_pickle as rpckl
from pylab import rcParams

rcParams["figure.figsize"] = 8, 5
rcParams["figure.dpi"] = 100

plt.style.use("seaborn-dark-palette")  # PC HEVS
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.rcParams["text.usetex"] = True

attacked_color = "#FF4A1C"
regular_color = "#23967F"

save_extension = "png"


# %% PATH/NAME MANAGER
os.chdir(Path(__file__).resolve().parent.parent / "codes")

raw_folder = "raw_data"

sys.path.append(os.getcwd())

from functions import load_data

# %% [1] PARAMETERS
os.chdir("..")  # Working directory is repo main

os.makedirs(pjoin("figures", "anomalies_distribution"), exist_ok=True)


case = "CH"

# %% [2] LOAD DATA

ds_path = pjoin("datasets", case)

load_p, gen_p, _ = load_data(case)

attacked_gens = rpckl(pjoin(ds_path, "attacked_gens.p"))
attacked_index = rpckl(pjoin(ds_path, "attacked_timesteps.p"))[0].to_list()

regular_index = [t for t in gen_p.index if t not in attacked_index]

# %% LOOP FOR ATTACKED NODE

# for attacked_gen in attacked_gens:
for attacked_gen in [923]:
    X = gen_p

    # anomalies
    X_attacked_gen = rpckl(pjoin(ds_path, f"{attacked_gen}_p_attacked.p"))
    attacked_gen_label = f"{attacked_gen}_attacked"
    X_attacked_gen.name = attacked_gen_label

    # history and context
    seq = 4
    X_hist = pd.DataFrame()
    for t in range(seq):
        X_temp = np.reshape(X.values, (20, 8736, -1))
        X_temp = np.roll(X_temp, seq - t, axis=1)
        X_temp = pd.DataFrame(np.reshape(X_temp, (20 * 8736, -1)))
        X_temp.columns = X.columns.astype(str) + f"_t-{seq - t}"
        X_hist = pd.concat([X_hist, X_temp], axis=1)
    X_context = X.drop(columns=attacked_gen)

    data = X_attacked_gen.to_frame()

    t1_label = f"{attacked_gen}_t-1"
    t2_label = f"{attacked_gen}_t-2"
    data = pd.concat([data, X_hist[t1_label].to_frame()], axis=1)

    # construct other metrics

    sum_label = f"{attacked_gen}_sum"
    data[sum_label] = X_context.sum(axis=1) + X_attacked_gen

    sum_t1_label = f"{attacked_gen}_sum_t-1"
    data[sum_t1_label] = X_hist[[f"{gen}_t-1" for gen in X.columns]].sum(axis=1)

    context_sum_label = f"{attacked_gen}_context_sum"
    data[context_sum_label] = X_context.sum(axis=1)

    # context_t1_sum_label = f'{attacked_gen}_context_sum_t-1'
    # data[context_t1_sum_label] = X_hist[[f'{gen}_t-1' for gen in X_context.columns]].sum(axis=1)

    sum_diff01_label = f"{attacked_gen}_sum_t_minus_t-1"
    data[sum_diff01_label] = data[sum_label] - data[sum_t1_label]

    tdiff01_label = f"{attacked_gen}_t_minus_t-1"
    data[tdiff01_label] = X_attacked_gen - X_hist[t1_label]

    tdiff02_label = f"{attacked_gen}_t_minus_t-2"
    data[tdiff02_label] = X_attacked_gen - X_hist[t2_label]

    tdiff12_label = f"{attacked_gen}_t-1_minus_t-2"
    data[tdiff12_label] = X_hist[t1_label] - X_hist[t2_label]

    # # plot power output at t-1 against t
    # fig, ax = plt.subplots(figsize=(4, 3), dpi=600)
    # data.loc[regular_index, :].plot.scatter(x=attacked_gen_label, y=t1_label, ax=ax, color=regular_color, alpha=0.5)
    # data.loc[attacked_index, :].plot.scatter(x=attacked_gen_label, y=t1_label, ax=ax, color=attacked_color, alpha=0.5)
    # ax.set(xlabel=r'$P_t$  [MW]', ylabel=r'$P_{t-1}$  [MW]')
    # fig.tight_layout()
    # fig.savefig(pjoin('figures', 'anomalies_distribution', f't-1_vs_t_{attacked_gen}.{save_extension}'))

    # # plot sum against power output at time t
    # fig, ax = plt.subplots(figsize=(4, 3), dpi=600)
    # data.loc[regular_index, :].plot.scatter(x=attacked_gen_label, y=sum_label, ax=ax, color=regular_color, alpha=0.5)
    # data.loc[attacked_index, :].plot.scatter(x=attacked_gen_label, y=sum_label, ax=ax, color=attacked_color, alpha=0.5)
    # ax.set(xlabel=r'$P_t$  [MW]', ylabel=r'$P_t^{tot}$  [MW]')
    # fig.tight_layout()
    # fig.savefig(pjoin('figures', 'anomalies_distribution', f'sum_vs_t_{attacked_gen}.{save_extension}'))

    # plot power difference at t against t-1
    fig, ax = plt.subplots(figsize=(4, 3), dpi=600)
    data.loc[regular_index, :].plot.scatter(
        x=t1_label,
        y=tdiff01_label,
        ax=ax,
        color=regular_color,
        marker="+",
        label="regular",
        alpha=0.5,
    )
    data.loc[attacked_index, :].plot.scatter(
        x=t1_label,
        y=tdiff01_label,
        ax=ax,
        color=attacked_color,
        marker="+",
        label="attacked",
        alpha=0.5,
    )
    ax.set(xlabel=r"$P_{t-1}$  [MW]", ylabel=r"$P_t - P_{t-1}$  [MW]")
    ax.legend().remove()
    fig.tight_layout()
    fig.savefig(
        pjoin(
            "figures",
            "anomalies_distribution",
            f"t-minus-t-1_vs_t-1_{attacked_gen}.{save_extension}",
        )
    )

    # plot sum at t against power difference at t
    fig, ax = plt.subplots(figsize=(4, 3), dpi=600)
    data.loc[regular_index, :].plot.scatter(
        x=tdiff01_label,
        y=sum_label,
        ax=ax,
        color=regular_color,
        marker="+",
        label="regular",
        alpha=0.5,
    )
    data.loc[attacked_index, :].plot.scatter(
        x=tdiff01_label,
        y=sum_label,
        ax=ax,
        color=attacked_color,
        marker="+",
        label="attacked",
        alpha=0.5,
    )
    ax.set(xlabel=r"$P_t - P_{t-1}$  [MW]", ylabel=r"$P_t^{tot}$  [MW]")
    ax.legend().remove()
    fig.tight_layout()
    fig.savefig(
        pjoin(
            "figures",
            "anomalies_distribution",
            f"sum_vs_t-minus-t-1_{attacked_gen}.{save_extension}",
        )
    )

    # plot power difference (t, t-1) against power difference (t, t-2)
    fig, ax = plt.subplots(figsize=(4, 3), dpi=600)
    data.loc[regular_index, :].plot.scatter(
        x=tdiff01_label,
        y=tdiff02_label,
        ax=ax,
        color=regular_color,
        marker="+",
        label="regular",
        alpha=0.5,
    )
    data.loc[attacked_index, :].plot.scatter(
        x=tdiff01_label,
        y=tdiff02_label,
        ax=ax,
        color=attacked_color,
        marker="+",
        label="attacked",
        alpha=0.5,
    )
    ax.set(xlabel=r"$P_t - P_{t-1}$  [MW]", ylabel=r"$P_t - P_{t-2}$  [MW]")
    ax.legend().remove()
    fig.tight_layout()
    fig.savefig(
        pjoin(
            "figures",
            "anomalies_distribution",
            f"t-minus-t-2_vs_t-minus-t-1_{attacked_gen}.{save_extension}",
        )
    )

    # # plot power difference (t, t-1) against power difference (t-1, t-2)
    # fig, ax = plt.subplots(figsize=(4, 3), dpi=600)
    # data.loc[regular_index, :].plot.scatter(x=tdiff12_label, y=tdiff01_label, ax=ax,
    #                                         color=regular_color, marker='+', label="regular", alpha=0.5)
    # data.loc[attacked_index, :].plot.scatter(x=tdiff12_label, y=tdiff01_label, ax=ax,
    #                                          color=attacked_color, marker='+', label="attacked", alpha=0.5)
    # ax.set(xlabel=r'$P_{t-1} - P_{t-2}$  [MW]', ylabel=r'$P_t - P_{t-1}$  [MW]')
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig(pjoin('figures', 'anomalies_distribution', f't-minus-t-1_vs_t-1-minus-t-2_{attacked_gen}.{save_extension}'))

    # plot sum difference (t, t-1) against power difference (t, t-1)
    fig, ax = plt.subplots(figsize=(4, 3), dpi=600)
    data.loc[regular_index, :].plot.scatter(
        x=tdiff01_label,
        y=sum_diff01_label,
        ax=ax,
        color=regular_color,
        marker="+",
        label="regular",
        alpha=0.5,
    )
    data.loc[attacked_index, :].plot.scatter(
        x=tdiff01_label,
        y=sum_diff01_label,
        ax=ax,
        color=attacked_color,
        marker="+",
        label="attacked",
        alpha=0.5,
    )
    ax.set(xlabel=r"$P_t - P_{t-1}$  [MW]", ylabel=r"$P_t^{tot} - P_{t-1}^{tot}$  [MW]")
    ax.legend().remove()
    fig.tight_layout()
    fig.savefig(
        pjoin(
            "figures",
            "anomalies_distribution",
            f"sum-diff_vs_t-diff_{attacked_gen}.{save_extension}",
        )
    )

    # plot sum at t against sum at t-1
    fig, ax = plt.subplots(figsize=(4, 3), dpi=600)
    data.loc[regular_index, :].plot.scatter(
        x=sum_t1_label,
        y=sum_label,
        ax=ax,
        color=regular_color,
        marker="+",
        label="regular",
        alpha=0.5,
    )
    data.loc[attacked_index, :].plot.scatter(
        x=sum_t1_label,
        y=sum_label,
        ax=ax,
        color=attacked_color,
        marker="+",
        label="attacked",
        alpha=0.5,
    )
    ax.set(xlabel=r"$ P_{t-1}^{tot}$  [MW]", ylabel=r"$P_t^{tot}$  [MW]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        pjoin(
            "figures",
            "anomalies_distribution",
            f"sum_vs_sum-t-1_{attacked_gen}.{save_extension}",
        )
    )
