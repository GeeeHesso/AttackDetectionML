#!/usr/bin/env python3
"""Generate anomaly dataset."""

# %% PACKAGES

import os
import time
from os.path import join as pjoin
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import rcParams

os.chdir(Path(__file__).resolve().parent)
import sys

sys.path.append(os.getcwd())

from functions import load_data

# MATPLOTLIB PARAMETERS

rcParams["figure.figsize"] = 8, 3
rcParams["figure.dpi"] = 600

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


start_time = time.time()  # For total running time


# %% PATH/NAME MANAGER
os.chdir("..")

save = True
case = "CH"

ds_path = pjoin("datasets", case)
os.makedirs(ds_path, exist_ok=True)

os.makedirs(pjoin("figures", "make_anomalies"), exist_ok=True)

# %% LOAD RAW PROFILES
# load_p, gen_p, gen_info = load_data(case, n_year=4)
load_p, gen_p, gen_info = load_data(case)


# %%% REMOVE VALUES NEAR 0 MW
pthreshold = 0.3

## LOAD
load_zero = load_p < pthreshold
# load_zero.sum().sum()  # very few load values under 0.3MW

# load_p[load_zero] = 0  # Replace value by zeros

criteria = load_zero.sum().to_frame("zero_count")
criteria["z_ratio"] = criteria.zero_count / load_p.shape[0] * 100

fig, ax = plt.subplots()
criteria.z_ratio.plot.hist(ax=ax, bins=50)
ax.set(xlabel=f"loads - proportion of values < {pthreshold} MW", ylabel="# load nodes")
fig.tight_layout()
fig.savefig(pjoin("figures", "make_anomalies", f"{case}_zero_loads.pdf"), dpi=600)
plt.close(fig)


## GEN
gen_zero = gen_p < pthreshold
gen_p[gen_zero] = 0  # Replace value by zeros

criteria = gen_zero.sum().to_frame("zero_count")
criteria["z_ratio"] = criteria.zero_count / gen_p.shape[0] * 100

fig, ax = plt.subplots()
criteria.z_ratio.plot.hist(ax=ax, bins=50)
ax.set(
    xlabel=f"generations - proportion of values < {pthreshold} MW", ylabel="# gen nodes"
)
fig.tight_layout()
fig.savefig(pjoin("figures", "make_anomalies", f"{case}_zero_gens.pdf"), dpi=600)
plt.close(fig)


# %%% RAW PROFILES DESCRIPTION
print("RAW PROFILES")
print(1 * "\t", "load_p shape :", load_p.shape)
print(1 * "\t", "gen_p shape :", gen_p.shape, "\n")
print(2 * "\t", load_p.shape[0], "pas de temps")
print(2 * "\t", load_p.shape[1], "profile de consommation")
print(2 * "\t", gen_p.shape[1], "profile de production")
print()
print(2 * "\t", (load_p < 0).sum().sum(), "valeurs de consommation < 0 MW")
print(2 * "\t", load_zero.sum().sum(), f"valeurs de consommation < {pthreshold} MW")
print(2 * "\t", (gen_p < 0).sum().sum(), "valeurs de production < 0 MW")
print(2 * "\t", gen_zero.sum().sum(), f"valeurs de production < {pthreshold} MW")
print(
    4 * "\t",
    round(gen_zero.sum().sum() / (gen_zero.shape[0] * gen_zero.shape[1]), 2),
    "rapport avec le nombre de valeurs totales",
)

fig, ax = plt.subplots()
load_p.sum(axis=1).plot(ax=ax, label="total consumption", legend=True)
gen_p.sum(axis=1).plot(ax=ax, label="total production", legend=True)
(load_p.sum(axis=1) - gen_p.sum(axis=1)).plot(
    ax=ax, label="import (+) / export (-)", legend=True
)
ax.set(
    ylabel="active power [MW]",
    xlabel="time step [h]",
)
fig.tight_layout()
fig.savefig(pjoin("figures", "make_anomalies", f"{case}_power_balance.pdf"), dpi=600)
plt.close(fig)

# %%% RAW GEN PROFILS DESCRIPTION
gen_p_describe = gen_p.describe()
means = gen_p_describe.loc["mean", :].T
criteria.insert(len(criteria.columns), "p_mean", means)

fig, ax = plt.subplots(figsize=(6, 2))
means.plot.box(vert=False)
ax.set(xlabel="puissance active [MW]")
fig.tight_layout()
fig.savefig(
    pjoin("figures", "make_anomalies", f"{case}_mean_power_output.pdf"), dpi=600
)
plt.close(fig)

# for col in gen_p.columns:  # all profil plot
#     fig, ax = plt.subplots()
#     gen_p[col].plot(ax=ax, label=col, legend=True)
#     ax.set(ylabel='puissance active [MW]', xlabel='pas de temps [h]')
#     fig.tight_layout()


# %%% GEN MAX OPF
criteria["p_max"] = gen_info.pmax.tolist()


# %%% MOST FREQUENT POWER
max_values, max_occ = list(), list()
for col in gen_p:
    values, bins = np.histogram(gen_p[col], bins=122)  # At least bin size of 10 MW
    max_index = np.argmax(values)  # Finding the most frequently power
    max_value = bins[max_index]

    if max_value == 0:
        values = np.delete(values, max_index)  # delete max values
        bins = np.delete(bins, max_index)  # delete bin of max values
        max_index = np.argmax(values)
        max_value = bins[max_index]

    max_values.append(max_value)
    max_occ.append(values.max())


# max_values = pd.Series(max_values, index=gen_p.columns)
# criteria.insert(len(criteria.columns), 'p_nom', max_values)
max_values = pd.DataFrame(
    {
        "p_nom": max_values,
        "p_nom_count": max_occ,
    },
    index=gen_p.columns,
)
criteria = pd.concat([criteria, max_values], axis=1)
criteria["p_nom_ratio"] = criteria.p_nom_count / gen_p.shape[0] * 100
criteria = criteria.round({"z_ratio": 0, "p_mean": 0, "p_nom": 0, "p_nom_ratio": 0})


# %%% PEARSON CORRELATION
data_dict = {"gen": gen_p, "injection": pd.concat([load_p, gen_p], axis=1)}
for ds_type, data in data_dict.items():
    # Stack correlation matrix
    corr = data.corr()
    n1, n2, values = list(), list(), list()
    for col in range(corr.shape[0]):
        for row in range(col + 1, corr.shape[0]):
            # print(row, col)
            n1.append(corr.index[row])
            n2.append(corr.columns[col])
            values.append(corr.iloc[row, col])

    df = pd.DataFrame(
        {
            "n1": n1,
            "n2": n2,
            "pearson": values,
        }
    )

    fig, ax = plt.subplots()  # Histogram of correlation values
    df.pearson.plot.hist(bins=np.linspace(-1, 1, 50), ax=ax)
    ax.set(
        title=f"Histogramme des corrélations par paire de profils de {ds_type} (p={df.shape[0]})",
        xlabel="coefficient de corrélation de Pearson",
        ylabel="récurrence",
    )
    fig.tight_layout()
    fig.savefig(
        pjoin("figures", "make_anomalies", f"{case}_{ds_type}_correlations.pdf"),
        dpi=600,
    )


# %%% PLOT OF ALL PROFILES
cst = []
for col in gen_p.columns:
    cst.append(gen_p[col].round(0).value_counts().to_dict())

cst = pd.DataFrame({"value": cst}, index=gen_p.columns)
cst["len"] = [len(x) for x in cst.value]
cst["most_frequent"] = [item[list(item.keys())[0]] for item in cst.value]
print(cst[cst.most_frequent > 0.99 * gen_p.shape[0]].index)


# %%%% PLOT OF ALL NUCLEAR PROFILES
cols = gen_info.id[gen_info.type == "nuclear"]
fig, ax = plt.subplots(figsize=(8.5, 3))
gen_p[cols].plot(xlabel="time step [h]", ylabel="active power [MW]", legend=True, ax=ax)
ax.set_ylim(bottom=0)
ax.legend(ncol=len(cols))
fig.tight_layout()
fig.savefig(pjoin("figures", "make_anomalies", f"{case}_nuclear.pdf"), dpi=600)
plt.close(fig)

for col in cols:
    print(gen_p[col].round(1).value_counts())


# %% SELECTION OF GEN PROFILS


# %%% 1ST SEL BY HISTOGRAM
for col in gen_info.id:
    ## PLOTTING
    fig, ax = plt.subplots(dpi=200)
    gen_p[col].plot.hist(bins=100, ax=ax)
    ax.set(
        xlabel="puissance active [MW]",
        ylabel="récurrence",
        # ylim=(0,9e3),
    )
    # ax.legend([col[:-4]])
    # ax.legend([gen_info.name[gen_info.id==col].values[0]])
    ax.legend([col])
    fig.tight_layout()
    fig.savefig(
        pjoin("figures", "make_anomalies", f"{case}_gen{col}_distribution.pdf"), dpi=600
    )
    plt.close(fig)

# Profiles with two clearly distinct production methods (two spikes on the histograms)
first_selection = {
    "CH": [923, 918, 933, 934, 173, 932, 924, 931, 915, 927],
    "DE": [
        253,
        235,
        255,  ## HARD COAL
        969,
        234,
        271,
        940,
        301,
        957,  ## GAS
        299,
        300,
        208,
    ],
    "ES": [
        1010,
        997,
        82,
        81,
        905,
        142,
        993,
        152,
        956,  # hydro generators with max output between 100 and 400 MW
        1076,
        998,  # 2 big gas generators
        1004,
        1005,
        1061,  # 3 coal generators
    ],
}


# %%% 2ND SEL BY SEASON PEAKS
for col in first_selection[case]:  # 2ND SELECTION BY VISUAL INSPECTION
    fig, ax = plt.subplots(figsize=(8, 2))
    gen_p[col].plot(ax=ax, legend=True)
    ax.set(ylabel="puissance active [MW]", xlabel="pas de temps [h]")
    fig.tight_layout()
    fig.savefig(
        pjoin("figures", "make_anomalies", f"{case}_gen{col}_peaks.pdf"), dpi=600
    )
    plt.close(fig)

# Profiles with summer daily peaks at the same level as winter peaks
attack_gens = {
    "CH": [923, 918, 933, 934, 173, 932, 924, 931, 915, 927],
    "DE": [
        ## ONLY 271 DO NOT MEET DE 2ND CRITERIA
        253,
        235,
        255,  ## DE HARD COAL
        969,
        234,
        940,
        301,
        957,  ## DE GAS
        299,
        300,
        208,
    ],
    "ES": [
        1010,
        997,
        82,
        81,
        905,
        142,
        993,
        152,
        956,  # hydro generators with max output between 100 and 400 MW
        998,  # 2 big gas generators
        1004,
        1005,  # 3 coal generators
    ],
}

## SAVE ATTACKED GENS
pd.Series(attack_gens[case]).to_pickle(pjoin(ds_path, "attacked_gens.p"))

criteria = criteria.loc[attack_gens[case], :]


# %%% GEN NAMES & NOMINAL POWER
{n: gen_info.name[gen_info.id == n].values[0] for n in attack_gens[case]}
{n: criteria.p_nom.at[n] for n in attack_gens[case]}


# %%% PEARSON CORRELATION BY NODES
for col in attack_gens[case]:
    fig, ax = plt.subplots()
    pair = df[(df.n1 == col) | (df.n2 == col)]

    # Histogram of correlation values
    pair.pearson.plot.hist(bins=np.linspace(-1, 1, 50), ax=ax, label=col, legend=True)
    ax.legend(loc="upper left")
    ax.set(
        title=f"Histogramme des corrélations par paire de profile de {ds_type} "
        f"de la production XX (p={pair.shape[0]})"
    )
    ax.set(ylabel="récurrence", xlabel="coefficient de corrélation de Pearson")
    ax.set(ylim=(0, 130))
    fig.tight_layout()
    fig.savefig(
        pjoin("figures", "make_anomalies", f"{case}_gen{col}_correlation.pdf"), dpi=600
    )
    plt.close(fig)

print(2 * "\t", len(attack_gens[case]), "profil de production respectant les critères")


# %% MAKE ANOMALIES ON-OFF

# RANDOM INDEX FOR ANOMALIES (SAME FOR ALL SINGLE NODE ATTACK)
attack_ratio = 10  # %
attack_timesteps = (
    gen_p.sample(frac=attack_ratio / 100, random_state=1)
    .index.sort_values()
    .to_frame(False)
)
attack_timesteps.to_pickle(pjoin(ds_path, "attacked_timesteps.p"))
attack_timesteps = attack_timesteps[0].to_list()

criteria["attack_count"] = len(attack_timesteps)
criteria["attack_ratio"] = attack_ratio
criteria["off_to_on_count"] = -1

for attack_col in attack_gens[case]:  # Gen number with anomalies
    # print(1 * '\t', f'make single node attack dataset for generation {attack_col}')

    gen_p_a = gen_p[
        attack_col
    ].copy()  # Init df that will containt anomalies  # /!\ Big mistake

    # MAKE ON/OFF ANOMALIES
    off_on_counter = 0
    for ts in attack_timesteps:
        if gen_p_a[ts] > 0:
            gen_p_a[ts] = 0
        else:
            gen_p_a[ts] = criteria["p_nom"][attack_col]
            off_on_counter += 1

    criteria.at[attack_col, "off_to_on_count"] = off_on_counter

    # SAVE DATASETS
    if save:
        gen_p_a.to_pickle(pjoin(ds_path, f"{attack_col}_p_attacked.p"))


criteria["off_to_on_ratio"] = criteria["off_to_on_count"] / len(attack_timesteps) * 100


# %% VISUALISE ANOMALY
for attack_col in attack_gens[case]:  # Gen number with anomalies
    gen_p_a = pd.read_pickle(pjoin(ds_path, f"{attack_col}_p_attacked.p"))

    t0 = 0
    period = 24 * 7
    fig, ax = plt.subplots()
    gen_p_a.plot(ax=ax, label="value sent back to the operator", style="--")
    gen_p[attack_col].plot(ax=ax, label="real value")
    ax.set(
        title=f"Example of a single attack on production {attack_col} (occ: {attack_ratio})",
        ylabel="active power [MW]",
        xlabel="time step [h]",
        # xlim=(0, period),
        xlim=(t0, t0 + period),
        # ylim=(0, 250),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        pjoin(
            "figures", "make_anomalies", f"{case}_gen{attack_col}_attack_example.pdf"
        ),
        dpi=600,
    )
    plt.close(fig)


# %% SET ANALYSIS


# %%% TEST SET ANALYSIS

# RANDOM INDEX FOR TEST (SAME FOR ALL SINGLE NODE ATTACK)
test_timesteps = (
    gen_p.sample(frac=0.2, random_state=3).index.sort_values().to_frame(False)
)
test_timesteps.to_pickle(pjoin(ds_path, "test_timesteps.p"))
test_timesteps = test_timesteps[0].to_list()

test_attack_timestep = list(set(attack_timesteps).intersection(test_timesteps))
criteria["test_attack_count"] = len(test_attack_timestep)
criteria["test_attack_ratio"] = (
    criteria["test_attack_count"] / len(test_timesteps) * 100
)

criteria["test_off_to_on_count"] = -1
for attack_col in attack_gens[case]:  # Gen number with anomalies
    off_on_counter = 0
    for ts in test_attack_timestep:
        if gen_p.at[ts, attack_col] == 0:
            off_on_counter += 1
    criteria.at[attack_col, "test_off_to_on_count"] = off_on_counter

criteria["test_off_to_on_ratio"] = (
    criteria["test_off_to_on_count"] / criteria["test_attack_count"] * 100
)


# %%% TRAIN SET ANALYSIS
train_timesteps = [ts for ts in gen_p.index if ts not in test_timesteps]

train_attack_timestep = list(set(attack_timesteps).intersection(train_timesteps))
criteria["train_attack_count"] = len(train_attack_timestep)
criteria["train_attack_ratio"] = (
    criteria["train_attack_count"] / len(train_timesteps) * 100
)

criteria["train_off_to_on_count"] = -1
for attack_col in attack_gens[case]:  # Gen number with anomalies
    off_on_counter = 0
    for ts in train_attack_timestep:
        if gen_p.at[ts, attack_col] == 0:
            off_on_counter += 1
    criteria.at[attack_col, "train_off_to_on_count"] = off_on_counter

criteria["train_off_to_on_ratio"] = (
    criteria["train_off_to_on_count"] / criteria["train_attack_count"] * 100
)


criteria = criteria.round(
    {
        "z_ratio": 0,
        "p_mean": 0,
        "p_nom": 0,
        "p_nom_ratio": 0,
        "off_to_on_ratio": 0,
        "test_off_to_on_ratio": 0,
        "test_attack_ratio": 0,
        "train_off_to_on_ratio": 0,
        "train_attack_ratio": 0,
    },
)

criteria["delta"] = abs(
    criteria["train_off_to_on_ratio"] - criteria["test_off_to_on_ratio"]
)


# %% RUNNING TIME
ex_time = int(time.time() - start_time)
print(f"\nTotal run time :\t{ex_time} [s]  -  {ex_time / 3600:.2f} [h] ")
