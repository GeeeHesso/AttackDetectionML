#!/usr/bin/env python3

# %% PACKAGES

import os
import time
from os.path import join as pjoin

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import rcParams

rcParams["figure.figsize"] = 8, 3
rcParams["figure.dpi"] = 600

plt.style.use("seaborn-dark-palette")  # PC HEVS

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

""" 
01Explore_raw_data.py :     Explore multiple raw data
                                - Total load by country
                                - Power grid description by country
                                - Correlation
                                - Swiss production structure

"""

start_time = time.time()  # For total running time


# %% PATH/NAME MANAGER
os.chdir(os.path.dirname(os.path.realpath(__file__)))

raw_folder = "raw_data"

# %% RAW DATA

## LOADS
load_p_all = pd.read_csv(pjoin(raw_folder, "loads_2016_1.csv"))
load_p_all.columns = load_p_all.columns.astype(int)
load_p_all *= 100  # per units /!\

load_info_all = pd.read_csv(pjoin(raw_folder, "loads_info.csv"))


## GENS with noise on all costs
gen_p_all = pd.read_csv(pjoin(raw_folder, "gens_2016_1.csv"))
gen_p_all.columns = gen_p_all.columns.astype(int)
gen_p_all *= 100  # per units /!\

gen_info_all = pd.read_csv(pjoin(raw_folder, "gens_info.csv"))


# %%% SERIES
print("RAW PROFILES")
print(1 * "\t", "load_p_all shape :", load_p_all.shape)
print(1 * "\t", "gen_p_all shape :", gen_p_all.shape, "\n")
print(2 * "\t", load_p_all.shape[0], "pas de temps")
print(2 * "\t", load_p_all.shape[1], "profils de consommation")
print(2 * "\t", gen_p_all.shape[1], "profils de production")


# %%% INFO
print("RAW INFOS\n")
print(1 * "\t", "load_info shape :", load_info_all.shape)
for col in load_info_all.columns:
    print(2 * "\t", col)
print("\n")

print(1 * "\t", "gen_info shape :", gen_info_all.shape)
for col in gen_info_all.columns:
    print(2 * "\t", col)
print("\n")

country_all = load_info_all["country"].unique()
print(1 * "\t", country_all.shape[0], "pays différent")
print(2 * "\t", country_all[:10])
print("\n")

# gen_type_all = gen_info['type'].unique()
gen_type_all = gen_info_all["type"].value_counts()
print(1 * "\t", gen_type_all.shape[0], "type de production")
# print(2*'\t', gen_type_all[:3])
for typ, count in gen_type_all[:6].items():
    print(2 * "\t", typ, "\t", count, "profils")
print("\n")


# %%% TOTAL PRODUCTION & CONSUMPTION
total_load_p = load_p_all.sum(axis=1)
total_gen_p = gen_p_all.sum(axis=1)

fig, ax = plt.subplots()
total_load_p.plot(ax=ax, label="total load")
total_gen_p.plot(ax=ax, label="total gen")
ax.set(
    title="Pantagruel",
    xlabel="hour",
    ylabel="active power [MW]",
    xlim=(0, 24 * 7),
)
ax.legend()
fig.tight_layout()
fig.savefig(pjoin("figures", "raw_data", "power_balance.pdf"), dpi=600)


# %% CH, FR & DE
description = {}
for country in ["CH", "DE", "ES"]:
    load_info = load_info_all[load_info_all.country == country]
    gen_info = gen_info_all[gen_info_all.country == country]

    load_p = load_p_all[load_info.id]
    gen_p = gen_p_all[gen_info.id]

    # TOTAL LOAD BY COUNTRY
    total_load_p = load_p.sum(axis=1)
    total_gen_p = gen_p.sum(axis=1)
    total_gap_p = total_load_p - total_gen_p

    total_gen = total_gen_p.sum()

    fig, ax = plt.subplots()
    total_load_p.plot(ax=ax, label="total load")
    total_gen_p.plot(ax=ax, label="total gen")
    total_gap_p.plot(ax=ax, label="gap")
    ax.set(
        title=f"{country}",
        xlabel="hour",
        ylabel="active power [MW]",
        # xlim=(0,24*7),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(pjoin("figures", "raw_data", "%s_power_balance.pdf" % country), dpi=600)

    # PROFILES DESCRIPTION
    gen_types, total_gen_percents, mean_power = [], [], []
    median_power, counts = [], []
    for gen_type, info in gen_info.groupby("type"):
        gen_p_type = gen_p[info.id]

        gen_types.append(gen_type)
        counts.append(info.shape[0])
        total_gen_percents.append(gen_p_type.sum().sum() / total_gen * 100)
        mean_power.append(gen_p_type.mean().mean())
        median_power.append(gen_p_type.mean().median())

    description[country] = (
        pd.DataFrame(
            {
                "type": gen_types,
                "count": counts,
                "annual production %": total_gen_percents,
                "mean power": mean_power,
                "median power": median_power,
            }
        )
        .sort_values("annual production %", ascending=False)
        .round(1)
    )

    print("\nDescription: ", country)
    print(description[country])

# %% PEARSON CORRELATION
for country in ["CH", "DE", "ES"]:
    gen_p = gen_p_all[gen_info_all.id[gen_info_all.country == country]]
    load_p = load_p_all[load_info_all.id[load_info_all.country == country]]

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

        count, division = np.histogram(df.pearson, bins=np.linspace(-1, 1, 50))
        percentages = count / count.sum(axis=0) * 100

        x = []
        for i in range(division.shape[0] - 1):
            x.append((division[i] + division[i + 1]) / 2)

        ax.bar(x, percentages, width=division[1] - division[0])
        ax.set(
            title=f"{country} - Histogramme des corrélations par paire de profils de {ds_type} (p={df.shape[0]})",
            xlabel="coefficient de corrélation de Pearson",
            ylabel="récurrence [%]",
            ylim=(0, 35),
        )
        fig.tight_layout()
        fig.savefig(
            pjoin("figures", "raw_data", "%s_correlations.pdf" % country), dpi=600
        )


# %%% COMPARISON WITH CH
all_list = [
    ["CH", "DE"],
    ["CH", "ES"],
]

for countrys in all_list:
    fig, ax = plt.subplots()  # Histogram of correlation values
    for country in countrys:
        gen_p = gen_p_all[gen_info_all.id[gen_info_all.country == country]]
        load_p = load_p_all[load_info_all.id[load_info_all.country == country]]

        data_dict = {"injection": pd.concat([load_p, gen_p], axis=1)}
        for ds_type, data in data_dict.items():
            # Stack correlation matrix
            corr = data.corr()
            n1, n2, values = [], [], []
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

            count, division = np.histogram(df.pearson, bins=np.linspace(-1, 1, 50))
            percentages = count / count.sum(axis=0) * 100

            x = []
            for i in range(division.shape[0] - 1):
                x.append((division[i] + division[i + 1]) / 2)

            ax.bar(
                x,
                percentages,
                width=division[1] - division[0],
                alpha=0.5,
                # label=country,
                label=f"{country} (p={df.shape[0]})",
            )
    ax.set(
        title=f"histogramme des corrélations par paire de profils de {ds_type}",
        xlabel="coefficient de corrélation de Pearson",
        ylabel="récurrence [%]",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        pjoin(
            "figures", "raw_data", "%s-%s_correlations.pdf" % (countrys[0], countrys[1])
        ),
        dpi=600,
    )


# %%% BY PROD TYPE

countries = ["ES", "CH", "DE"]


for country in countries:
    gen_info_country = gen_info_all[gen_info_all.country == country]

    for gen_type in gen_info_country.type.unique():
        gen_info = gen_info_country[gen_info_country.type == gen_type]

        fig, ax = plt.subplots()  # Histogram of correlation values

        sample = (
            gen_p_all[gen_info.id].sample(min(4, gen_info.shape[0]), axis=1).columns
        )

        datasets = {
            "2016_1": gen_p_all,
        }

        for label, p_all in datasets.items():
            gen_p = p_all[gen_info.id]

            data_dict = {"gen": gen_p}
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

                ## ABSOLUTE
                df.pearson.plot.hist(
                    bins=np.linspace(-1, 1, 50),
                    ax=ax,
                    alpha=0.5,
                    label=f"{label}",
                )

        ax.set(
            title=f"Correlations by pair of {ds_type} profiles - {gen_type} - {country} (p={df.shape[0]})",
            xlabel="Pearson correlation coefficient",
            # ylabel='recurrence [%]',
            ylabel="count",
            # ylim=(0,35),
        )

        ax.legend()
        fig.tight_layout()
        fig.savefig(
            pjoin(
                "figures", "raw_data", "%s_correlations_%s.pdf" % (country, gen_type)
            ),
            dpi=600,
        )


# %% ANNUAL PROD STRUCTURE IN VARIOUS COUNTRIES
countries = [country for country in description.keys()]

prod_by_type = {gen_type: [] for gen_type in gen_info_all["type"].unique()}

for country in countries:
    for gen_type in prod_by_type.keys():
        sum_by_type = 0
        if gen_type in description[country].type.values:
            prod = description[country][description[country].type == gen_type][
                "annual production %"
            ]
            sum_by_type += prod.values[0]
        prod_by_type[gen_type].append(sum_by_type)

structure = pd.DataFrame(prod_by_type, index=countries)


# import matplotlib.ticker as mtick
fig, ax = plt.subplots()
structure.plot.bar(ax=ax, stacked=True)
ax.set(
    title="production structure",
    ylabel="annual production [%]",
    ylim=(0, 1e2),
)
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

# Même ordre de couleur dans la légende et le graphe (foncé vers clair)
handles, labels = ax.get_legend_handles_labels()
order = [len(labels) - x - 1 for x in range(len(labels))]
ax.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    bbox_to_anchor=(1, 0, 0, 1),
)
ax.tick_params(axis="x", labelrotation=0)
fig.tight_layout()
fig.savefig(pjoin("figures", "raw_data", "production_structure.pdf"), dpi=600)


# %% MONTHLY PROD STRUCTURE IN CH
country = "CH"
load_info = load_info_all[load_info_all.country == country]
gen_info = gen_info_all[gen_info_all.country == country]

load_p = load_p_all[load_info.id]
gen_p = gen_p_all[gen_info.id]

load_p.index = pd.date_range(start="1/1/2018", periods=8736, freq="h")
gen_p.index = load_p.index

months, loads = [], []
prod_by_type = {gen_type: [] for gen_type in gen_info.type.unique()}
for month, monthly_data in gen_p.groupby(gen_p.index.month):
    months.append(month)

    ## TOTAL LOAD
    total_load = load_p.loc[monthly_data.index, :].sum().sum() / 1000
    loads.append(total_load)  # GWh

    ## TOTAL GEN BY TYPE
    for gen_type, info in gen_info.groupby("type"):
        gen_p_type = monthly_data[info.id]

        prod_by_type[gen_type].append(gen_p_type.sum().sum() / 1000)  # GWh


monthly_structure = pd.DataFrame(
    prod_by_type,
    index=months,
)

monthly_structure = monthly_structure[["nuclear", "hydro_ror", "hydro_storage"]]

c = [
    "#FED304",
    "#6B97B5",
    "#60AEC0",
]

fig, ax = plt.subplots()
ax.plot([m - 1 for m in months], loads, label="consumption", color="#000000")
monthly_structure.plot.bar(ax=ax, stacked=True, color=c)
ax.set(
    title="production structure",
    ylabel="energy [GWh]",
    xlabel="month",
)
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax.tick_params(axis="x", labelrotation=0)

fig.tight_layout()
fig.savefig(pjoin("figures", "raw_data", "CH_production_structure.pdf"), dpi=600)


# %% RUNNING TIME
ex_time = int(time.time() - start_time)
print(f"\nTotal run time :\t{ex_time} [s]  -  {ex_time / 3600:.2f} [h] ")
