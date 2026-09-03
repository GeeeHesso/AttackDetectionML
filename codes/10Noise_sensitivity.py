#!/usr/bin/env python3
"""Compare model performance (F2 score) across input-noise levels.

Reads the results produced by codes/supervised/01GSCV.py and/or
codes/unsupervised/01GSCV.py + codes/unsupervised/03Classification.py for a
baseline (noise-free) run and for one or more runs trained/tested with
Gaussian noise added to the load/gen data (see the ``noise_std`` parameter in
those scripts). Noisy runs are saved under a distinct model key (e.g.
"mlpc_10MW_noise", see ``functions.noisy_model_key``) so they never leak into
the regular model comparison scripts (05Results_unique_all.py,
supervised/02Results_unique.py, unsupervised/02Results_regression.py,
unsupervised/04Results_classification_unique.py) - this script is the
dedicated place to compare across noise levels.
"""

# %% [0] PACKAGES
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
from functions import get_gen_names, noisy_model_key

# MATPLOTLIB PARAMETERS
rcParams["figure.figsize"] = 8, 3
rcParams["figure.dpi"] = 400
plt.style.use("seaborn-dark-palette")

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# %% [1] PARAMETERS
os.chdir("..")  # Working directory is repo main

os.makedirs(pjoin("figures", "noise_sensitivity"), exist_ok=True)

case = "CH"

# "supervised" -> codes/supervised/01GSCV.py results (e.g. model_keys=["mlpc"])
# "unsupervised" -> codes/unsupervised/03Classification.py results (model_keys=["mlpr"])
learning = "supervised"

# Models to compare (keys of load_models()); each must have been trained both
# without noise and with every noise level listed below (set the `noise_std`
# parameter in the corresponding 01GSCV.py / 03Classification.py accordingly).
model_keys = ["mlpc"]

# Noise standard deviations [MW] to compare; 0 stands for the noise-free run.
noise_stds = [0, 5, 10, 20, 50]

ds_type = "injection"
seq = 4
contextual = "hist"


# %% [2] LOAD RESULTS
ds_path = pjoin("datasets", case)

if learning == "supervised":
    res_dir = pjoin("results", "supervised", "single_node_attack", case)
    result_file = "confusion_df.p"
elif learning == "unsupervised":
    res_dir = pjoin("results", "unsupervised", case, "regression")
    result_file = "confusion_test_set.p"
else:
    raise ValueError(f"Unknown learning type: {learning}")

attacked_gens = pd.read_pickle(pjoin(ds_path, "attacked_gens.p"))

f2_all = pd.DataFrame()
missing = []
for model_key, noise_std in product(model_keys, noise_stds):
    save_key = noisy_model_key(model_key, noise_std)

    for attack_gen in attacked_gens:
        res_path = pjoin(
            res_dir,
            save_key,
            ds_type,
            f"{attack_gen}",
            f"sequence_len-{seq}",
            f"contextual_{contextual}",
        )

        f_path = pjoin(res_path, result_file)
        if not os.path.isfile(f_path):
            missing.append((model_key, noise_std, attack_gen))
            continue

        result_df = pd.read_pickle(f_path)
        if learning == "supervised":
            result_df = result_df[result_df.train_size == 1]

        row = pd.DataFrame(
            {
                "model": model_key,
                "noise_std": noise_std,
                "attack_gen": attack_gen,
                "f2_score": result_df["f2_score"].iat[0],
            },
            index=[0],
        )
        f2_all = pd.concat([f2_all, row], ignore_index=True)

if missing:
    print(f"Missing {len(missing)} result(s), e.g.:")
    for model_key, noise_std, attack_gen in missing[:10]:
        print(
            f"  - model={model_key}, noise_std={noise_std}MW, attack_gen={attack_gen}"
        )

f2_all["gen_name"] = f2_all["attack_gen"].map(get_gen_names(case))


# %% [3] F2 SCORE VS NOISE LEVEL - BOX PLOT PER MODEL
for model_key, df_model in f2_all.groupby("model"):
    f2_by_noise = pd.DataFrame(
        {
            noise_std: df_noise.set_index("gen_name")["f2_score"]
            for noise_std, df_noise in df_model.groupby("noise_std")
        }
    )
    f2_by_noise = f2_by_noise[sorted(f2_by_noise.columns)]

    fig, ax = plt.subplots(figsize=(8, 3))
    f2_by_noise.plot.box(ax=ax)
    ax.set(xlabel="Noise std [MW]", ylabel="F₂", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(
        pjoin(
            "figures",
            "noise_sensitivity",
            f"{case}_{learning}_{model_key}_f2_vs_noise_box.pdf",
        ),
        dpi=600,
    )
    plt.close(fig)


# %% [4] F2 SCORE VS NOISE LEVEL - MEDIAN LINE, ALL MODELS
fig, ax = plt.subplots(figsize=(6, 4))
for i, (model_key, df_model) in enumerate(f2_all.groupby("model")):
    median_f2 = df_model.groupby("noise_std")["f2_score"].median()
    ax.plot(
        median_f2.index, median_f2.values, marker="o", color=colors[i], label=model_key
    )

ax.set(xlabel="Noise std [MW]", ylabel="Median F₂", ylim=(0, 1))
ax.legend()
fig.tight_layout()
fig.savefig(
    pjoin("figures", "noise_sensitivity", f"{case}_{learning}_f2_vs_noise_median.pdf"),
    dpi=600,
)
plt.close(fig)
