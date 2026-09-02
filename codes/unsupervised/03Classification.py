#!/usr/bin/env python3
"""Classification of unsupervised results with threshold optimization."""

# %% PACKAGES
import os
import sys
from itertools import product
from os.path import join as pjoin
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import read_pickle as rpckl
from pylab import rcParams
from sklearn.metrics import confusion_matrix, fbeta_score
from sklearn.metrics import mean_absolute_percentage_error as mape

os.chdir(Path(__file__).resolve().parent.parent)

sys.path.append(os.getcwd())
from functions import get_gen_names, get_p_nom, load_models

# MATPLOTLIB PARAMETERS

rcParams["figure.figsize"] = 8, 3
rcParams["figure.dpi"] = 100
plt.style.use("seaborn-dark-palette")

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


start_time = time()  # For total running time


# %% PARAMETERS
os.chdir("..")  # Working directory is repo main

os.makedirs(pjoin("figures", "unsupervised_classification"), exist_ok=True)

case = "CH"

make_figures = True

nets_dict = [case]  # List with all nets
types_dict = ["generation", "injection"]  # List with all dataset type

models_dict = load_models()  # Dictionary with all regression models
models_dict = {"mlpr": models_dict["mlpr"]}  # Dictionary with all regression models
# models_dict = {}  # Dictionary with all regression models
models_dict["lstm"] = {}  # From google colab

sequence_lens = [4, 24]  # 6 hours or 1 hour
contextuals_lens = ["t", "hist"]  # W/O historical values for contectual variable

p_nom = get_p_nom(case)

## NOT SORTED YET
sorted_nodes = list(get_p_nom(case).keys())


# %% DEBUG PARAMETERS
# keys = ['lstm'] # Select specific models
# keys = ['mlpr'] # Select specific models
# models_dict = {key: models_dict[key] for key in keys}

# types_dict = ['generation'] # Select specific ds_type
# types_dict = ['injection'] # Select specific ds_type

# sequence_lens = [1*4] # Select specific sequence length
# sequence_lens = [6*4] # Select specific sequence length

# contextuals_lens = ['t'] # Select specific contextual sequence length
# contextuals_lens = ['hist'] # Select specific contextual sequence length


# %% NET/MODEL/DATASET LOOP
cartesian = product(
    nets_dict, models_dict.keys(), types_dict, sequence_lens, contextuals_lens
)

low_power_lstm = {}
low_power_lstm["val"] = {key: {} for key in types_dict}
low_power_lstm["test"] = {key: {} for key in types_dict}

mape_lstm = {}
mape_lstm["val"] = {key: {} for key in types_dict}
mape_lstm["test"] = {key: {} for key in types_dict}

std_lstm = {key: {key2: {} for key2 in sorted_nodes} for key in types_dict}

net_key_old = ""
c = 0  # Fit counter
cartesian_length = (
    len(nets_dict)
    * len(models_dict.keys())
    * len(types_dict)
    * len(sequence_lens)
    * len(contextuals_lens)
)
for net_key, model_key, ds_type, seq, contextual in cartesian:
    c += 1

    if net_key != net_key_old:  # Avoid doing same things
        net_key_old = net_key

        dir_dataset = pjoin("datasets", net_key)
        path_result = pjoin("results", "unsupervised", net_key, "regression")

        attacked_gens = rpckl(pjoin(dir_dataset, "attacked_gens.p"))
        attacked_index = rpckl(pjoin(dir_dataset, "attacked_timesteps.p"))[0].to_list()

        # load_p, gen_p = load_data(case)

        # test_index is the index for testing the classification
        classification_test_index = rpckl(pjoin(dir_dataset, "test_timesteps.p"))[
            0
        ].to_list()
        # regression_index = [ts for ts in gen_p.index if ts not in classification_test_index]
        regression_validation_index = rpckl(
            pjoin(dir_dataset, "regression_validation_timesteps.p")
        ).to_list()

    # LOOP FOR ATTACKED NODE
    # attacked_gens = attacked_gens[:1]  ## DEBUG
    # attacked_gens = [300]  ## DEBUG
    for attacked_gen in attacked_gens:
        # LOAD ATTACKED
        attacked_gen_p = rpckl(pjoin(dir_dataset, f"{attacked_gen}_p_attacked.p"))

        # LOAD PREDICTION
        dir_res = pjoin(
            path_result,
            f"{model_key}",
            ds_type,
            f"{attacked_gen}",
            f"sequence_len-{seq}",
            f"contextual_{contextual}",
        )

        if not os.path.exists(dir_res):
            continue  # LSTM NOT 6*4 seq or contectual_t

        y_val = rpckl(pjoin(dir_res, "prediction_validation_set.p")).sort_index()
        y_test = rpckl(pjoin(dir_res, "prediction_test_set.p")).sort_index()

        y_val.index = regression_validation_index
        y_test.index = classification_test_index

        # Hacked time series
        y_val["y_hacked"] = attacked_gen_p.loc[y_val.index]
        y_test["y_hacked"] = attacked_gen_p.loc[y_test.index]

        # Index of hacked timestep
        val_hacked_index = [idx for idx in attacked_index if idx in y_val.index]
        test_hacked_index = [idx for idx in attacked_index if idx in y_test.index]

        # Real class
        y_val["label"] = False
        y_val.loc[val_hacked_index, "label"] = True

        y_test["label"] = False
        y_test.loc[test_hacked_index, "label"] = True

        # MAPE
        low_power = 0.2 * p_nom[attacked_gen]

        # Valisation
        y_val_low = y_val[y_val.y_val > low_power]
        mape_lstm["val"][ds_type][attacked_gen] = mape(
            y_val_low.y_val, y_val_low.y_val_predict
        )

        # Test
        y_test_low = y_test[y_test.y_test > low_power]
        mape_lstm["test"][ds_type][attacked_gen] = mape(
            y_test_low.y_test, y_test_low.y_test_predict
        )

        # ERROR
        ## VALIDATION SET
        y_val["error"] = y_val["y_val"] - y_val["y_val_predict"]
        y_val["error_hacked"] = y_val["y_hacked"] - y_val["y_val_predict"]

        error_val = y_val["error"][y_val["label"] == False]
        error_val_hacked = y_val["error_hacked"][y_val["label"] == True]
        error_val = pd.concat([error_val, error_val_hacked], axis=1)

        ## TEST SET
        y_test["error"] = y_test["y_test"] - y_test["y_test_predict"]
        y_test["error_hacked"] = y_test["y_hacked"] - y_test["y_test_predict"]

        error_test = y_test["error"][y_test["label"] == False]
        error_test_hacked = y_test["error_hacked"][y_test["label"] == True]
        error_test = pd.concat([error_test, error_test_hacked], axis=1)

        if make_figures:
            # ERROR DISTRIBUTION IN VALIDATION SET
            fig, ax1 = plt.subplots(figsize=(8, 2.5))
            # ax1.set(title=f'{model_key} - {attacked_gen} - {ds_type} - len-{seq} - contextual_{contextual}')
            ax1.hist(error_val.error, bins=100, alpha=0.6)
            ax1.set(
                ylabel="# d'erreurs sans attaques",
                # ylabel='récurrence sans anomalie',
                # ylim=(0, 600),
                xlabel="erreur [MW]",
                # xlim=(-150,150),
            )
            ax1.yaxis.label.set_color(color=colors[0])
            ax1.tick_params(axis="y", colors=colors[0])

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.hist(error_val.error_hacked, bins=100, alpha=0.6, color=colors[1])
            ax2.set(
                ylabel="# d'erreurs avec attaques",
                # ylabel='récurrence avec anomalie',
                # ylim=(0, 60),
            )
            ax2.yaxis.label.set_color(color=colors[1])
            ax2.tick_params(axis="y", colors=colors[1])

            # print(f'{model_key} - {attacked_gen} - {ds_type} - len-{seq} - contextual_{contextual}')
            # print(f'écart type sans attaques {error_val.error.std():.2f}')

            fig.tight_layout()
            fig.savefig(
                pjoin(
                    "figures",
                    "unsupervised_classification",
                    f"{net_key}_{model_key}_{ds_type}_len{seq}_{contextual}_{attacked_gen}_error.pdf",
                ),
                dpi=600,
            )
            plt.close(fig)

        # ABSOLUTE ERROR

        ## VALIDATION SET
        y_val["absolute_error"] = y_val["error"].abs()
        y_val["absolute_error_hacked"] = y_val["error_hacked"].abs()

        # error_abs_val = y_val['absolute_error'][y_val['label']==False]
        # error_abs_val_hacked = y_val['absolute_error_hacked'][y_val['label']==True]
        # error_abs_val = pd.concat([error_abs_val, error_abs_val_hacked], axis=1)

        ## VALIDATION SET
        y_test["absolute_error"] = y_test["error"].abs()
        y_test["absolute_error_hacked"] = y_test["error_hacked"].abs()

        # error_abs_test = y_test['absolute_error'][y_test['label']==False]
        # error_abs_test_hacked = y_test['absolute_error_hacked'][y_test['label']==True]
        # error_abs_test = pd.concat([error_abs_test, error_abs_test_hacked], axis=1)

        # ABSOLUTE ERROR DISTRIBUTION
        # fig, ax1 = plt.subplots()
        # ax1.hist(error_abs.absolute_error, bins=100, alpha=.6)
        # ax1.set(ylabel='récurrence sans anomalie', ylim=(0, 800),
        #         xlabel='erreur absolue', xlim=(0,150))
        # ax1.yaxis.label.set_color(color=colors[0])
        # ax1.tick_params(axis='y', colors=colors[0])

        # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        # ax2.hist(error_abs.absolute_error_hacked, bins=100, alpha=.6, color=colors[1])
        # ax2.set(ylabel='récurrence avec anomalie', ylim=(0, 80))
        # ax2.yaxis.label.set_color(color=colors[1])
        # ax2.tick_params(axis='y', colors=colors[1])

        # OPTIMISE THRESHOLD WITH VALIDATION SET
        threshold_max = int(y_val["absolute_error"].max())

        cm, f2_score = list(), list()
        for threshold in range(1, threshold_max):
            y_class_predict = y_val["absolute_error_hacked"] > threshold

            cm.append(confusion_matrix(y_val["label"], y_class_predict).ravel())
            f2_score.append(fbeta_score(y_val["label"], y_class_predict, beta=2))

        cm = np.array(cm)
        threshold_fit = pd.DataFrame(
            {
                "threshold": range(1, threshold_max),
                "f2_score": f2_score,
                "tn": cm[:, 0],
                "fp": cm[:, 1],
                "fn": cm[:, 2],
                "tp": cm[:, 3],
            }
        ).sort_values("f2_score", ascending=False)

        threshold_fit.iloc[0:1, :].to_pickle(
            pjoin(dir_res, "confusion_validation_set.p")
        )

        best_threshold = threshold_fit.threshold.iat[0]

        y_val["classification"] = y_val["absolute_error_hacked"] > best_threshold
        y_val["mistakes"] = y_val["label"] ^ y_val["classification"]
        y_val[["label", "classification", "mistakes"]].to_pickle(
            pjoin(dir_res, "mistakes_validation_set.p")
        )

        y_test["classification"] = y_test["absolute_error_hacked"] > best_threshold
        y_test["mistakes"] = y_test["label"] ^ y_test["classification"]
        y_test[["label", "classification", "mistakes"]].to_pickle(
            pjoin(dir_res, "mistakes_test_set.p")
        )

        # PLOT EFFECT OF THRESHOLD
        if make_figures:
            fig, ax = plt.subplots(figsize=(8, 2.5))
            # ax.plot(threshold_fit.threshold, threshold_fit.f2_score)
            ax.scatter(threshold_fit.threshold, threshold_fit.f2_score)
            ax.set(
                ylabel="F\u2082",
                ylim=(0, 1),
                title=get_gen_names(case)[attacked_gen],
                xlabel="Seuil [MW]",
                xlim=(0, threshold_max),
            )

            fig.tight_layout()
            fig.savefig(
                pjoin(
                    "figures",
                    "unsupervised_classification",
                    f"{net_key}_{model_key}_{ds_type}_len{seq}_{contextual}_{attacked_gen}_threshold.pdf",
                ),
                dpi=600,
            )
            plt.close(fig)

        # STD & THRESHOLD
        std_lstm[ds_type][attacked_gen]["std"] = error_val.error.std()
        std_lstm[ds_type][attacked_gen]["threshold"] = best_threshold

        # LOW POWER COUNTS
        # Valisation
        y_val_hacked = y_val[y_val.label == True]
        y_val_hacked = y_val_hacked[
            y_val_hacked.y_hacked == 0
        ]  # Attaque de mise à l'arrêt

        low_power = 0.2 * p_nom[attacked_gen]
        low_power_lstm["val"][ds_type][attacked_gen] = (
            y_val_hacked.y_val < low_power
        ).sum()

        # Test
        y_test_hacked = y_test[y_test.label == True]
        y_test_hacked = y_test_hacked[
            y_test_hacked.y_hacked == 0
        ]  # Attaque de mise à l'arrêt

        # low_power = 0.2 * p_nom[attacked_gen]
        low_power_lstm["test"][ds_type][attacked_gen] = (
            y_test_hacked.y_test < low_power
        ).sum()

        # OPTIMISE  2 THRESHOLD
        # postive_idx = y_val.index[y_val['error_hacked'] >=0 ]
        # negative_idx = y_val.index[y_val['error_hacked'] < 0 ]

        # threshold_pos, threshold_neg = list(), list()
        # cm, f2_score = list(), list()
        # for pos in range(1, 101):
        #     for neg in range(1, 101):
        #         y_class_predict_pos = y_val.loc[postive_idx,'error_hacked'] > pos
        #         y_class_predict_neg = y_val.loc[negative_idx,'error_hacked'] < -neg

        #         y_class_predict = pd.concat([y_class_predict_pos, y_class_predict_neg])
        #         y_class_predict = y_class_predict.sort_index()

        #         threshold_pos.append(pos)
        #         threshold_neg.append(neg)
        #         cm.append(confusion_matrix(y_val['hacked'], y_class_predict).ravel())
        #         f2_score.append(fbeta_score(y_val['hacked'], y_class_predict,  beta=2))

        # cm = np.array(cm)
        # threshold_2_fit = pd.DataFrame({
        #     'threshold_pos': threshold_pos,
        #     'threshold_neg': threshold_neg,
        #     'f2_score': f2_score,
        #     'tn': cm[:, 0], 'fp': cm[:, 1],
        #     'fn': cm[:, 2], 'tp': cm[:, 3],
        #     })

        # RESULTS IN TEST SET
        y_class_predict = y_test["absolute_error_hacked"] > best_threshold

        f2_score = fbeta_score(y_test["label"], y_class_predict, beta=2)
        cm = confusion_matrix(y_test["label"], y_class_predict).ravel()

        # SAVE RESULTS
        pd.DataFrame(
            {
                "best_threshold": best_threshold,
                "f2_score": f2_score,
                "tn": cm[0],
                "fp": cm[1],
                "fn": cm[2],
                "tp": cm[3],
            },
            index=[0],
        ).to_pickle(pjoin(dir_res, "confusion_test_set.p"))


# %% STD ANALYSIS
std_lstm_inj = pd.DataFrame(std_lstm["injection"]).T
std_lstm_gen = pd.DataFrame(std_lstm["generation"]).T

std_lstm_inj["threshold/std"] = std_lstm_inj["threshold"] / std_lstm_inj["std"]
std_lstm_gen["threshold/std"] = std_lstm_gen["threshold"] / std_lstm_gen["std"]

gen_index = {gen: i for i, gen in enumerate(std_lstm_gen.index)}
std_lstm_gen["gen_index"] = [gen_index[gen] for gen in std_lstm_gen.index]
std_lstm_inj["gen_index"] = [gen_index[gen] for gen in std_lstm_inj.index]


fig, ax = plt.subplots(figsize=(8, 3.5))
# fig, ax = plt.subplots()
# ax.plot(std_lstm_inj.index, std_lstm_inj['threshold/std'])
ax.scatter(std_lstm_inj["gen_index"], std_lstm_inj["threshold/std"], label="Injections")
ax.scatter(
    std_lstm_gen["gen_index"], std_lstm_gen["threshold/std"], label="Productions"
)
mean_all = (
    std_lstm_inj["threshold/std"].mean() + std_lstm_gen["threshold/std"].mean()
) / 2
ax.axhline(mean_all, ls="--", c="k", label=f"Moyenne: {mean_all:.1f}")
ax.set(
    ylabel="Seuil / écart type",
    ylim=(0, 2 * mean_all),
)
ax.legend(ncol=3)

gen_names = {i: get_gen_names(case)[gen] for gen, i in gen_index.items()}
ax.set_xticks(list(range(len(gen_index))))
ax.set_xticklabels([gen_names[i] for i in range(len(gen_index))])
plt.xticks(rotation=45, ha="right")

# labels = [l.split('_')[0] for l in std_lstm_inj.index]
# labels = ax.set_xticklabels(labels)
# labels = ax.set_xticklabels(std_lstm_inj.index)
# for i, label in enumerate(labels):
#     label.set_y(label.get_position()[1] - (i % 2) * 0.075)

fig.tight_layout()

figname = f"{case}_threshold_std.pdf"
plt.savefig(pjoin("figures", "unsupervised_classification", figname), dpi=600)


# %% LOW POWER ANALYSIS
low_power_lstm_val = pd.DataFrame(low_power_lstm["val"])
low_power_lstm_test = pd.DataFrame(low_power_lstm["test"])


# %% MAPE ANALYSIS
mape_lstm_val = pd.DataFrame(mape_lstm["val"])
mape_lstm_test = pd.DataFrame(mape_lstm["test"])

mape_lstm_val = mape_lstm_val.reindex(sorted_nodes)
mape_lstm_test = mape_lstm_test.reindex(sorted_nodes)


# %% RUNNING TIME
ex_time = int(time() - start_time)
print(f"\nTotal run time :\t{ex_time} [s]  -  {ex_time / 3600:.2f} [h] ")
