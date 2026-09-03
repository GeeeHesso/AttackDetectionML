#!/usr/bin/env python3
"""Grid search cross-validation for supervised models, with optimisation of hyperparameters."""

# %% [0] PACKAGES
import os
import pickle
import sys
import warnings
from itertools import product
from os.path import join as pjoin
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from pandas import read_pickle as rpckl
from sklearn.metrics import confusion_matrix, fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

os.chdir(Path(__file__).resolve().parent.parent)


sys.path.append(os.getcwd())

from functions import add_noise, load_data, load_models, noisy_model_key

warnings.filterwarnings(action="ignore", category=FutureWarning)

start_time = time()  # For total running time


# %% [1] PARAMETERS
os.chdir("..")  # Working directory is repo main

# nets_dict = ['CH', 'DE', 'ES']  # List with all nets
nets_dict = ["CH"]

models_dict = load_models()  # Dictionary with all models
models_dict.pop("mlpr")  # Dictionary with all supervised models
# models_dict.pop('svc')  # SVC have memory problems

types_dict = ["generation", "injection"]  # List with all dataset type

sequence_lens = [0, 4, 24]  # 4 hours or 24 hours
contextual_lens = ["t", "hist"]  # W/O historical values for contextual variable

# Standard deviation [MW] of the zero-mean Gaussian noise added to load/gen
# data before training and testing, to probe model sensitivity to noisy
# input data. Set to None to disable and train on the original data.
noise_std = None
# noise_std = 10

ftwo_scorer = make_scorer(fbeta_score, beta=2)  # Error that weights recall
# Higher than precision


# %%  [2]  DEBUG PARAMETERS
# keys = ["nb", "rf"]  # few minutes
# keys = ['svc', 'knn'] # few hours
# keys = ['gbc', 'mlpc'] # many hours
# keys = ['mlpc']
# models_dict = {key: models_dict[key] for key in keys}
# models_dict.pop('svc')

# types_dict = ['injection']
# sequence_lens = [0]
# contextual_lens = ['t']

# %% [3] NET/MODEL/DATASET LOOP
cartesian = product(
    nets_dict, models_dict.keys(), types_dict, sequence_lens, contextual_lens
)

net_key_old = ""
cartesian_length = (
    len(nets_dict)
    * len(models_dict.keys())
    * len(types_dict)
    * len(sequence_lens)
    * len(contextual_lens)
)

for c, (net_key, model_key, ds_type, seq, contextual) in enumerate(cartesian, start=1):
    if net_key != net_key_old:  # Avoid doing same things
        net_key_old = net_key

        ds_path = pjoin("datasets", net_key)
        res_dir = pjoin("results", "supervised", "single_node_attack", net_key)

        attacked_gens = rpckl(pjoin(ds_path, "attacked_gens.p"))
        attacked_index = rpckl(pjoin(ds_path, "attacked_timesteps.p"))[0].to_list()

        load_p, gen_p, _ = load_data(net_key)

        if noise_std is not None:
            load_p = add_noise(load_p, noise_std, random_state=42)
            gen_p = add_noise(gen_p, noise_std, random_state=43)

        test_index = rpckl(pjoin(ds_path, "test_timesteps.p"))[0].to_list()
        train_index = [ts for ts in gen_p.index if ts not in test_index]

    model = models_dict[model_key]

    # [3.1] LOOP FOR ATTACKED NODE
    for i, attack_gen in enumerate(attacked_gens):
        t_prep = time()  # For fit running time

        # [3.2] LOAD DATASET
        X_all = gen_p.copy()
        if ds_type == "injection":
            X_all = pd.concat([X_all, load_p], axis=1)

        # add anomalies
        attacked_gen_p = rpckl(pjoin(ds_path, f"{attack_gen}_p_attacked.p"))
        if noise_std is not None:
            attacked_gen_p = add_noise(attacked_gen_p, noise_std, random_state=44 + i)
        X_attacked = X_all.copy()
        X_attacked[attack_gen] = attacked_gen_p
        X_attacked.columns = X_attacked.columns.astype(str) + "_t"

        if contextual == "t":
            X_all = X_all[[attack_gen]]

        X_hist = pd.DataFrame()
        for t in range(seq):
            X_temp = np.reshape(X_all.values, (20, 8736, -1))
            X_temp = np.roll(X_temp, seq - t, axis=1)
            X_temp = pd.DataFrame(np.reshape(X_temp, (20 * 8736, -1)))
            X_temp.columns = X_all.columns.astype(str) + f"_t-{seq - t}"
            X_hist = pd.concat([X_hist, X_temp], axis=1)

        X = pd.concat([X_attacked, X_hist], axis=1)

        y = pd.Series(False, index=X.index)
        y[attacked_index] = True

        X_train = X.loc[train_index, :]
        y_train = y[train_index]

        X_test = X.loc[test_index, :]
        y_test = y[test_index]

        # [3.3] TRAIN SIZE LOOP
        train_sizes_eff = []
        (
            n_trains,
            hacked_train,
            time_predict,
            time_train,
        ) = [], [], [], []
        cm, f2_score, f5_score = [], [], []

        train_sizes = [1]
        for train_size in train_sizes:
            # print(9 * '\t', f'train size: {train_size} - '
            #       f'exec: {time()-start_time:.0f} [s]')
            n_train = int(train_size * X_train.shape[0])
            # reduce dataset size for SVC, dividing the size of the dataset by 17
            # (17 is coprime with 24, 7, 52 to avoid unwanted periodicities)
            step = 17 if model_key == "svc" else 1
            X_tr = X_train[:n_train:step]
            y_tr = y_train[:n_train:step]

            # [3.4] SCALER
            X_scaler = None
            if model_key == "mlpc":
                X_scaler = StandardScaler()
                X_tr = X_scaler.fit_transform(X_tr)
                X_test_scaled = X_scaler.transform(X_test)
            else:
                X_test_scaled = X_test

            # [3.5] GRID SEARCH CV
            # model = make_pipeline(MinMaxScaler(), estimator) # /!\
            print(2 * "\n")
            print(f"fit {c}/{cartesian_length}")
            print(f"- {net_key}")
            print(f"- {model_key}")
            print(f"- {attack_gen} ({i + 1}/{len(attacked_gens)})")
            print(f"- train size {train_size}")
            print(f"- type {ds_type}")
            print(f"- sequence {seq}")
            print(f"- contextual {contextual}")
            print(f"- noise_std {noise_std} MW")
            print(f"- X shape {X_train.shape}")
            print(f"- preparation {time() - t_prep:.0f}s")
            t_train = time()

            grid = GridSearchCV(
                estimator=model["estimator"](),
                param_grid=model["parameters"][net_key][ds_type],
                cv=5,
                scoring=ftwo_scorer,
                verbose=0,
                n_jobs=-1,
                refit=True,
            )

            grid.fit(X_tr, y_tr)
            print(f"- training : {time() - t_train:.0f}s")
            print(1 * "\n")
            t_train = time()

            print(2 * "\n")
            print(grid.best_params_)
            print(2 * "\n")
            print("Confusion matrix for test set:")
            print(1 * "\n")
            print("[tn,\tfp,\tfn,\ttp]")

            t0 = time()
            y_predict = grid.best_estimator_.predict(X_test_scaled)
            t_predict = time() - t0

            print(confusion_matrix(y_test, y_predict).ravel(), "\n")
            print(3 * "\n", 60 * "= ")

            # [3.6] KEEP INFOS
            cm.append(confusion_matrix(y_test, y_predict).ravel())
            f2_score.append(fbeta_score(y_test, y_predict, beta=2))
            f5_score.append(fbeta_score(y_test, y_predict, beta=5))
            train_sizes_eff.append(train_size)
            n_trains.append(n_train)
            hacked_train.append(y_train.sum())
            time_predict.append(t_predict)
            time_train.append(t_train)

            # [3.6] SAVE BEST MODEL
            dir_path = pjoin(
                res_dir,
                noisy_model_key(model_key, noise_std),
                ds_type,
                f"{attack_gen}",
                f"sequence_len-{seq}",
                f"contextual_{contextual}",
            )
            os.makedirs(dir_path, exist_ok=True)

            if train_size == 1:
                f_path = pjoin(dir_path, "gscv_trained.p")
                with open(f_path, "wb") as f:
                    pickle.dump(grid.best_estimator_, f)

                f_path = pjoin(dir_path, "gscv_best_params.p")
                with open(f_path, "wb") as f:
                    pickle.dump(grid.best_params_, f)

                if X_scaler is not None:
                    f_path = pjoin(dir_path, "scaler.p")
                    with open(f_path, "wb") as f:
                        pickle.dump(X_scaler, f)

        # [10] SAVE INFOS
        cm = np.array(cm)
        confusion_df = pd.DataFrame(
            {
                "train_size": train_sizes_eff,
                "train_sample": n_trains,
                "train_hacked": hacked_train,
                "train_occ": np.divide(hacked_train, n_trains),
                "test_hacked": y_test.sum(),
                "test_occ": y_test.sum() / len(y_test),
                "f2_score": f2_score,
                "f5_score": f5_score,
                "tn": cm[:, 0],
                "fp": cm[:, 1],
                "fn": cm[:, 2],
                "tp": cm[:, 3],
            }
        )

        compute_time = pd.DataFrame(
            {
                "inputs": X_tr.shape[1],
                "#_predict_sample": len(y_test),
                "predict_time_second": time_predict,
                "train_size": train_sizes_eff,
                "train_sample": n_trains,
                "train_time_second": time_train,
            }
        )

        confusion_df.to_pickle(pjoin(dir_path, "confusion_df.p"))
        compute_time.to_pickle(pjoin(dir_path, "computing_time.p"))

        mistakes_df = pd.DataFrame({"label": y_test, "classification": y_predict})
        mistakes_df["mistakes"] = mistakes_df["label"] ^ mistakes_df["classification"]
        mistakes_df.to_pickle(pjoin(dir_path, "mistakes.p"))


# %% [4] RUNNING TIME
ex_time = int(time() - start_time)
print(f"\nTotal run time :\t{ex_time} [s]  -  {ex_time / 3600:.2f} [h] ")
