#!/usr/bin/env python3
"""Grid search cross-validation for unsupervised models, with optimisation of hyperparameters."""

# %% PACKAGES
import os
import pickle
import sys
from itertools import product
from os.path import join as pjoin
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from pandas import read_pickle as rpckl
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler

# from sklearn.metrics import confusion_matrix, fbeta_score, make_scorer

os.chdir(Path(__file__).resolve().parent.parent)

sys.path.append(os.getcwd())
from functions import load_data, load_models, years_timesteps

start_time = time()  # For total running time


# %% PARAMETERS
os.chdir("..")  # Working directory is repo main

case = "CH"

nets_dict = [case]  # List with all nets
types_dict = ["generation", "injection"]  # List with all dataset type

models_dict = load_models()  # Dictionary with all models
models_dict = {"mlpr": models_dict["mlpr"]}  # Dictionary with all regression models

sequence_lens = [4, 24]  # 6 hours or 1 hour
contextual_lens = ["t", "hist"]  # W/O historical values for contextual variable

# %%% TRAIN/VALIDATION SPLIT METHOD
# "random": 80/20 random split of the non-test timesteps (original method)
# "fixed_years": entire simulated years held out as validation set
val_split_method = "fixed_years"

# Only used when val_split_method == "fixed_years"
# Each entry is a (calendar year, series) pair, e.g. (2018, 3) is the third
# simulated replica of 2018 (raw_data/gens_2018_3.csv).
val_years = [(2018, 3), (2019, 3), (2020, 3)]


# %% DEBUG PARAMETERS
# keys = ["mlpr"]  # Select specific models
# models_dict = {key: models_dict[key] for key in keys}

types_dict = ["generation"]  # Select specific ds_type
# types_dict = ['injection'] # Select specific ds_type

# sequence_lens = [1, 2, 3, 8, 48, 96, 168]

# contextual_lens = ["t"]  # Select specific sequence length
# contextual_lens = ['hist'] # Select specific sequence length


# %% NET/MODEL/DATASET LOOP
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
        dir_dataset = pjoin("datasets", net_key)
        path_result = pjoin("results", "unsupervised", net_key, "regression")

        attacked_gens = rpckl(pjoin(dir_dataset, "attacked_gens.p"))
        # attacked_index = rpckl(pjoin(dir_dataset, 'attacked_timesteps.p'))[0].to_list()

        load_p, gen_p, _ = load_data(case)

        # test_index is the index for testing the classification
        classification_test_index = rpckl(pjoin(dir_dataset, "test_timesteps.p"))[
            0
        ].to_list()
        regression_index = [
            ts for ts in gen_p.index if ts not in classification_test_index
        ]

        if val_split_method == "fixed_years":
            regression_val_index = [
                ts for ts in years_timesteps(val_years) if ts in regression_index
            ]
            regression_train_index = [
                ts for ts in regression_index if ts not in regression_val_index
            ]
        elif val_split_method != "random":
            raise ValueError(f"Unknown val_split_method: {val_split_method!r}")

    model = models_dict[model_key]

    # LOOP FOR ATTACKED NODE
    for n_gen, attack_gen in enumerate(attacked_gens):
        fit_time = time()  # For fit running time

        # LOAD DATASET
        X_all = gen_p.copy()
        if ds_type == "injection":
            X_all = pd.concat([X_all, load_p], axis=1)

        y = X_all[attack_gen].to_frame()
        y.columns = y.columns.astype(str) + "_t"
        X_context = X_all.drop(columns=attack_gen)
        X_context.columns = X_context.columns.astype(str) + "_t"

        if contextual == "t":
            X_all = X_all[[attack_gen]]

        X_hist = pd.DataFrame()
        for i in range(seq):
            X_temp = np.reshape(X_all.values, (20, 8736, -1))
            X_temp = np.roll(X_temp, seq - i, axis=1)
            X_temp = pd.DataFrame(np.reshape(X_temp, (20 * 8736, -1)))
            X_temp.columns = X_all.columns.astype(str) + f"_t-{seq - i}"
            X_hist = pd.concat([X_hist, X_temp], axis=1)

        X = pd.concat([X_context, X_hist], axis=1)

        X_test = X.iloc[classification_test_index, :]
        y_test = y.iloc[classification_test_index, :]

        if val_split_method == "random":
            X_train_val = X.iloc[regression_index, :]
            y_train_val = y.iloc[regression_index, :]

            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.2, random_state=42
            )
        else:  # "fixed_years"
            X_train = X.iloc[regression_train_index, :]
            y_train = y.iloc[regression_train_index, :]
            X_val = X.iloc[regression_val_index, :]
            y_val = y.iloc[regression_val_index, :]

        ## SAVE INDEX FOR GOOGLE COLAB
        reg_val_idx = y_val.reset_index()["index"].sort_values()
        reg_val_idx.to_pickle(pjoin(dir_dataset, "regression_validation_timesteps.p"))
        reg_val_idx = rpckl(
            pjoin(dir_dataset, "regression_validation_timesteps.p")
        ).to_list()

        reg_train_idx = y_train.reset_index()["index"].sort_values()
        reg_train_idx.to_pickle(pjoin(dir_dataset, "regression_train_timesteps.p"))
        reg_train_idx = rpckl(
            pjoin(dir_dataset, "regression_train_timesteps.p")
        ).to_list()

        # SCALER
        X_scaler = MinMaxScaler()
        X_train_scaled = X_scaler.fit_transform(X_train)
        X_val_scaled = X_scaler.transform(X_val)
        X_test_scaled = X_scaler.transform(X_test)

        y_scaler = MinMaxScaler()
        y_train_scaled = y_scaler.fit_transform(y_train)

        # GRID SEARCH CV
        # model = make_pipeline(MinMaxScaler(), estimator) # /!\
        print(2 * "\n")
        print(f"fit {c}/{cartesian_length}")
        print(f"- network       : {net_key}")
        print(f"- model         : {model_key}")
        print(f"- attacked gen  : {attack_gen} ({n_gen + 1}/{len(attacked_gens)})")
        print(f"- type          : {ds_type}")
        print(f"- sequence      : {seq}")
        print(f"- contextual    : {contextual}")
        print(f"- X_train shape : {X_train.shape}")
        print(f"- preparation   : {time() - fit_time:.0f}s")
        fit_time = time()

        # param_key = f'{ds_type}_{seq}_{contextual}'
        param_key = (
            f"{ds_type}_24_{contextual}" if seq == 24 else f"{ds_type}_4_{contextual}"
        )
        grid = GridSearchCV(
            estimator=model["estimator"](),
            param_grid=model["parameters"][net_key][param_key],
            cv=5,
            scoring="sc",
            verbose=0,
            n_jobs=-1,
            refit=True,
        )

        grid.fit(X_train_scaled, y_train_scaled.ravel())  # GSCV

        print(f"- training      : {time() - fit_time:.0f}s")

        # PREDICTION

        # VALIDATION SET PREDICTION
        y_val_predict_scaled = grid.best_estimator_.predict(X_val_scaled)
        y_val_predict = y_scaler.inverse_transform(y_val_predict_scaled.reshape(-1, 1))

        ## Metrics
        r2_val = r2_score(y_val, y_val_predict)
        rmse_val = mean_squared_error(y_val, y_val_predict, squared=False)

        # TEST SET PREDICTION
        y_test_predict_scaled = grid.best_estimator_.predict(X_test_scaled)
        y_test_predict = y_scaler.inverse_transform(
            y_test_predict_scaled.reshape(-1, 1)
        )

        ## Metrics
        r2_test = r2_score(y_test, y_test_predict)
        rmse_test = mean_squared_error(y_test, y_test_predict, squared=False)

        # SAVE
        metrics = pd.DataFrame(
            {
                "r\u00b2": [r2_val, r2_test],
                "rmse": [rmse_val, rmse_test],
            },
            index=["validation set", "test set"],
        )
        print(1 * "\n")
        print(grid.best_params_)
        print(1 * "\n")
        print(metrics.round(3))
        print(3 * "\n", 30 * "= ")

        # SAVE MODEL & HYPERPARAMS
        dir_result = pjoin(
            path_result,
            f"{model_key}",
            ds_type,
            f"{attack_gen}",
            f"sequence_len-{seq}",
            f"contextual_{contextual}",
        )

        os.makedirs(dir_result, exist_ok=True)

        f_path = pjoin(dir_result, "gscv_trained.p")
        with open(f_path, "wb") as f:
            pickle.dump(grid.best_estimator_, f)

        f_path = pjoin(dir_result, "gscv_best_params.p")
        with open(f_path, "wb") as f:
            pickle.dump(grid.best_params_, f)

        # SAVE METRICS
        metrics.to_pickle(pjoin(dir_result, "gscv_regression_metrics.p"))

        # SAVE PREDICTIONS
        pd.DataFrame(
            {
                "y_val": y_val.iloc[:, 0],
                "y_val_predict": y_val_predict.ravel(),
            },
            index=y_val.index,
        ).to_pickle(pjoin(dir_result, "prediction_validation_set.p"))

        pd.DataFrame(
            {
                "y_test": y_test.iloc[:, 0],
                "y_test_predict": y_test_predict.ravel(),
            },
            index=y_test.index,
        ).to_pickle(pjoin(dir_result, "prediction_test_set.p"))

        # SAVE SCALERS
        f_path = pjoin(dir_result, "scaler_x.p")
        with open(f_path, "wb") as f:
            pickle.dump(X_scaler, f)

        f_path = pjoin(dir_result, "scaler_y.p")
        with open(f_path, "wb") as f:
            pickle.dump(y_scaler, f)

# %% RUNNING TIME
ex_time = int(time() - start_time)
print(f"\nTotal run time :\t{ex_time} [s]  -  {ex_time / 3600:.2f} [h] ")
