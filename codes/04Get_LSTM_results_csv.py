#!/usr/bin/env python3
"""Recover LSTM results obtained in in Google Colab."""

# %% PACKAGES
import os
import shutil
from itertools import product
from os.path import join as pjoin
from pathlib import Path

import pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)


nets = ["CH", "DE", "ES"]  # List with all nets
models = ["lstm"]  # List with colab models
types = ["injection", "generation"]  # List with all dataset type

approaches = ["supervised", "unsupervised"]  # List with all ml approaches

sequences = [4, 24]

lstm_dir = pjoin("codes", "lstm")

# %% LOOP

cartesian = product(approaches, nets, models, types, sequences)

net_key_old = ""
for approach, net_key, model_key, ds_type, seq in cartesian:
    ds_path = pjoin("datasets", net_key)
    attacked_gens = pd.read_pickle(pjoin(ds_path, "attacked_gens.p"))

    src_dir = pjoin(lstm_dir, "results", approach, "single_node_attack", net_key)

    if approach == "supervised":
        dst_dir = pjoin("results", "supervised", "single_node_attack", net_key)
    else:
        dst_dir = pjoin("results", "unsupervised", net_key, "regression")

    # ANOMALY LOAD LOOP
    for i, attack_gen in enumerate(attacked_gens):
        src_path = pjoin(
            src_dir,
            model_key,
            ds_type,
            f"{attack_gen}",
            f"sequence_len-{seq}",
            "contextual_hist",
        )
        if not os.path.isdir(src_path):
            continue

        dst_path = pjoin(
            dst_dir,
            model_key,
            ds_type,
            f"{attack_gen}",
            f"sequence_len-{seq}",
            "contextual_hist",
        )
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        print(dst_path)

        # CSV FILES
        csv_files = [
            "confusion_df",
            "mistakes",
            "gscv_regression_metrics",
            "prediction_test_set",
            "prediction_validation_set",
        ]

        for file in csv_files:
            src_file = pjoin(src_path, file + ".csv")
            if not os.path.isfile(src_file):
                continue
            df = pd.read_csv(src_file)  # Load CSV results

            if file == "gscv_regression_metrics":
                df = df.set_index("Unnamed: 0")
                df.index.name = "index"
            else:
                df = df.drop(columns=["Unnamed: 0"])

            df.to_pickle(pjoin(dst_path, file + ".p"))  # Resave in PICKLE

        # TRAINED MODEL
        shutil.copy2(
            pjoin(src_path, "best_checkpoint.ckpt"),
            pjoin(dst_path, "best_checkpoint.ckpt"),
        )
