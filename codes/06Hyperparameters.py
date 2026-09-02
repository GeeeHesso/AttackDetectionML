#!/usr/bin/env python3
"""Display optimal hyperparameters for all trained models."""

import os
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)

cases = ["CH", "DE", "ES"]
models = ["gbc"]
params = []
ds_types = ["injection", "generation"]
sequence_lens = [4, 24]
contextual_lens = ["t", "hist"]


for model in models:
    for ds_type in ds_types:
        for seq in sequence_lens:
            for context in contextual_lens:
                print(
                    "\nModel: ",
                    model,
                    "  type: ",
                    ds_type,
                    " sequence: ",
                    seq,
                    " context:",
                    context,
                )
                params_list = {p: [] for p in params}
                for net in cases:
                    dataset_dir = pjoin("datasets", net)
                    attacked_gens = pd.read_pickle(
                        pjoin(dataset_dir, "attacked_gens.p")
                    )

                    result_dir = (
                        pjoin("results", "unsupervised", net, "regression")
                        if model in ["mlpr", "lstmr"]
                        else pjoin("results", "supervised", "single_node_attack", net)
                    )

                    for gen in attacked_gens:
                        gen_dir = pjoin(result_dir, model, ds_type, str(gen))

                        gen_dir = pjoin(
                            gen_dir, f"sequence_len-{seq}", f"contextual_{context}"
                        )

                        file = pjoin(gen_dir, "gscv_best_params.p")
                        if not os.path.isfile(file):
                            continue
                        df = pd.read_pickle(file)
                        print(f"{net}-{gen}", df)
                        for p in params:
                            if p in df:
                                params_list[p].append(df[p])
                print(
                    "Min     : ",
                    {p: np.array(values).min() for p, values in params_list.items()},
                )
                print(
                    "Max     : ",
                    {p: np.array(values).max() for p, values in params_list.items()},
                )
                print(
                    "Average : ",
                    {p: np.array(values).mean() for p, values in params_list.items()},
                )
