#!/usr/bin/env python3
"""Compare unsupervised classification F2 score: optimized vs fixed threshold.

Reads the results produced by codes/unsupervised/03Classification.py, run
once with optimize_threshold = True (the anomaly-detection threshold is
optimized on the validation set - the default, partly-supervised approach)
and once with optimize_threshold = False (the threshold is fixed to half the
generator's rated power instead - a fully unsupervised baseline). The
optimize_threshold = False run is saved under a distinct model key (see
functions.threshold_model_key) so it never overwrites, or gets mixed into,
the optimize_threshold = True results consumed by the regular comparison
script (codes/unsupervised/04Results_classification_unique.py).

Switching optimize_threshold does not change the regression model or its
predictions at all - it only changes how those predictions are turned into a
classification threshold - so there is no RMSE difference to report here
(unlike codes/10Noise_sensitivity.py, where noise does change regression
accuracy). Only the resulting best_threshold and F2 score are compared.
"""

# %% PACKAGES
import os
import sys
from itertools import product
from os.path import join as pjoin
from pathlib import Path

import pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)

sys.path.append(os.getcwd())
from functions import get_gen_names, threshold_model_key

# %% PARAMETERS
os.chdir("..")  # Working directory is repo main

case = "CH"

# Models covered by codes/unsupervised/03Classification.py's threshold step.
model_keys = ["mlpr", "lstm"]
optimize_thresholds = [True, False]

types_dict = ["generation", "injection"]
sequence_lens = [4, 24]
contextual_lens = ["t", "hist"]


# %% LOAD RESULTS
dir_dataset = pjoin("datasets", case)
path_result = pjoin("results", "unsupervised", case, "regression")

attacked_gens = pd.read_pickle(pjoin(dir_dataset, "attacked_gens.p"))

cartesian = product(
    model_keys, optimize_thresholds, types_dict, sequence_lens, contextual_lens
)

f2_all = pd.DataFrame()
missing = []
for model_key, optimize_threshold, ds_type, seq, contextual in cartesian:
    save_key = threshold_model_key(model_key, optimize_threshold)

    for attacked_gen in attacked_gens:
        res_path = pjoin(
            path_result,
            save_key,
            ds_type,
            f"{attacked_gen}",
            f"sequence_len-{seq}",
            f"contextual_{contextual}",
        )

        f_path = pjoin(res_path, "confusion_test_set.p")
        if not os.path.isfile(f_path):
            missing.append(
                (model_key, optimize_threshold, ds_type, seq, contextual, attacked_gen)
            )
            continue

        result_df = pd.read_pickle(f_path)

        row = pd.DataFrame(
            {
                "model": model_key,
                "optimize_threshold": optimize_threshold,
                "ds_type": ds_type,
                "sequence": seq,
                "contextual": contextual,
                "attacked_gen": attacked_gen,
                "best_threshold": result_df["best_threshold"].iat[0],
                "f2_score": result_df["f2_score"].iat[0],
            },
            index=[0],
        )
        f2_all = pd.concat([f2_all, row], ignore_index=True)

if missing:
    print(f"Missing {len(missing)} result(s), e.g.:")
    for (
        model_key,
        optimize_threshold,
        ds_type,
        seq,
        contextual,
        attacked_gen,
    ) in missing[:10]:
        print(
            f"  - model={model_key}, optimize_threshold={optimize_threshold}, "
            f"ds_type={ds_type}, seq={seq}, contextual={contextual}, "
            f"attacked_gen={attacked_gen}"
        )

f2_all["gen_name"] = f2_all["attacked_gen"].map(get_gen_names(case))


# %% BUILD COMPARISON TABLE (ONE ROW PER MODEL/NODE/DATASET CONFIG)
index_cols = ["model", "gen_name", "ds_type", "sequence", "contextual"]

f2_wide = f2_all.pivot(
    index=index_cols, columns="optimize_threshold", values="f2_score"
)
f2_wide = f2_wide.rename(columns={True: "f2_optimized", False: "f2_fixed"})

threshold_wide = f2_all.pivot(
    index=index_cols, columns="optimize_threshold", values="best_threshold"
)
threshold_wide = threshold_wide.rename(
    columns={True: "threshold_optimized", False: "threshold_fixed"}
)

comparison = pd.concat([f2_wide, threshold_wide], axis=1)
comparison["f2_diff"] = comparison["f2_optimized"] - comparison["f2_fixed"]
comparison = comparison.sort_values("f2_diff")

comparison.to_csv(pjoin("results", f"{case}_threshold_comparison_f2.csv"))

print(f"\nMedian F2 (optimized threshold) : {comparison['f2_optimized'].median():.3f}")
print(f"Median F2 (fixed threshold)     : {comparison['f2_fixed'].median():.3f}")
print(f"Median F2 difference            : {comparison['f2_diff'].median():+.3f}")
