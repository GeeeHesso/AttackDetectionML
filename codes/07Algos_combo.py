#!/usr/bin/env python3

import os
from os.path import join as pjoin
from pathlib import Path

import pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)


cases = ["CH"]
models = [
    ("gbc", "generation", 4, "hist"),
    ("mlpc", "generation", 4, "hist"),
    ("lstmc", "generation", 24, "hist"),
    ("mlpr", "generation", 4, "hist"),
    ("lstmr", "generation", 4, "hist"),
]

n = len(models)
assert n > 1


# %% mistakes count
count = {}
label = None
mistakes = {}

print("List of cases:")
for model, ds_type, seq, context in models:
    count[model] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    unsupervised = model in ["mlpr", "lstmr"]

    for net in cases:
        net_dir = (
            pjoin("results", "unsupervised", net, "regression")
            if unsupervised
            else pjoin("results", "supervised", "single_node_attack", net)
        )

        dataset_dir = pjoin("datasets", net)
        attacked_gens = pd.read_pickle(pjoin(dataset_dir, "attacked_gens.p"))

        for gen in attacked_gens:
            file = pjoin(
                net_dir,
                "lstm" if model in ["lstmc", "lstmr"] else model,
                ds_type,
                str(gen),
                f"sequence_len-{seq}",
                f"contextual_{context}",
                "mistakes_test_set.p" if unsupervised else "mistakes.p",
            )
            if not os.path.isfile(file):
                print(f" - {model}-{net}-{gen} NOT FOUND ({file})")
                continue
            df = pd.read_pickle(file).astype(bool).reset_index(drop=True)

            if label is None:
                label = df["label"]
            else:
                assert label.equals(df["label"]), (
                    f"{model}-{net}-{gen} INCONSISTENT LABEL"
                )

            if gen not in mistakes:
                mistakes[gen] = df["mistakes"].astype(int)
            else:
                mistakes[gen] += df["mistakes"].astype(int)

            prediction = df["classification"]

            count[model]["fp"] += (~label & prediction).values.sum()
            count[model]["fn"] += (label & ~prediction).values.sum()
            count[model]["tp"] += (label & prediction).values.sum()
            count[model]["tn"] += (~label & ~prediction).values.sum()

            print(f" - {model} {net}-{gen} {ds_type}-{seq}-{context}")

# %%% summary for each model
for model, _, _, _ in models:
    print(f"\n\n{model.upper()}")
    precision = count[model]["tp"] / (count[model]["tp"] + count[model]["fp"])
    recall = count[model]["tp"] / (count[model]["tp"] + count[model]["fn"])
    f2 = 5 * precision * recall / (4 * precision + recall)
    print(" - true negatives  :", count[model]["tn"])
    print(" - true positives  :", count[model]["tp"])
    print(" - false positives :", count[model]["fp"])
    print(" - false negatives :", count[model]["fn"])
    print(f" - precision       : {precision:.3f}")
    print(f" - recall          : {recall:.3f}")
    print(f" - F2              : {f2:.3f}")


# %%% summary for all models
all_models = " - ".join([model[0].upper() for model in models])

false_positives = pd.concat(
    [val[label == False] for val in mistakes.values()], ignore_index=True
).value_counts()
false_negatives = pd.concat(
    [val[label == True] for val in mistakes.values()], ignore_index=True
).value_counts()

print(f"\n\nAll models {all_models}")
print(" - true negatives        :", false_positives[0])
print(" - true positives        :", false_negatives[0])
for i in range(n):
    print(f" - false positives ({i + 1}/{n}) :", false_positives[i + 1])
for i in range(n):
    print(f" - false negatives ({i + 1}/{n}) :", false_negatives[i + 1])

# %%% combination ANY

tn = false_positives[0]
tp = sum(false_negatives[i] for i in range(n))
fn = false_negatives[n]
fp = sum(false_positives[i + 1] for i in range(n))

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f2 = 5 * precision * recall / (4 * precision + recall)

print(f"\n\nCombination: any of {all_models}")
print(" - true negatives  :", tn)
print(" - true positives  :", tp)
print(" - false positives :", fp)
print(" - false negatives :", fn)
print(f" - precision       : {precision:.3f}")
print(f" - recall          : {recall:.3f}")
print(f" - F2              : {f2:.3f}")

# %%% combination ALL

tn = sum(false_positives[i] for i in range(n))
tp = false_negatives[0]
fn = sum(false_negatives[i + 1] for i in range(n))
fp = false_positives[n]

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f2 = 5 * precision * recall / (4 * precision + recall)

print(f"\n\nCombination: all of {all_models}")
print(" - true negatives  :", tn)
print(" - true positives  :", tp)
print(" - false positives :", fp)
print(" - false negatives :", fn)
print(f" - precision       : {precision:.3f}")
print(f" - recall          : {recall:.3f}")
print(f" - F2              : {f2:.3f}")


# %%% combination with majority rule

if n > 2 and n % 2 == 1:
    half_n = n // 2
    tn = sum(false_positives[i] for i in range(half_n + 1))
    tp = sum(false_negatives[i] for i in range(half_n + 1))
    fn = sum(false_negatives[i] for i in range(half_n + 1, n + 1))
    fp = sum(false_positives[i] for i in range(half_n + 1, n + 1))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f2 = 5 * precision * recall / (4 * precision + recall)

    print(f"\n\nCombination: majority of {all_models}")
    print(" - true negatives  :", tn)
    print(" - true positives  :", tp)
    print(" - false positives :", fp)
    print(" - false negatives :", fn)
    print(f" - precision       : {precision:.3f}")
    print(f" - recall          : {recall:.3f}")
    print(f" - F2              : {f2:.3f}")

# %%% combination with majority or equality rule
if n > 2 and n % 2 == 0:
    half_n = n // 2
    tn = sum(false_positives[i] for i in range(half_n + 1))
    tp = sum(false_negatives[i] for i in range(half_n + 1))
    fn = sum(false_negatives[i] for i in range(half_n + 1, n + 1))
    fp = sum(false_positives[i] for i in range(half_n + 1, n + 1))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f2 = 5 * precision * recall / (4 * precision + recall)

    print(f"\n\nCombination: majority or equality among {all_models}")
    print(" - true negatives  :", tn)
    print(" - true positives  :", tp)
    print(" - false positives :", fp)
    print(" - false negatives :", fn)
    print(f" - precision       : {precision:.3f}")
    print(f" - recall          : {recall:.3f}")
    print(f" - F2              : {f2:.3f}")

# %%% combination with strict majority rule
if n > 2 and n % 2 == 0:
    half_n = (n - 1) // 2
    tn = sum(false_positives[i] for i in range(half_n + 1))
    tp = sum(false_negatives[i] for i in range(half_n + 1))
    fn = sum(false_negatives[i] for i in range(half_n + 1, n + 1))
    fp = sum(false_positives[i] for i in range(half_n + 1, n + 1))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f2 = 5 * precision * recall / (4 * precision + recall)

    print(f"\n\nCombination: strict majority of {all_models}")
    print(" - true negatives  :", tn)
    print(" - true positives  :", tp)
    print(" - false positives :", fp)
    print(" - false negatives :", fn)
    print(f" - precision       : {precision:.3f}")
    print(f" - recall          : {recall:.3f}")
    print(f" - F2              : {f2:.3f}")
