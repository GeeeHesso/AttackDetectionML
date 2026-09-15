import pandas as pd

f2_scores = {
    country: pd.read_csv(f"results/{country}_f2_scores.csv").drop("count", axis=1)
    for country in ["CH", "DE", "ES"]
}

f2_combined = pd.concat(
    [d.set_index(d.columns[0]) for d in f2_scores.values()],
    keys=f2_scores.keys(),
    names=["Network", "Algorithm"],
)

print(
    f2_combined.to_latex(
        multirow=True,
        column_format="ll" + "r" * (f2_combined.shape[1]),
        float_format="%.3f",
        escape=True,
    )
)


multiple_attacks_f2 = pd.read_csv(
    "results/CH_multiple_attacks_f2.csv", index_col=0
).drop("count", axis=1)

print(multiple_attacks_f2.to_latex(float_format="%.4f"))
