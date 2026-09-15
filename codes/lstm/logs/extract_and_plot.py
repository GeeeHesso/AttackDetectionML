from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tfevents_reader import find_event_file, read_scalars

COUNTRY = "CH"
GENS = ["173", "915", "918", "923", "924", "927", "931", "932", "933", "934"]
CFG = "gen_Seq-24ts_Hid-256_Lay-3"

# LOGS = "supervised"
# TAG = "val_f2_score"
# LABEL = "F₂ score (validation set)"
# YLIM = (0.7, 1)

LOGS = "unsupervised"
TAG = "val_rmse_score"
LABEL = "Relative error (validation set)"
YLIM = (0, 0.18)


# ---- 1. Extract into a tidy table -----------------------------------------
rows = []
for gen_id in GENS:
    run_dir = (
        Path(__file__).parent
        / LOGS
        / "lightning_logs"
        / COUNTRY
        / f"{gen_id}_{CFG}"
        / "version_0"
    )
    ev_path = find_event_file(run_dir)
    if ev_path is None:
        raise FileNotFoundError(f"No event file in {run_dir!s}")
    points = sorted((s, v) for s, t, v in read_scalars(ev_path) if t == TAG)
    for epoch_idx, (step, value) in enumerate(points, start=1):
        rows.append(
            {
                "gen_id": gen_id,
                "epoch": epoch_idx,
                "step": step,
                "value": value,
            }
        )

df = pd.DataFrame(rows)
wide = df.pivot(index="epoch", columns="gen_id", values="value")[GENS]

out_file = f"{LOGS}_{TAG}_gen_seq-24"
out_csv = f"{out_file}.csv"
wide.to_csv(out_csv)
print("Wrote", out_csv, wide.shape)
print(wide.head())

# ---- 2. Plot ---------------------------------------------------------------
# Categorical palette (dataviz skill reference palette, validated for CVD
# separation and contrast on a light surface), slots 1-8. Slots 9-10 reuse
# slot 1/2 hues with a dashed linestyle as secondary encoding.
PALETTE = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

fig, ax = plt.subplots(figsize=(6, 3))

# order legend by final-epoch loss (best first) for a readable legend
final_order = wide.iloc[-1].sort_values().index.tolist()

for i, gen_id in enumerate(final_order):
    color = PALETTE[i % 8]
    linestyle = "-" if i < 8 else "--"
    ax.plot(
        wide.index,
        wide[gen_id],
        color=color,
        linestyle=linestyle,
        linewidth=2,
        label=gen_id,
        solid_capstyle="round",
    )

ax.set_xlim(1, 50)
ax.set_ylim(YLIM)

ax.set_xlabel("Epoch")
ax.set_ylabel(LABEL)

# legend = ax.legend(
#     loc="lower right",
#     title="Generator ID",
#     ncols=5,
# )

out_pdf = f"{out_file}.pdf"
fig.tight_layout()
fig.savefig(out_pdf)
print("Wrote", out_pdf)
