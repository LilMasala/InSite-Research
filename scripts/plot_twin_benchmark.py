"""Render the reviewed aggregate benchmark summary; no individual records read."""
from pathlib import Path
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "insite-portfolio-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
summary = json.loads((ROOT / "examples/twin-benchmark-summary.json").read_text())
results = summary["results"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
fig.patch.set_facecolor("#f6f9fd")
colors = ["#376f91", "#ab7b8e", "#769a90"]
for ax in axes:
    ax.set_facecolor("#f6f9fd")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
for ax, cohort, sub in zip(axes[:2], ["HUPA-UCM", "T1D-UOM"],
                            ["17 people · 217 windows", "4 people · 56 windows"]):
    metrics = results[cohort]["forecastRmseMgDl"]
    keys = ["t1dTwinV2Warm", "t1dsimAiPersonalLearningRate1eMinus3"]
    labels = ["InSite twin v2\n(warm start)", "T1DSim_AI\n(tuned personal)"]
    if "replayBgCold" in metrics:
        keys.append("replayBgCold")
        labels.append("ReplayBG\n(cold start)")
    values = [metrics[k] for k in keys]
    ax.barh(np.arange(len(keys)), values, color=colors[:len(keys)], height=.53)
    ax.set_yticks(np.arange(len(keys)), labels)
    ax.invert_yaxis()
    for i, value in enumerate(values):
        ax.text(value + .7, i, f"{value:.2f}", va="center", fontsize=10)
    ax.set_xlim(0, 61)
    ax.set_xlabel("Mean sequence RMSE (mg/dL) ↓")
    ax.set_title(f"{cohort}\n{sub}", loc="left", fontsize=11, pad=15, weight="bold")
low = results["HUPA-UCM"]["hypoglycemia"]["metrics"]
values = [low[k]["auroc"] for k in ["t1dTwinV2WarmProbability",
          "t1dsimAiTunedPlusNoiseProbability", "replayBgColdPlusNoiseProbability"]]
ax = axes[2]
ax.barh(np.arange(3), values, color=colors, height=.53)
ax.set_yticks(np.arange(3), ["InSite twin v2", "T1DSim_AI\n+ CGM noise", "ReplayBG\n+ CGM noise"])
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.axvline(.5, color="#8893a0", linestyle=":", linewidth=1)
for i, value in enumerate(values):
    ax.text(value + .015, i, f"{value:.3f}", va="center", fontsize=10)
ax.set_title("HUPA-UCM low events\n50 events / 217 windows", loc="left", fontsize=11, pad=15, weight="bold")
ax.set_xlabel("AUROC ↑")
fig.suptitle("Held-out five-hour replay with recorded meals and insulin", x=.04, ha="left", fontsize=15, weight="bold")
fig.text(.04, .095, "Warm-start history differs between models. Paired RMSE intervals include zero for personalized comparators.", fontsize=9, color="#465567")
fig.text(.04, .05, "HUPA AUROC difference vs T1DSim_AI + noise: +0.142 [0.058, 0.222]; vs ReplayBG + noise: +0.065 [0.016, 0.126].", fontsize=9, color="#465567")
fig.text(.04, .015, "Brackets: 95% participant-cluster bootstrap intervals. Saved exploratory benchmark; see methods and provenance.", fontsize=9, color="#465567")
fig.subplots_adjust(left=.10, right=.97, top=.77, bottom=.24, wspace=.65)
fig.savefig(ROOT / "assets/twin-benchmark.png", dpi=180, facecolor=fig.get_facecolor())
plt.close(fig)
print("Rendered aggregate benchmark figure.")
