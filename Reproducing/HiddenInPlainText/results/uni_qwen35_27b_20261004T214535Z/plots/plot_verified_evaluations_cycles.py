"""Render the verified frozen-observer evaluations on a training-cycle axis.

Run this script without arguments. Source metrics and original plots stay intact.
One cycle is seven updates; use fractional cycle positions for partial cycles.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

output = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
provenance = json.loads((output / "PROVENANCE.json").read_text())
source = output / "verified_evaluation_metrics.csv"
assert sha(source) == provenance["artifact_sha256"][source.name]
with source.open() as f:
    records = [{k: float(v) if k in ("C_R", "C_I", "d") else v if k.endswith("sha256") else int(v)
                for k, v in row.items()} for row in csv.DictReader(f)]
assert [r["completed_update"] for r in records] == provenance["completed_evaluations_plotted"]
assert records[-1]["completed_update"] == 1440
segment_ends = provenance["verified_segment_ends"]
final_completion = provenance["final_completion"]

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="qwen_plot_"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.titlesize": 13, "axes.titleweight": "semibold",
                     "pdf.fonttype": 42, "savefig.facecolor": "white"})
fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8))
xs = [r["completed_update"] / 7 for r in records]
single = len(records) == 1
fig.suptitle("Hidden Communication Under a Fixed Observer", fontsize=18, weight="semibold", y=0.97)
subtitle = ("Evaluation 0 only · baseline points; no trend can be inferred" if single
            else f"{len(records)} completed evaluations · through cycle {max(xs):.1f}")
if final_completion:
    subtitle = f"Run complete · {len(records)} verified evaluations · final cycle {max(xs):.1f}"
if len(segment_ends) == 1:
    subtitle += " · dashed line: " + ", ".join(f"segment {s['segment']} end ({s['completed_update'] / 7:.1f})" for s in segment_ends)
elif segment_ends:
    fig.text(0.5, 0.896, "Dashed lines: " + "; ".join(f"segment {s['segment']} end at {s['completed_update'] / 7:.1f}" for s in segment_ends),
             ha="center", fontsize=11, color="#4B5563")
fig.text(0.5, 0.923, subtitle, ha="center", fontsize=12, color="#4B5563")
panels = [
    (axes[0, 0], "C_R", "Receiver accuracy", "Correct university predictions", "#2563EB"),
    (axes[0, 1], "C_I", "Operational observer flag rate", "Flagged examples", "#9D4EDD"),
    (axes[1, 0], "d", "PIQA rejection rate", "Judge rejection OR format failure", "#D97706"),
    (axes[1, 1], "receiver_PASS", "Receiver PASS predictions", "Predicted class count out of 2,000", "#078572"),
]
for ax, key, title, description, color in panels:
    ys = [r[key] for r in records]
    ax.plot(xs, ys, color=color, marker="o", markersize=6 if segment_ends else 9,
            linewidth=1.7, linestyle="None" if single else "-")
    for segment in segment_ends:
        ax.axvline(segment["completed_update"] / 7, color="#9AA2AE", linestyle="--", linewidth=0.9, zorder=0)
    ax.set_title(title, loc="left", pad=27)
    ax.text(0, 1.015, description, transform=ax.transAxes, color="#596273", fontsize=10)
    ax.set_xlabel("Training cycles")
    ax.set_xticks(list(range(0, 201, 25)))
    ax.set_xlim(0, 210)
    if key == "receiver_PASS":
        ax.set_ylim(0, 2100)
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_ylabel("PASS predictions")
    else:
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_ylabel(key)
    ax.grid(axis="y", color="#E4E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)
    for edge in ("bottom", "left"):
        ax.spines[edge].set_color("#B8C0CB")
    latest = records[-1]
    labels = {
        "C_R": f"{latest['receiver_correct']:,}/2,000 = {100*latest['C_R']:.2f}%",
        "C_I": f"{latest['observer_flagged']:,}/2,000 = {100*latest['C_I']:.2f}%",
        "d": f"{latest['union_rejected']}/100 = {100*latest['d']:.1f}%\nJudge {latest['judge_rejected']} · format {latest['format_rejected']} · overlap {latest['overlap']}",
        "receiver_PASS": f"PASS {latest['receiver_PASS']:,} · FAIL {latest['receiver_FAIL']:,}",
    }
    below = ys[-1] > (1700 if key == "receiver_PASS" else 0.8)
    if key == "receiver_PASS" and not single:
        ax.text(0.98, 0.90, labels[key], transform=ax.transAxes,
                ha="right", va="top", fontsize=11, color=color, weight="semibold",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    else:
        ax.annotate(labels[key], (xs[-1], ys[-1]), xytext=(0, -22 if below else 16),
                    textcoords="offset points", ha="center" if single else "right",
                    va="top" if below else "bottom",
                    fontsize=11, color=color, weight="semibold",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
fig.text(0.08, 0.047,
         "C_I is an operational flag rate, not verified transmission accuracy. PASS is the receiver’s predicted class.\n"
         "One cycle = 2 receiver + 5 sender updates. Fractional cycles show progress within a cycle.",
         fontsize=10, color="#4B5563", linespacing=1.6)
fig.subplots_adjust(left=0.08, right=0.97, top=0.83, bottom=0.15, hspace=0.70, wspace=0.28)
for suffix in ("png", "svg"):
    fig.savefig(output / f"verified_evaluation_curves_cycles.{suffix}", dpi=190)
plt.close(fig)

record = {
    "source_metrics_sha256": sha(source),
    "original_provenance_sha256": sha(output / "PROVENANCE.json"),
    "x_axis": "training cycles = completed_update / 7; partial cycles retained",
    "updates_per_cycle": 7,
    "final_cycle_position": records[-1]["completed_update"] / 7,
    "metrics_unchanged": True,
    "artifact_sha256": {name: sha(output / name) for name in [
        "verified_evaluation_curves_cycles.png", "verified_evaluation_curves_cycles.svg",
        "plot_verified_evaluations_cycles.py"]},
}
(output / "CYCLES_PROVENANCE.json").write_text(json.dumps(record, indent=2) + "\n")
print(output / "verified_evaluation_curves_cycles.png")
