"""Plot verified frozen evaluation rates; never load training/model state."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="uni-curves-mpl-"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(run, baseline):
    done = read(run / "COMPLETE.json")
    baseline_done = read(baseline / "COMPLETE.json")
    assert sha(baseline / "SUMMARY.json") == baseline_done["summary_sha256"]
    remote = read(run / "diagnostics/completion_remote_hash_verification.json")
    verified = {r["update"]: r for r in remote["checkpoints"]}
    samples = [(0, 0.0, baseline / "SUMMARY.json", "pre-training baseline")]
    for path in sorted((run / "evaluations").glob("step_*/SUMMARY.json")):
        complete = read(path.parent / "COMPLETE.json")
        update = complete["completed_update"]
        assert update <= done["completed_update"]
        assert complete["config_digest"] == done["config_digest"]
        assert sha(path) == complete["summary_sha256"]
        cp_path = run / "checkpoints" / f"step_{update:06d}" / "COMPLETE.json"
        cp = read(cp_path)
        assert sha(cp_path) == complete["source_checkpoint"]["complete_sha256"]
        assert cp["state_sha256"] == verified[update]["state_sha256"]
        if update == done["completed_update"]:
            assert sha(path.parent / "COMPLETE.json") == done["evaluation_complete_sha256"]
        samples.append((update, cp["elapsed"], path, "frozen evaluation"))
    rows = []
    for update, elapsed, path, kind in samples:
        summary = read(path)
        uni, piqa = summary["university"], summary["piqa"]
        assert uni["n"] == 2000 and piqa["n"] == 100
        correct = uni["counts"]["receiver_correct"]
        flagged = uni["counts"]["observer_flagged"]
        rejected = piqa["counts"]["total_rejected_d"]
        predictions = uni["receiver_prediction_counts"]
        assert predictions["PASS"] + predictions["FAIL"] == uni["n"]
        assert correct / uni["n"] == uni["C_R"]
        assert flagged / uni["n"] == uni["C_I"]
        assert rejected / piqa["n"] == piqa["d"]
        rows.append(dict(
            update=update, measurement=kind, checkpoint_active_seconds=elapsed,
            checkpoint_active_hours=elapsed / 3600,
            university_n=uni["n"], receiver_correct=correct, observer_flagged=flagged,
            receiver_pass=predictions["PASS"], receiver_fail=predictions["FAIL"],
            true_pass=uni["true_label_counts"]["PASS"],
            C_R_percent=100 * uni["C_R"], C_I_percent=100 * uni["C_I"],
            piqa_n=piqa["n"], piqa_rejected=rejected, d_percent=100 * piqa["d"],
            summary_path=str(path.resolve()), summary_sha256=sha(path),
        ))
    events = [read(p) for p in (run / "runtime_events").glob("l40s_*.json")]
    migrations = [e for e in events if e.get("kind") == "L40S_continuation"
                  and e.get("status") == "restored"]
    first = min(migrations, key=lambda e: e["completed_update"])
    return rows, first["completed_update"], first["restored_elapsed_seconds"] / 3600


def draw(rows, output, axis, migration_update, migration_hours):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#BCC6CE", "axes.linewidth": 0.7,
        "text.color": "#172D3F", "axes.labelcolor": "#34495A",
        "xtick.color": "#536674", "ytick.color": "#536674",
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axs = plt.subplots(3, 1, figsize=(9.4, 9.0), sharex=True)
    fig.subplots_adjust(left=.105, right=.91, bottom=.17, top=.855, hspace=.46)
    fig.suptitle("University run · frozen evaluation curves", x=.105, y=.965,
                 ha="left", fontsize=20, fontweight="bold")
    fig.text(.105, .919, "Same 2,000 university examples and 100 PIQA examples at every evaluation",
             fontsize=11, color="#526675")
    xkey = "checkpoint_active_hours" if axis == "hours" else "update"
    x = [row[xkey] for row in rows]
    migration = migration_hours if axis == "hours" else migration_update
    specs = [
        ("C_R_percent", r"$C_R$  Receiver accuracy", "#236AAB", (45, 55), [45, 47.5, 50, 52.5, 55], 50, "chance 50%"),
        ("C_I_percent", r"$C_I$  Static-observer flag fraction", "#B66B16", (0, 15), [0, 5, 10, 15], rows[0]["C_I_percent"], "baseline 4.4%"),
        ("d_percent", r"$d$  PIQA rejection fraction", "#8053A3", (0, 12), [0, 4, 8, 12], rows[0]["d_percent"], "baseline 8%"),
    ]
    pct = FuncFormatter(lambda v, _: f"{v:g}%")
    for ax, (key, title, color, limits, ticks, reference, label) in zip(axs, specs):
        y = [row[key] for row in rows]
        ax.set_facecolor("white")
        ax.axvspan(0, migration, color="#EEF1F4", zorder=0)
        ax.axvline(migration, color="#B5BEC7", lw=.9, ls=(0, (2, 3)), zorder=1)
        ax.grid(axis="y", color="#E5EAEE", linewidth=.65, zorder=0)
        ax.axhline(reference, color="#8F9DA7", lw=1, ls=(0, (4, 4)), zorder=1)
        ax.plot(x, y, color=color, lw=2.2, marker="o", ms=5.5,
                markeredgecolor="white", markeredgewidth=.8, zorder=3)
        ax.scatter([x[0]], [y[0]], s=60, facecolor="white", edgecolor=color,
                   linewidth=1.8, zorder=4)
        ax.text(0, 1.07, title, transform=ax.transAxes, fontsize=12.5, fontweight="bold")
        ax.text(1, 1.07, "axis zoomed: 45–55%" if key == "C_R_percent" else label,
                transform=ax.transAxes, ha="right", fontsize=9.5, color="#677985")
        if key == "C_R_percent":
            ax.text(.988, (reference - limits[0]) / (limits[1] - limits[0]) - .05,
                    label, transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color="#73828C")
        ax.annotate(f"{y[-1]:.2f}%".replace(".00%", "%"),
                    xy=(x[-1], y[-1]), xytext=(10, 0), textcoords="offset points",
                    color=color, va="center", fontsize=11, fontweight="bold")
        ax.set_ylim(*limits)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(pct)
        ax.tick_params(axis="both", labelsize=10, length=3)
        ax.set_xlim(-x[-1] * .013, x[-1] * 1.08)
    if axis == "hours":
        axs[-1].set_xticks(range(9))
        axs[-1].set_xlabel("Recorded active time at checkpoint (hours)", labelpad=9)
    else:
        axs[-1].set_xticks([0, 100, 200, 300, 400, 500, 532])
        axs[-1].set_xlabel("Combined training updates (receiver + sender)", labelpad=9)
    fig.text(.105, .096, "Open marker: pre-training baseline. Filled markers: 11 frozen evaluations. No smoothing.", fontsize=9, color="#536674")
    fig.text(.105, .071, "Shaded: A100 through update 82; L40S thereafter. C_I is flagging, not transmission accuracy.", fontsize=9, color="#536674")
    note = ("Active time uses the checkpoint clock before evaluation; queue/restart downtime is excluded."
            if axis == "hours" else "Each complete cycle contains 2 receiver updates followed by 5 sender PPO updates.")
    fig.text(.105, .046, note, fontsize=9, color="#536674")
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(output / f"frozen_evaluation_curves_{axis}.{extension}", dpi=180,
                    facecolor="white", metadata={"Creator": "Verified university evaluation plot"})
    plt.close(fig)


def draw_pass(rows, output, axis, migration_update, migration_hours):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.subplots_adjust(left=.10, right=.94, bottom=.24, top=.76)
    fig.suptitle("Receiver PASS predictions out of 2,000", x=.10, y=.95,
                 ha="left", fontsize=20, fontweight="bold")
    fig.text(.10, .875, "Same 2,000 university examples at every frozen evaluation",
             fontsize=11, color="#526675")
    xkey = "checkpoint_active_hours" if axis == "hours" else "update"
    x = [row[xkey] for row in rows]
    y = [row["receiver_pass"] for row in rows]
    migration = migration_hours if axis == "hours" else migration_update
    color = "#167D87"
    ax.axvspan(0, migration, color="#EEF1F4", zorder=0)
    ax.axvline(migration, color="#B5BEC7", lw=.9, ls=(0, (2, 3)))
    ax.grid(axis="y", color="#E5EAEE", linewidth=.65)
    assert len({row["true_pass"] for row in rows}) == 1
    true_pass = rows[0]["true_pass"]
    ax.axhline(true_pass, color="#8F9DA7", lw=1, ls=(0, (4, 4)))
    ax.text(x[-1] * 1.05, true_pass + 45, f"True PASS labels: {true_pass:,}",
            ha="right", va="bottom", fontsize=9, color="#677985")
    ax.plot(x, y, color=color, lw=2.2, marker="o", ms=6,
            markeredgecolor="white", markeredgewidth=.8, zorder=3)
    ax.scatter([x[0]], [y[0]], s=65, facecolor="white", edgecolor=color,
               linewidth=1.8, zorder=4)
    for index, (xx, yy) in enumerate(zip(x, y)):
        # Separate the baseline and first evaluation, which are close in time.
        offset = -19 if index == 1 else 10
        ax.annotate(f"{yy:,}", (xx, yy), xytext=(0, offset),
                    textcoords="offset points", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=.5))
    ax.set_ylim(0, 2200)
    ax.set_yticks([0, 500, 1000, 1500, 2000])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylabel("Number predicted PASS")
    ax.set_xlim(-x[-1] * .035, x[-1] * 1.07)
    if axis == "hours":
        ax.set_xticks(range(9))
        ax.set_xlabel("Recorded active time at checkpoint (hours)", labelpad=9)
    else:
        ax.set_xticks([0, 100, 200, 300, 400, 500, 532])
        ax.set_xlabel("Combined training updates (receiver + sender)", labelpad=9)
    fig.text(.10, .115, "Open marker: baseline. Shaded: A100 through update 82; L40S thereafter. No smoothing.",
             fontsize=9, color="#536674")
    fig.text(.10, .070, "PASS is the receiver’s predicted class. Accuracy depends on matching the true labels.",
             fontsize=9, color="#536674")
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(output / f"receiver_pass_counts_{axis}.{extension}", dpi=180,
                    facecolor="white", metadata={"Creator": "Verified university evaluation plot"})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("baseline_dir", type=Path)
    args = parser.parse_args()
    rows, migration_update, migration_hours = collect(args.run_dir, args.baseline_dir)
    output = args.run_dir / "plots"
    output.mkdir(exist_ok=True)
    with (output / "frozen_evaluation_curves.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for axis in ["hours", "updates"]:
        draw(rows, output, axis, migration_update, migration_hours)
        draw_pass(rows, output, axis, migration_update, migration_hours)
    (output / "PROVENANCE.json").write_text(json.dumps({
        "source": "Verified baseline and frozen evaluation summaries only",
        "points": len(rows), "last_update": rows[-1]["update"],
        "script_sha256": sha(Path(__file__)),
        "data_sha256": sha(output / "frozen_evaluation_curves.csv"),
        "C_R": "Receiver label accuracy on fixed 2000 university examples",
        "C_I": "Operational static-observer flag fraction, not transmission accuracy",
        "d": "Judge rejection OR format failure on fixed 100 PIQA examples",
        "receiver_pass": "Number predicted PASS by the receiver on fixed 2000 university examples; a class count, not a correctness count",
        "active_time": "Checkpoint elapsed before the corresponding evaluation; includes prior retained evaluation/setup overhead, excludes queue/restart downtime and work lost after a checkpoint. The separately measured pre-training baseline is displayed at zero.",
        "migration_update": migration_update, "migration_active_hours": migration_hours,
        "smoothing": False, "uncertainty": "Descriptive rates for one run on fixed cohorts; no independent repeats represented.",
    }, indent=2) + "\n")
    print(output.resolve())


if __name__ == "__main__":
    main()
