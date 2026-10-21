"""Reproduce corrected-run figures from verified public evaluation artifacts only.

Usage: python3 /path/to/plots/plot_corrected_curves.py
Writes only beside this script; no model weights, training source or run data change.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="corrected-curves-mpl-"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUTPUT = Path(__file__).resolve().parent
RUN = OUTPUT.parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    report = read(RUN / "REPORT.json")
    done = read(RUN / "COMPLETE.json")
    expected_updates = [0, *range(50, 501, 50), 530]
    assert report["completed_update"] == done["completed_update"] == 530
    assert report["evaluation_updates"] == expected_updates
    assert report["corrected_baseline_measured"] is True
    assert report["baseline_d"] == .08
    rows = []
    source_hashes = {"REPORT.json": sha(RUN / "REPORT.json"),
                     "COMPLETE.json": sha(RUN / "COMPLETE.json")}
    for reported in report["evaluations"]:
        update = reported["update"]
        directory = RUN / "evaluations" / f"step_{update:06d}"
        summary = read(directory / "SUMMARY.json")
        marker = read(directory / "COMPLETE.json")
        cp_path = RUN / "checkpoints" / f"step_{update:06d}" / "COMPLETE.json"
        checkpoint = read(cp_path)
        assert marker["status"] == "complete"
        assert summary["completed_update"] == marker["completed_update"] == checkpoint["completed_update"] == update
        assert summary["config_digest"] == marker["config_digest"] == checkpoint["config_digest"] == done["config_digest"]
        assert sha(directory / "SUMMARY.json") == marker["summary_sha256"]
        assert summary["source_checkpoint"] == marker["source_checkpoint"]
        assert sha(cp_path) == marker["source_checkpoint"]["complete_sha256"]
        assert summary["baseline_d"] == report["baseline_d"]
        if update == 530:
            assert sha(directory / "COMPLETE.json") == done["evaluation_complete_sha256"]
            identity = read(RUN / "diagnostics/evaluation530_remote_runtime_identity.json")
            assert identity["checkpoint_state_sha256_verified_remotely"] is True
            assert identity["checkpoint"]["state_sha256"] == checkpoint["state_sha256"]
            assert identity["checkpoint_manifest_sha256"] == sha(cp_path)
        uni, piqa = summary["university"], summary["piqa"]
        uc, pc = uni["counts"], piqa["counts"]
        predicted = uni["receiver_prediction_counts"]
        assert uni["n"] == reported["university"]["n"] == 2000
        assert piqa["n"] == reported["piqa"]["n"] == 100
        assert predicted["PASS"] + predicted["FAIL"] == 2000
        assert uni["true_label_counts"] == {"PASS": 1001, "FAIL": 999}
        assert uc["receiver_correct"] == reported["university"]["receiver_correct"]
        assert uc["observer_flagged"] == reported["university"]["observer_flagged"]
        assert predicted["PASS"] == reported["university"]["receiver_pass"]
        assert predicted["FAIL"] == reported["university"]["receiver_fail"]
        assert pc["total_rejected_d"] == reported["piqa"]["total_rejected_d"]
        assert pc["total_rejected_d"] == pc["judge_rejected"] + pc["format_rejected"] - pc["overlap"]
        assert uc["receiver_correct"] / 2000 == uni["C_R"] == reported["university"]["C_R"]
        assert uc["observer_flagged"] / 2000 == uni["C_I"] == reported["university"]["C_I"]
        assert pc["total_rejected_d"] / 100 == piqa["d"] == reported["piqa"]["d"]
        for path in (directory / "SUMMARY.json", directory / "COMPLETE.json", cp_path):
            source_hashes[str(path.relative_to(RUN))] = sha(path)
        rows.append(dict(
            update=update, checkpoint_active_seconds=checkpoint["elapsed"],
            checkpoint_active_hours=checkpoint["elapsed"] / 3600,
            university_n=2000, receiver_correct=uc["receiver_correct"],
            observer_flagged=uc["observer_flagged"], receiver_pass=predicted["PASS"],
            receiver_fail=predicted["FAIL"], true_pass=1001, true_fail=999,
            C_R_percent=100*uni["C_R"], C_I_percent=100*uni["C_I"],
            piqa_n=100, piqa_judge_rejected=pc["judge_rejected"],
            piqa_format_rejected=pc["format_rejected"], piqa_overlap=pc["overlap"],
            piqa_union_rejected=pc["total_rejected_d"], d_percent=100*piqa["d"],
            delta_d_percentage_points=100*(piqa["d"]-report["baseline_d"]),
            summary_sha256=sha(directory / "SUMMARY.json"),
            checkpoint_manifest_sha256=sha(cp_path),
        ))
    assert [row["update"] for row in rows] == expected_updates
    assert all(a["checkpoint_active_seconds"] < b["checkpoint_active_seconds"]
               for a, b in zip(rows, rows[1:]))
    return rows, source_hashes


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#AFBCC7", "axes.linewidth": .7,
        "text.color": "#1D3041", "axes.labelcolor": "#405363",
        "xtick.color": "#5B6C78", "ytick.color": "#5B6C78",
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def plot_line(ax, x, y, color, label=None, marker="o", linestyle="-"):
    ax.plot(x, y, color=color, linewidth=2.15, marker=marker, markersize=5.7,
            markeredgecolor="white", markeredgewidth=.75, label=label,
            linestyle=linestyle, zorder=3)
    ax.scatter([x[0]], [y[0]], s=62, marker=marker, facecolor="white",
               edgecolor=color, linewidth=1.65, zorder=4)
    ax.scatter([x[-1]], [y[-1]], s=75, marker="D", facecolor=color,
               edgecolor="white", linewidth=.9, zorder=5)


def draw(rows, axis):
    style()
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.0))
    fig.subplots_adjust(left=.08, right=.97, bottom=.19, top=.815, wspace=.28, hspace=.61)
    fig.suptitle("Corrected university run · frozen evaluations", x=.08, y=.961,
                 ha="left", fontsize=22, fontweight="bold")
    fig.text(.08, .912, "12 measured checkpoints, including the corrected baseline · Final update 530 · L40S",
             fontsize=11.5, color="#526674")
    xkey = "checkpoint_active_hours" if axis == "hours" else "update"
    x = [row[xkey] for row in rows]
    specs = [
        ("C_R_percent", r"A   $C_R$  Receiver accuracy", "#226DA9", (45, 60), [45, 50, 55, 60], 50,
         "2,000 university examples · axis zoomed", f"Final: {rows[-1]['receiver_correct']:,}/2,000 = {rows[-1]['C_R_percent']:.2f}%"),
        ("C_I_percent", r"B   $C_I$  Static-observer flags", "#B7751C", (0, 16), [0, 4, 8, 12, 16], rows[0]["C_I_percent"],
         "2,000 university examples · operational flag rate", f"Final: {rows[-1]['observer_flagged']}/2,000 = {rows[-1]['C_I_percent']:.2f}%"),
        ("d_percent", r"C   $d$  PIQA union rejection", "#8050A2", (0, 10), [0, 2, 4, 6, 8, 10], rows[0]["d_percent"],
         "100 PIQA examples · judge rejection OR format failure", f"Final: {rows[-1]['piqa_union_rejected']}/100 = {rows[-1]['d_percent']:.0f}%"),
    ]
    pct = FuncFormatter(lambda value, _: f"{value:g}%")
    for ax, spec in zip(axes.flat, specs):
        key, title, color, limits, ticks, reference, note, final_note = spec
        ax.set_title(title, loc="left", fontsize=13.5, fontweight="bold", pad=30)
        ax.text(0, 1.055, note, transform=ax.transAxes, fontsize=9.3, color="#617482")
        ax.axhline(reference, color="#8F9FA9", linewidth=.9, linestyle=(0, (4, 4)), zorder=1)
        plot_line(ax, x, [row[key] for row in rows], color)
        ax.set_ylim(*limits)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(pct)
        ax.text(.98, .055 if key == "C_R_percent" else .93, final_note,
                transform=ax.transAxes, ha="right", va="bottom" if key == "C_R_percent" else "top",
                fontsize=10.1, color=color, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=2))
    ax = axes[1, 1]
    ax.set_title("D   Receiver class predictions", loc="left", fontsize=13.5,
                 fontweight="bold", pad=30)
    ax.text(0, 1.055, "Out of 2,000 university examples · PASS + FAIL = 2,000",
            transform=ax.transAxes, fontsize=9.3, color="#617482")
    ax.axhline(1000, color="#8F9FA9", linewidth=.9, linestyle=(0, (4, 4)), zorder=1)
    plot_line(ax, x, [row["receiver_pass"] for row in rows], "#087F83", label="PASS", marker="o")
    plot_line(ax, x, [row["receiver_fail"] for row in rows], "#5E6878", label="FAIL", marker="s", linestyle="--")
    ax.set_ylim(-70, 2170)
    ax.set_yticks([0, 500, 1000, 1500, 2000])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none",
              ncol=2, fontsize=9.4, handlelength=2.4, columnspacing=1.4)
    ax.text(.02, .12, f"Final: {rows[-1]['receiver_pass']:,} PASS · {rows[-1]['receiver_fail']:,} FAIL",
            transform=ax.transAxes, ha="left", fontsize=10.1, color="#374F5E", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=2))
    for ax in axes.flat:
        ax.grid(axis="y", color="#E6EBEF", linewidth=.7, zorder=0)
        ax.axvline(x[-1], color="#BCC7CE", linewidth=.9, linestyle=(0, (2, 3)), zorder=1)
        ax.set_xlim(-.025*x[-1], 1.045*x[-1])
        ax.tick_params(length=3, labelsize=10)
        if axis == "hours":
            ax.set_xticks(range(9))
            ax.set_xlabel("Recorded active time at checkpoint (hours)", labelpad=8, fontsize=10.5)
        else:
            ax.set_xticks([0, 100, 200, 300, 400, 500])
            ax.set_xlabel("Combined training updates (receiver + sender)", labelpad=8, fontsize=10.5)
    fig.text(.08, .108, "Open markers: corrected checkpoint-zero baseline. Diamonds and vertical dotted line: final update 530. No smoothing.",
             fontsize=9.5, color="#4C6373")
    fig.text(.08, .078, "Dashed references: 50% accuracy, 7.5% observer flags, 8% PIQA rejection, and 1,000 class predictions.",
             fontsize=9.5, color="#4C6373")
    fig.text(.08, .048, "C_I measures observer flagging, not information-transmission accuracy. True university labels: 1,001 PASS and 999 FAIL.",
             fontsize=9.5, color="#4C6373")
    if axis == "hours":
        fig.text(.08, .022, "Clock is captured before each evaluation; includes prior retained evaluations/setup, excludes queue/restart downtime and lost work.",
                 fontsize=8.8, color="#687C8A")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT / f"corrected_evaluation_overview_{axis}.{extension}",
                    dpi=180, facecolor="white", metadata={"Creator": "Corrected university evaluation curves"})
    plt.close(fig)


def main():
    rows, source_hashes = collect()
    with (OUTPUT / "corrected_evaluation_curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for axis in ("updates", "hours"):
        draw(rows, axis)
    (OUTPUT / "PROVENANCE.json").write_text(json.dumps({
        "run": RUN.name, "source": "Verified public REPORT.json, evaluation summaries and checkpoint manifests",
        "points": len(rows), "evaluation_updates": [row["update"] for row in rows],
        "final_update": 530, "corrected_baseline_d": .08,
        "script_sha256": sha(Path(__file__)), "csv_sha256": sha(OUTPUT / "corrected_evaluation_curves.csv"),
        "source_sha256": source_hashes,
        "time_semantics": "Actual saved checkpoint elapsed before its corresponding evaluation, including checkpoint0's initialization time; not evaluation completion walltime. Includes prior retained setup/evaluation overhead, excludes queue/restart downtime and work lost after a checkpoint.",
        "C_R": "Receiver argmax label accuracy on fixed 2000 university examples; axis45–60% explicitly zoomed",
        "C_I": "Corrected last-real-token static observer operational flag fraction; not information-transmission accuracy",
        "d": "Judge rejection OR format failure on fixed100 PIQA examples; overlap counted once",
        "receiver_counts": "Predicted PASS and FAIL counts sum to2000; true labels are1001 PASS and999 FAIL",
        "smoothing": False, "uncertainty": "Descriptive one-run rates on fixed cohorts; no uncertainty bands or significance claim.",
        "old_observer_comparison": False,
    }, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "points": len(rows), "final": rows[-1]}, indent=2))


if __name__ == "__main__":
    main()
