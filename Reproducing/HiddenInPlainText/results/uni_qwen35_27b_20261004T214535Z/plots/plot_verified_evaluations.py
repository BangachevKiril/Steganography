"""Plot only public, completed evaluations bound to a verified checkpoint snapshot.

Usage: python3 plot_verified_evaluations.py SNAPSHOT_JSON LOCAL_SOURCE LOCAL_RESULTS
No remote access, private text output, or model/checkpoint loading is performed.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import tempfile

CONFIG_DIGEST = "42580b347ec09edac9f497c73c30f97f0c457a7955000873b81a8d0b5494e12d"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    raw = args.snapshot.read_bytes()
    snapshot = json.loads(raw)
    files = snapshot["files"]
    hashes = snapshot["files_sha256"]
    assert snapshot["state_sha256_verified"] is True
    assert snapshot["verified_update"] >= 0
    assert set(files) == set(hashes)
    for name, content in files.items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        assert sha(content.encode()) == hashes[name], name
    config = json.loads(files["config.json"])
    assert sha(json.dumps(config, sort_keys=True).encode()) == CONFIG_DIGEST
    source_hashes = config["source_sha256"]
    assert len(source_hashes) == 41
    for name, expected in source_hashes.items():
        assert sha((args.source / name).read_bytes()) == expected, name
    helper_name = "orcd/monitoring/snapshot_qwen_university.py"
    verifier = runpy.run_path(str(args.source / helper_name))
    run = Path(snapshot["checkpoint"]).parent.parent
    verified = verifier["verify_completed_evaluations"](
        files, run, CONFIG_DIGEST, snapshot["verified_update"]
    )
    assert verified == snapshot["verified_evaluations"] and verified
    highest = json.loads(files[f"checkpoints/step_{snapshot['verified_update']:06d}/COMPLETE.json"])
    assert highest["state_sha256"] == snapshot["state_sha256"]
    consumed = {"config.json"}
    records = []
    for identity in verified:
        update = identity["completed_update"]
        prefix = f"evaluations/step_{update:06d}/"
        consumed.update(name for name in files if name.startswith(prefix))
        consumed.add(f"checkpoints/step_{update:06d}/COMPLETE.json")
        summary = json.loads(files[prefix + "SUMMARY.json"])
        university = json.loads(files[prefix + "UNIVERSITY_RESULTS.json"])["rows"]
        piqa = json.loads(files[prefix + "PIQA_JUDGED.json"])["rows"]
        correct = sum(r["receiver_prediction"] == r["true_label"] for r in university)
        flagged = sum(r["observer_flag"] for r in university)
        passes = sum(r["receiver_prediction"] == 1 for r in university)
        fails = sum(r["receiver_prediction"] == 0 for r in university)
        judge = sum(not r["judge_pass"] for r in piqa)
        fmt = sum(not r["format_ok"] for r in piqa)
        overlap = sum(not r["judge_pass"] and not r["format_ok"] for r in piqa)
        rejected = sum(not r["judge_pass"] or not r["format_ok"] for r in piqa)
        assert all(math.isfinite(r["observer_score"]) for r in university)
        assert all(r["observer_flag"] == (r["observer_score"] < 0) for r in university)
        assert all(r["d"] == int(not r["judge_pass"] or not r["format_ok"]) for r in piqa)
        u, p = summary["university"], summary["piqa"]
        assert (len(university), len(piqa)) == (2000, 100)
        assert passes + fails == 2000
        assert u["counts"]["receiver_correct"] == correct
        assert u["counts"]["observer_flagged"] == flagged
        assert u["receiver_prediction_counts"] == {"PASS": passes, "FAIL": fails}
        assert p["counts"]["total_rejected_d"] == rejected == judge + fmt - overlap
        assert p["counts"]["judge_rejected"] == judge
        assert p["counts"]["format_rejected"] == fmt
        assert p["counts"]["overlap"] == overlap
        assert (u["C_R"], u["C_I"], p["d"]) == (correct / 2000, flagged / 2000, rejected / 100)
        records.append(dict(
            completed_update=update, university_n=2000, receiver_correct=correct,
            C_R=correct / 2000, observer_flagged=flagged, C_I=flagged / 2000,
            receiver_PASS=passes, receiver_FAIL=fails, piqa_n=100,
            judge_rejected=judge, format_rejected=fmt, overlap=overlap,
            union_rejected=rejected, d=rejected / 100,
            checkpoint_complete_sha256=identity["source_checkpoint"]["complete_sha256"],
            summary_sha256=identity["summary_sha256"],
        ))
    segment_ends = []
    for name in sorted(files):
        if not (name.startswith("SEGMENT_") and name.endswith("_COMPLETE.json")) or "BUDGET" in name:
            continue
        segment = json.loads(files[name])
        identity = next(v for v in verified if v["completed_update"] == segment["completed_update"])
        assert segment["config_digest"] == CONFIG_DIGEST
        assert segment["active_threshold_seconds"] == 27900 * segment["segment"]
        assert segment["total_active_budget_seconds"] == 83700
        assert segment["evaluation_complete_sha256"] == identity["complete_sha256"]
        assert segment["checkpoint"]["complete_sha256"] == identity["source_checkpoint"]["complete_sha256"]
        segment_ends.append({"segment": segment["segment"], "completed_update": segment["completed_update"],
                             "complete_sha256": hashes[name], "active_threshold_seconds": segment["active_threshold_seconds"]})
        consumed.add(name)
    final_completion = None
    if "COMPLETE.json" in files:
        complete = json.loads(files["COMPLETE.json"])
        budget = json.loads(files["BUDGET_COMPLETE.json"])
        final_identity = verified[-1]
        assert [s["segment"] for s in segment_ends] == [1, 2, 3]
        assert complete["config_digest"] == budget["config_digest"] == CONFIG_DIGEST
        assert complete["completed_update"] == budget["completed_update"] == snapshot["verified_update"] == final_identity["completed_update"]
        assert complete["segment"] == budget["segment"] == 3
        assert complete["active_threshold_seconds"] == budget["active_threshold_seconds"] == 83700
        assert complete["total_active_budget_seconds"] == budget["total_active_budget_seconds"] == 83700
        assert complete["evaluation_complete_sha256"] == final_identity["complete_sha256"]
        assert complete["checkpoint"] == snapshot["checkpoint"] == budget["checkpoint"]["path"]
        assert budget["checkpoint"]["complete_sha256"] == final_identity["source_checkpoint"]["complete_sha256"]
        assert budget["checkpoint"]["state_sha256"] == snapshot["state_sha256"]
        final_completion = {"completed_update": complete["completed_update"],
                            "complete_sha256": hashes["COMPLETE.json"],
                            "budget_complete_sha256": hashes["BUDGET_COMPLETE.json"],
                            "final_evaluation_complete_sha256": final_identity["complete_sha256"],
                            "final_state_sha256": snapshot["state_sha256"]}
        consumed.update(("COMPLETE.json", "BUDGET_COMPLETE.json"))
    for name in consumed:
        assert sha((args.results / name).read_bytes()) == hashes[name], name
    independently_verified_evaluation_states = []
    prior_state_snapshots = [
        (250, "monitoring/20261005T205249Z_snapshot.json",
         "c3f1f9ae04312fc2c500072255ca65f79ec3a803ce6a1f510afd9baa908b2c32",
         "8585d0c8cf393ecc5002fbd0d9b686d7056066fa8eed28a49afc860575e2cdb1"),
        (300, "monitoring/20261005T215426Z_snapshot.json",
         "8e070c83eadea9ae500595f0bccedcd5511fcd74128e405a893620fa77d26f3c",
         "02697f790ee9dee4bf8c894a2c93357d6b0006e3040759d09cb05eab653685ba"),
        (400, "monitoring/20261005T2348_checkpoint400_snapshot.json",
         "813d96d02251dcb50c494c59f02620ff77b278d1b8e07a556f46a27febb627b2",
         "2337758f549d5f461fca6f686e5aaff265c4b7a9854f99c26d8ae36070a27cca"),
        (450, "monitoring/20261006T003202Z_snapshot.json",
         "2b099a0a2ed974493a2cd0992e0f555c7c950cbc97fd1e11ed657c278207b2c7",
         "a3b05756663068de509caecc445241157fe8bd2bfa83d99858796b533849b7d8"),
        (1400, "monitoring/20261006T145548Z_snapshot.json",
         "64099edf6c9b02d7e59faf50fd7102db12be1286f79906ef40370d2ffefa19ad",
         "75c713489c8807e3f552e402a162a620e01609b500e3caf4f9b6f464de41ec4e"),
    ]
    for update, previous_name, previous_sha256, state_sha256 in prior_state_snapshots:
        if not any(identity["completed_update"] == update for identity in verified):
            continue
        previous_path = args.results / previous_name
        previous_bytes = previous_path.read_bytes()
        assert sha(previous_bytes) == previous_sha256
        previous = json.loads(previous_bytes)
        marker_name = f"checkpoints/step_{update:06d}/COMPLETE.json"
        marker = json.loads(files[marker_name])
        assert previous["state_sha256_verified"] is True
        assert previous["verified_update"] == marker["completed_update"] == update
        assert previous["files_sha256"][marker_name] == hashes[marker_name]
        assert previous["state_sha256"] == marker["state_sha256"] == state_sha256
        independently_verified_evaluation_states.append({
            "completed_update": update,
            "state_independently_rehashed_by_monitor": True,
            "state_sha256": previous["state_sha256"],
            "checkpoint_complete_sha256": hashes[marker_name],
            "remote_hash_snapshot_metadata_path": str(previous_path.resolve()),
            "remote_hash_snapshot_metadata_sha256": sha(previous_bytes),
            "qualification": f"The prior frozen snapshot helper independently hashed checkpoint {update} state.pt remotely before evaluation {update} finished. Its checkpoint COMPLETE hash matches the completed evaluation's source checkpoint in the current snapshot. No weights were copied locally.",
        })
    additional_state_qualifications = []
    if any(identity["completed_update"] in (350, 390) for identity in verified):
        remote_audit_path = args.results / "monitoring/MILESTONE_350_390_CHECKPOINT_REMOTE_AUDIT.json"
        remote_audit_bytes = remote_audit_path.read_bytes()
        assert sha(remote_audit_bytes) == "23c9ab5d51bdb988d89f40586c41f0877df763f2cecbfc38e97bc1f00e22126d"
        remote_audit = json.loads(remote_audit_bytes)
        for item in remote_audit["checkpoints"]:
            update = item["update"]
            if not any(identity["completed_update"] == update for identity in verified):
                continue
            marker_name = f"checkpoints/step_{update:06d}/COMPLETE.json"
            marker = json.loads(files[marker_name])
            assert item["marker_exists"] is True
            assert item["complete_sha256"] == hashes[marker_name]
            assert item["marker"] == marker
            evidence = {"completed_update": update, "checkpoint_complete_sha256": hashes[marker_name],
                        "remote_checkpoint_audit": str(remote_audit_path.resolve()),
                        "remote_checkpoint_audit_sha256": sha(remote_audit_bytes)}
            if update == 350:
                assert item["state_exists"] is False
                evidence.update(state_independently_rehashed_by_monitor=False,
                    qualification="The evaluation 350 checkpoint marker and its binding to the completed evaluation were verified. Its state.pt was already pruned when an independent rehash was attempted; other independently hashed checkpoints do not verify checkpoint 350 state bytes.")
                additional_state_qualifications.append(evidence)
            elif update == 390:
                assert item["state_exists"] is True and item["state_sha256_verified"] is True
                assert item["state_sha256"] == marker["state_sha256"] == "de8eba38536c759c71f65b645eb2bddf57fa5104bae507e50f38c3b6dc51ce0f"
                evidence.update(state_independently_rehashed_by_monitor=True, state_sha256=item["state_sha256"],
                    qualification="The segment 1 checkpoint 390 state.pt was independently hashed remotely. Its marker hash matches the completed evaluation and segment completion marker. No weights were copied locally.")
                independently_verified_evaluation_states.append(evidence)
    if any(identity["completed_update"] == 500 for identity in verified):
        remote_audit_path = args.results / "monitoring/EVALUATION_000500_CHECKPOINT_REMOTE_AUDIT.json"
        remote_audit_bytes = remote_audit_path.read_bytes()
        assert sha(remote_audit_bytes) == "571b6e55cc2fcece7cc766b0df32472ae52c74994b4ca18469c7fba5ca6ea536"
        remote_audit = json.loads(remote_audit_bytes)
        marker_name = "checkpoints/step_000500/COMPLETE.json"
        assert remote_audit["evaluation_update"] == 500
        assert remote_audit["marker_sha256"] == hashes[marker_name]
        assert remote_audit["marker"] == json.loads(files[marker_name])
        assert remote_audit["state_exists"] is False
        assert remote_audit["state_sha256_verified"] is False
        additional_state_qualifications.append({
            "completed_update": 500,
            "checkpoint_complete_sha256": hashes[marker_name],
            "state_independently_rehashed_by_monitor": False,
            "remote_checkpoint_audit": str(remote_audit_path.resolve()),
            "remote_checkpoint_audit_sha256": sha(remote_audit_bytes),
            "qualification": "The evaluation 500 checkpoint marker and its binding to the completed evaluation were verified. Its state.pt was already pruned when an independent rehash was attempted; the marker's attested state hash and other independently hashed checkpoints do not independently verify checkpoint 500 state bytes.",
        })
    catchup_updates = set(range(550, 1351, 50)) | {886}
    if any(identity["completed_update"] in catchup_updates for identity in verified):
        remote_audit_path = args.results / "monitoring/STAGE3_CATCHUP_CHECKPOINT_REMOTE_AUDIT.json"
        remote_audit_bytes = remote_audit_path.read_bytes()
        assert sha(remote_audit_bytes) == "28b201b2bd6034f9de6bb1fdcfbf9e80c763309fc1de1fc3aada987587142d68"
        remote_audit = json.loads(remote_audit_bytes)
        assert remote_audit["no_state_bytes_copied"] is True
        assert {item["completed_update"] for item in remote_audit["checkpoints"]} == catchup_updates
        for item in remote_audit["checkpoints"]:
            update = item["completed_update"]
            if not any(identity["completed_update"] == update for identity in verified):
                continue
            marker_name = f"checkpoints/step_{update:06d}/COMPLETE.json"
            marker = json.loads(files[marker_name])
            assert item["marker_sha256"] == hashes[marker_name]
            assert item["marker"] == marker
            evidence = {"completed_update": update, "checkpoint_complete_sha256": hashes[marker_name],
                        "remote_checkpoint_audit": str(remote_audit_path.resolve()),
                        "remote_checkpoint_audit_sha256": sha(remote_audit_bytes)}
            if update == 886:
                assert item["state_exists"] is True and item["state_sha256_verified"] is True
                assert item["state_sha256"] == marker["state_sha256"] == "257d1bc62202c044bdf332f5bb4c7e9b4898b4bc1d33fb98e77cad9a8703123a"
                evidence.update(state_independently_rehashed_by_monitor=True, state_sha256=item["state_sha256"],
                    qualification="The segment 2 checkpoint 886 state.pt was independently hashed remotely. Its marker hash matches the completed evaluation and segment completion marker. No weights were copied locally.")
                independently_verified_evaluation_states.append(evidence)
            else:
                assert item["state_exists"] is False and item["state_sha256_verified"] is False
                evidence.update(state_independently_rehashed_by_monitor=False,
                    qualification=f"The evaluation {update} checkpoint marker and its binding to the completed evaluation were verified. Its state.pt was already pruned when an independent rehash was attempted; the marker's attested state hash and other independently hashed checkpoints do not independently verify checkpoint {update} state bytes.")
                additional_state_qualifications.append(evidence)
    if final_completion:
        independently_verified_evaluation_states.append({
            "completed_update": snapshot["verified_update"],
            "state_independently_rehashed_by_monitor": True,
            "state_sha256": snapshot["state_sha256"],
            "checkpoint_complete_sha256": verified[-1]["source_checkpoint"]["complete_sha256"],
            "remote_hash_snapshot_path": str(args.snapshot.resolve()),
            "remote_hash_snapshot_sha256": sha(raw),
            "qualification": "The frozen snapshot helper independently hashed the final checkpoint state.pt remotely after the final evaluation completed. Its checkpoint COMPLETE hash matches the final evaluation and segment 3 completion marker. Global COMPLETE and BUDGET_COMPLETE bind the same checkpoint. No weights were copied locally.",
        })
    independently_verified_evaluation_states.sort(key=lambda item: item["completed_update"])
    output = args.results / "plots"
    output.mkdir(exist_ok=True)
    csv_path = output / "verified_evaluation_metrics.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="qwen_plot_"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 13, "axes.titleweight": "semibold",
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8))
    xs = [r["completed_update"] for r in records]
    single = len(records) == 1
    fig.suptitle("Qwen university rerun · verified evaluations", fontsize=18, weight="semibold", y=0.97)
    subtitle = ("Evaluation 0 only · baseline points; no trend can be inferred" if single
                else f"{len(records)} completed evaluations · through update {max(xs)}")
    if final_completion:
        subtitle = f"Run complete · {len(records)} verified evaluations · final update {max(xs):,}"
    if len(segment_ends) == 1:
        subtitle += " · dashed line: " + ", ".join(f"segment {s['segment']} end ({s['completed_update']})" for s in segment_ends)
    elif segment_ends:
        fig.text(0.5, 0.896, "Dashed lines: " + "; ".join(f"segment {s['segment']} end at {s['completed_update']:,}" for s in segment_ends),
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
            ax.axvline(segment["completed_update"], color="#9AA2AE", linestyle="--", linewidth=0.9, zorder=0)
        ax.set_title(title, loc="left", pad=27)
        ax.text(0, 1.015, description, transform=ax.transAxes, color="#596273", fontsize=10)
        ax.set_xlabel("Completed training updates")
        regular_xs = [x for x in xs if x % 50 == 0]
        tick_xs = regular_xs if len(regular_xs) < 12 else regular_xs[::math.ceil(len(regular_xs) / 9)]
        if tick_xs[-1] != regular_xs[-1] and regular_xs[-1] - tick_xs[-1] >= 0.06 * max(xs):
            tick_xs.append(regular_xs[-1])
        if tick_xs[-1] != max(xs):
            if max(xs) - tick_xs[-1] >= 0.06 * max(xs):
                tick_xs.append(max(xs))
            else:
                tick_xs[-1] = max(xs)
        ax.set_xticks(tick_xs)
        ax.set_xlim(-1, 1) if single else ax.margins(x=0.05)
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
             "This run’s evaluation 0 is its comparison baseline; the historical corrected baseline is a separate experiment.",
             fontsize=10, color="#4B5563", linespacing=1.6)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.83, bottom=0.15, hspace=0.70, wspace=0.28)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"verified_evaluation_curves.{suffix}", dpi=190)
    plt.close(fig)
    artifacts = ["verified_evaluation_curves.png", "verified_evaluation_curves.pdf",
                 "verified_evaluation_metrics.csv", "plot_verified_evaluations.py"]
    provenance = dict(
        schema_version=1, config_digest=CONFIG_DIGEST, source_files_verified=41,
        snapshot_sha256=sha(raw), snapshot_path=str(args.snapshot.resolve()),
        snapshot_verified_update=snapshot["verified_update"],
        snapshot_state_sha256=snapshot["state_sha256"],
        checkpoint_state_verification="Newest state.pt was hashed remotely by the frozen snapshot helper; weights were not copied or loaded locally.",
        evaluation_checkpoint_verification="Evaluation identity, checkpoint COMPLETE hash, summary COMPLETE hash and judge generation hash reverified from public bytes.",
        independently_verified_evaluation_states=independently_verified_evaluation_states,
        evaluation_state_verification_qualifications=[
            {
                "completed_update": 150,
                "checkpoint_complete_sha256": identity["source_checkpoint"]["complete_sha256"],
                "state_independently_rehashed_by_monitor": False,
                "milestone_snapshot_sha256": "2ee2de1676c755939cbc1c63b827421f86568329edcc21b2cdb0481d3ee6acc3",
                "corroborating_checkpoint_update": 152,
                "corroborating_checkpoint_state_sha256": "960bef28698d417173af34506aeab893aa05bd280f2208fd8be08962732a995e",
                "qualification": "The evaluation 150 checkpoint marker and its binding to the completed evaluation were verified. Its state.pt was already pruned when an independent rehash was attempted. The milestone snapshot independently hashed checkpoint 152; that does not independently verify checkpoint 150 state bytes.",
            }
            for identity in verified
            if identity["completed_update"] == 150
            and identity["source_checkpoint"]["complete_sha256"] == "c23dac1f3bcf748bded3b7fd7e0d91490c842fd628a0097c0865a61cb5838c7a"
        ] + [
            {
                "completed_update": 200,
                "checkpoint_complete_sha256": identity["source_checkpoint"]["complete_sha256"],
                "state_independently_rehashed_by_monitor": False,
                "milestone_snapshot_sha256": "39337e1ce5ce8220624f1a4ae383b49f49f475339383e3f18add64ee1054e328",
                "corroborating_checkpoint_update": 223,
                "corroborating_checkpoint_state_sha256": "12a87163db3c4d269e104c54d54a028074f06e75497824b7502cb83aeba1ddcd",
                "remote_checkpoint_audit": "monitoring/EVALUATION_000200_CHECKPOINT_REMOTE_AUDIT.json",
                "qualification": "The evaluation 200 checkpoint marker and its binding to the completed evaluation were verified. Its state.pt was already pruned when an independent rehash was attempted. The milestone snapshot independently hashed checkpoint 223; that does not independently verify checkpoint 200 state bytes.",
            }
            for identity in verified
            if identity["completed_update"] == 200
            and identity["source_checkpoint"]["complete_sha256"] == "52425704876c96d0461962e972ea81c5117a4b6f191ec3075cbff2aa70349bf4"
        ] + additional_state_qualifications,
        verifier_sha256=source_hashes[helper_name], verified_evaluations=verified,
        verified_segment_ends=segment_ends,
        final_completion=final_completion,
        completed_evaluations_plotted=xs, raw_row_counts_recomputed=True,
        baseline_only=single and xs == [0], trend_or_extrapolation=False,
        metric_definitions={"C_R": "Receiver correct/2000", "C_I": "Operational observer flagged/2000; not transmission accuracy",
                            "d": "PIQA judge rejection OR format failure / 100", "PASS": "Receiver PASS predictions / 2000 (class count)"},
        public_source_file_hashes={name: hashes[name] for name in sorted(consumed)},
        artifact_sha256={name: sha((output / name).read_bytes()) for name in artifacts},
    )
    (output / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"evaluations": xs, "verified_update_cap": snapshot["verified_update"],
                      "artifacts": artifacts + ["PROVENANCE.json"], "records": records}, indent=2))


if __name__ == "__main__":
    main()
