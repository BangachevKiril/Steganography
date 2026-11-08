"""Render the report figures from rounds.json (requires matplotlib).

Use --refresh only in the original experiment workspace to rebuild the aggregates.
"""
import argparse
import hashlib
import json
import sqlite3
import statistics
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
ROOT = DIR.parents[1]

RUNS = {
    "frozen": ("icrl_luna_flex_batch16_round50_20261107T051709Z", 24),
    "mini": ("icrl_adaptive_gpt41mini_batch16_round50_20261107T135800Z", 25),
    "symmetric": ("icrl_adaptive_luna_batch16_round20_20261107T135759Z", 10),
}


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(rows):
    return {role: sum(r[role] == r["label"] for r in rows) for role in ("receiver", "observer")}


def resources(run, end, mini):
    from experiments.icrl_university.private_memory_audit import costs

    # Use only receipts in the reported rounds and price with the experiment's own rate card.
    with sqlite3.connect(":memory:", uri=True) as db:
        db.execute("ATTACH DATABASE ? AS ledger", (f"file:{run}/private/openai_calls.sqlite3?mode=ro",))
        db.execute("CREATE TEMP TABLE calls AS SELECT * FROM ledger.calls WHERE round_idx BETWEEN 0 AND ?", (end,))
        db.execute("CREATE TEMP TABLE attempts AS SELECT a.* FROM ledger.attempts a JOIN calls c USING(id)")
        db.execute("CREATE INDEX temp.attempt_ids ON attempts(id,attempt)")
        rounds = db.execute("""
            SELECT c.round_idx, MAX(a.time) FROM calls c JOIN attempts a USING(id)
            WHERE a.state='complete' AND c.id LIKE 'r%.feedback.review.%'
            GROUP BY c.round_idx ORDER BY c.round_idx
        """).fetchall()
        assert [n for n,_ in rounds] == list(range(end+1))
        elapsed = [(b[1] - a[1])/60 for a,b in zip(rounds, rounds[1:])]
        assert min(elapsed) > 0
        def price():
            return costs(db, "flex", mini)["estimated_usd_completed_responses"]
        total = price()
        db.execute("DELETE FROM calls WHERE round_idx=0")
        return {"known_usd_through_cutoff": total, "mean_usd_per_revision": price()/end,
                "median_wall_minutes_per_revision": statistics.median(elapsed)}


def collect(name, run_name, end):
    from experiments.icrl_university.runner import digest

    run = ROOT / "results" / run_name
    rounds = []
    previous = None
    for n in range(end+1):
        filename = f"{n:04}.json"
        public_file = run / "public_rounds" / filename
        public = read(public_file)
        results_file = run / "private/round_results" / filename
        assert public["round"] == n and public["previous_commit_sha256"] == previous
        assert sha(results_file) == public["results_sha256"]
        assert sha(run / "private/snapshots" / filename) == public["snapshot_sha256"]
        previous = sha(public_file)
        result = read(results_file)
        row = {"round": n, "commit_sha256": previous}
        for split, key, size in (("heldout","heldout_metrics",100),("feedback","training_metrics",15)):
            assert len(result[split]) == size
            row[split] = counts(result[split])
            assert all(public[key][k]["correct"] == v for k,v in row[split].items())
        if n:
            b = run / "private/batches" / f"{n:04}"
            barrier_file = b / "PREDICTIONS_COMMITTED.json"
            barrier = read(barrier_file)
            assert sha(barrier_file) == read(b/"DONE.json")["barrier_sha256"]
            learn = [read(run / "private/evaluation" / f"r{n:03}.learn.{i:04}.json")["row"] for i in range(16)]
            assert [digest(v) for v in learn] == barrier["ordered_row_digests"]
            row["learn"] = counts(learn)
        rounds.append(row)
    mini = "gpt-4.1-mini-2025-04-14" if name == "mini" else None
    return {"run": run_name, "through": end, "denominators":{"heldout":100,"feedback":15,"learn":16},
            "rounds": rounds, "resources": resources(run, end, mini)}


def render(name, series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,
                         "axes.spines.right":False, "axes.spines.left":False})
    fig, axes = plt.subplots(1,3,figsize=(10.6,2.65),sharey=True)
    fig.subplots_adjust(left=.054,right=.987,top=.71,bottom=.21,wspace=.16)
    roles = {"receiver":("#006e72","o","-"), "observer":("#b44818","s","--")}
    for ax, (split,title) in zip(axes, (("heldout","Held-out · same 100 CVs"),
                                        ("learn","Learning batch · 16 CVs"),
                                        ("feedback","Training feedback · 15 CVs"))):
        sample = [v for v in series["rounds"] if split in v]
        ax.set_title(title,loc="left",fontsize=11.5,fontweight="bold",pad=7,color="#172b36")
        ax.set_axisbelow(True)
        ax.grid(axis="y",color="#e1e7e9",linewidth=.7)
        ax.axhline(.5,color="#85949a",linestyle=":",linewidth=1.2)
        for role,(color,marker,linestyle) in roles.items():
            ax.plot([v["round"] for v in sample],[v[split][role]/series["denominators"][split] for v in sample],
                    color=color,marker=marker,linestyle=linestyle,lw=1.8,ms=3.4,mfc=("white" if role=="observer" else color),
                    mew=.9,clip_on=False)
        end=series["through"]
        ax.set_xlim(-.42,end+.42)
        step=2 if end<=10 else 5
        ticks=list(range(0,end+1,step))
        if ticks[-1]!=end:ticks.append(end)
        ax.set_xticks(ticks);ax.set_xlabel("Round",labelpad=3)
        ax.set_ylim(-.025,1.035);ax.set_yticks([0,.25,.5,.75,1]);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.tick_params(length=0,labelsize=9,colors="#46565b",pad=4);ax.spines["bottom"].set_color("#b9c5c8")
    handles=[Line2D([],[],color=c,marker=m,ls=l,lw=2,ms=4,mfc=("white" if r=="observer" else c),label=r.title())
             for r,(c,m,l) in roles.items()]
    title = {"frozen":"Frozen observer","mini":"Adaptive GPT-4.1 mini observer","symmetric":"Adaptive Luna observer"}[name]
    fig.text(.054,.925,title + "  ·  Label-decoding accuracy",weight="bold",ha="left",va="center",fontsize=12)
    fig.legend(handles=handles,loc="center right",bbox_to_anchor=(.995,.922),ncol=2,frameon=False,fontsize=10)
    fig.savefig(DIR/f"{name}.png",dpi=180,facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="recollect from the local experiment ledgers")
    if parser.parse_args().refresh:
        sys.path.insert(0, str(ROOT))
        snapshot = {name:collect(name,run,end) for name,(run,end) in RUNS.items()}
        (DIR/"rounds.json").write_text(json.dumps(snapshot,indent=2,allow_nan=False)+"\n")
    else:
        snapshot = read(DIR/"rounds.json")
    for name,series in snapshot.items():
        render(name,series)
        print(name,series["through"],series["resources"])
