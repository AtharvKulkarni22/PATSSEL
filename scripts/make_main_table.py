#!/usr/bin/env python3
"""Build the Qwen block of the paper's main results table from fresh experiment outputs."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

ROWS = [
    ("Base", "ReGen", "base_regen"),
    ("Base", "Debug", "base_debug"),
    ("Base", "PATSSEL", "base_patssel"),
    ("RL", "ReGen", "rl_regen"),
    ("RL", "Debug", "rl_debug"),
    ("RL", "PATSSEL", "rl_patssel"),
    ("Gold", "ReGen", "gold_regen"),
    ("Gold", "Debug", "gold_debug"),
    ("Gold", "PATSSEL", "gold_patssel"),
]
QCOLS = ["Q1", "Q2", "Q3", "Q4"]


def load_json(path: Path):
    with path.open() as f: return json.load(f)


def leetcode_metrics(root: Path):
    data = load_json(root / "all_selected_experiments_summary.json")
    out = {}
    for r in data:
        per = r["non_backtracking"]["per_subset"]
        vals = [float(x["overall_accuracy_pct"]) for x in per]
        if len(vals) != 4:
            raise ValueError(f"Expected four LeetCode buckets for {r['experiment']}, got {len(vals)}")
        out[r["experiment"]] = vals + [float(r["non_backtracking"]["global"]["overall_accuracy_pct"])]
    return out


def hefix_metrics(root: Path):
    data = load_json(root / "all_selected_experiments_summary.json")
    return {r["experiment"]: float(r["non_backtracking"]["overall_accuracy_pct"]) for r in data}


def lcb_metrics(root: Path):
    # Bucket file is written by run_livecodebench.py and uses the buggy-code pass-rate bins.
    with (root / "all_selected_experiments_summary_by_bucket.tsv").open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    bucket_name = {"Easy":"Q1", "Medium":"Q2", "Hard":"Q3", "Very Hard":"Q4"}
    tmp = {}
    for r in rows:
        if r["mode"] != "non_backtracking": continue
        tmp.setdefault(r["experiment"], {})[bucket_name[r["bucket"]]] = float(r["overall_test_accuracy_pct"])
    with (root / "all_selected_experiments_summary.tsv").open(newline="") as f:
        overall = list(csv.DictReader(f, delimiter="\t"))
    avgs = {r["experiment"]: float(r["overall_test_accuracy_pct"]) for r in overall if r["mode"] == "non_backtracking"}
    return {e: [tmp[e][q] for q in QCOLS] + [avgs[e]] for e in tmp}


def fmt(x): return f"{x:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leetcode_dir", type=Path, default=Path("outputs/results/leetcode"))
    ap.add_argument("--livecodebench_dir", type=Path, default=Path("outputs/results/livecodebench"))
    ap.add_argument("--hefix_dir", type=Path, default=Path("outputs/results/hefix"))
    ap.add_argument("--output_dir", type=Path, default=Path("outputs"))
    args = ap.parse_args()
    lc, lcb, hf = leetcode_metrics(args.leetcode_dir), lcb_metrics(args.livecodebench_dir), hefix_metrics(args.hefix_dir)
    missing = [(tag,e) for tag,src in [("LeetCode",lc),("LCB",lcb),("HE+Fix",hf)] for _,_,e in ROWS if e not in src]
    if missing: raise RuntimeError(f"Missing experiment results: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tsv = args.output_dir / "main_results_qwen.tsv"
    with tsv.open("w", newline="") as f:
        w=csv.writer(f, delimiter="\t")
        w.writerow(["Test Gen","Strategy","LeetCode Q1","LeetCode Q2","LeetCode Q3","LeetCode Q4","LeetCode Avg","LCB Q1","LCB Q2","LCB Q3","LCB Q4","LCB Avg","HE+Fix Avg"])
        for tg,strat,e in ROWS:
            w.writerow([tg,strat,*map(fmt,lc[e]),*map(fmt,lcb[e]),fmt(hf[e])])
    tex = args.output_dir / "main_results_qwen.tex"
    lines=[r"\begin{tabular}{ll|ccccc|ccccc|c}",r"\toprule",r"Test Gen & Strategy & \multicolumn{5}{c|}{LeetCode} & \multicolumn{5}{c|}{LiveCodeBench} & HE+Fix \\",r"& & Q1 & Q2 & Q3 & Q4 & Avg. & Q1 & Q2 & Q3 & Q4 & Avg. & Avg. \\",r"\midrule"]
    prev=None
    for tg,strat,e in ROWS:
        label=tg if tg!=prev else ""
        lines.append(" & ".join([label,strat,*map(fmt,lc[e]),*map(fmt,lcb[e]),fmt(hf[e])])+r" \\")
        prev=tg
        if strat=="PATSSEL" and tg!="Gold": lines.append(r"\midrule")
    lines += [r"\bottomrule",r"\end{tabular}"]
    tex.write_text("\n".join(lines)+"\n")
    print(tsv); print(tex)

if __name__ == "__main__": main()
