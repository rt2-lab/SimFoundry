"""Success-rate stats for real xArm6 rollouts.

Rollout dirs (Data/cube2_scene/real_deploy/<YYYYMMDD_HHMMSS>/) hold args.json, log.jsonl and
cameras.mp4 but no outcome, so success is hand-labelled from the video in labels.csv
(columns: run,success[1/0],note). This script joins labels with args.json and prints rates.

  python scripts/xarm6/real_success_stats.py --date 20261020
"""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return (c - h) / d, (c + h) / d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Data/cube2_scene/real_deploy")
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    a = ap.parse_args()
    root = Path(a.root)
    labels = {r["run"]: r for r in csv.DictReader(open(root / f"labels_{a.date}.csv"))}
    by_color = defaultdict(list)
    for run, r in sorted(labels.items()):
        if r["success"] == "":
            print(f"unlabelled: {run}")
            continue
        prompt = json.load(open(root / run / "args.json"))["prompt"]
        color = prompt.split("the ")[1].split(" ")[0]
        by_color[color].append(int(r["success"]))
    allv = [v for vs in by_color.values() for v in vs]
    for c, vs in sorted(by_color.items()):
        print(f"{c:8s} {sum(vs)}/{len(vs)}")
    lo, hi = wilson(sum(allv), len(allv))
    print(f"TOTAL    {sum(allv)}/{len(allv)} = {sum(allv)/max(len(allv),1):.0%}  (95% Wilson CI {lo:.0%}-{hi:.0%})")


if __name__ == "__main__":
    main()
