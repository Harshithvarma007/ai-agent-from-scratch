"""Aggregate raw per-run JSONL from bench_native_vs_react.py / comparisons/langchain/compare.py
into a pass-rate table. Prints raw counts first: report raw numbers, then interpret.

Usage:
    python3 bench/analyze_bench.py bench/results/data_analyst_bench.jsonl [group_keys...]
    python3 bench/analyze_bench.py comparisons/langchain/results/langchain_bench.jsonl model agent_type
"""
import json
import sys
from collections import defaultdict


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    path = sys.argv[1]
    group_keys = sys.argv[2:] or ["model", "mode"]
    records = load(path)

    groups = defaultdict(list)
    for r in records:
        key = tuple(r.get(k) for k in group_keys)
        groups[key].append(r)

    header = list(group_keys) + ["n", "correct", "wrong", "no_answer",
                                  "confident_wrong", "pass_rate", "avg_s"]
    widths = [max(len(h), 12) for h in header]
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-" * (sum(widths) + 3 * len(widths)))

    for key in sorted(groups):
        rs = groups[key]
        n = len(rs)
        correct = sum(1 for r in rs if r["verdict"] == "correct")
        wrong = sum(1 for r in rs if r["verdict"] == "wrong")
        no_answer = sum(1 for r in rs if r["verdict"] == "no_answer")
        confident_wrong = sum(1 for r in rs if r["verdict"] == "wrong" and r.get("confident_wrong"))
        avg_s = sum(r["elapsed_s"] for r in rs) / n
        row = list(key) + [n, correct, wrong, no_answer, confident_wrong,
                            f"{correct / n:.0%}", f"{avg_s:.1f}"]
        print(" | ".join(str(c).ljust(w) for c, w in zip(row, widths)))

    print(f"\ntotal records: {len(records)}")
    if "question_id" not in group_keys:
        print("\n--- per question_id (all groups combined) ---")
        qgroups = defaultdict(list)
        for r in records:
            qgroups[r["question_id"]].append(r)
        for qid in sorted(qgroups):
            rs = qgroups[qid]
            n = len(rs)
            correct = sum(1 for r in rs if r["verdict"] == "correct")
            confident_wrong = sum(1 for r in rs if r["verdict"] == "wrong" and r.get("confident_wrong"))
            print(f"  {qid:10} n={n:3} correct={correct:3} confident_wrong={confident_wrong:3} "
                  f"pass_rate={correct / n:.0%}")


if __name__ == "__main__":
    main()
