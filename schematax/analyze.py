"""Summarises journals in results/: score, parse rate, tokens and truncation per model, task and condition,
plus the paired difference to the natural-format baseline with a bootstrap 95% interval."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ORDER = ["natural", "json_free", "json_constrained", "json_blind",
         "json_constrained_answer_first", "json_constrained_answer_only"]


def load(path: Path) -> dict:
    rows = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[(r["task"], r["cond"])][r["id"]] = r
    return rows


def paired_ci(a: dict, b: dict, iters: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    ids = sorted(set(a) & set(b))
    diffs = [a[i]["score"] - b[i]["score"] for i in ids]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(iters))
    return sum(diffs) / len(diffs), means[int(0.025 * iters)], means[int(0.975 * iters)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("journals", nargs="*", default=sorted(str(x) for x in Path("results").glob("*.jsonl")))
    p.add_argument("--json", help="write the summary table here")
    args = p.parse_args()

    table = []
    for path in map(Path, args.journals):
        rows = load(path)
        model = path.stem
        for task in sorted({t for t, _ in rows}):
            base = rows.get((task, "natural"), {})
            print(f"\n{model} / {task}")
            print(f"  {'condition':32} {'n':>4} {'score':>6} {'parsed':>6} {'tokens':>6} {'trunc':>5}  vs natural [95% CI]")
            for cond in ORDER:
                cur = rows.get((task, cond))
                if not cur:
                    continue
                v = list(cur.values())
                n = len(v)
                score = sum(x["score"] for x in v) / n
                parsed = sum(x["ok"] for x in v) / n
                tokens = sum(x["tokens"] for x in v) / n
                trunc = sum(x["finish"] == "length" for x in v)
                d = ""
                rec = {"model": model, "task": task, "cond": cond, "n": n, "score": score,
                       "parsed": parsed, "tokens": tokens, "truncated": trunc}
                if cond != "natural" and base:
                    m, lo, hi = paired_ci(cur, base)
                    d = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                    rec.update(diff=m, lo=lo, hi=hi)
                table.append(rec)
                print(f"  {cond:32} {n:4} {score:6.3f} {parsed:6.2f} {tokens:6.0f} {trunc:5}  {d}")
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
