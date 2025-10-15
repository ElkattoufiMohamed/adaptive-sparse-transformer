#!/usr/bin/env python3
import json, re
from pathlib import Path

def find_summaries(root="results/experiments"):
    for p in Path(root).rglob("summary.json"):
        yield p

def parse_name(p: Path):
    # expects: results/experiments/{base}_{dataset}_{modeltype}/summary.json
    m = re.search(r"experiments/([^/]+)_([^/]+)_(adaptive|baseline)/summary\.json$", str(p))
    if m: return m.group(1), m.group(2), m.group(3)
    return "unknown", "unknown", "unknown"

def main():
    rows = []
    for p in find_summaries():
        base, dataset, modeltype = parse_name(p)
        with open(p) as f:
            js = json.load(f)
        metrics = js.get("final_eval_metrics", {})
        rows.append({
            "run_base": base,
            "dataset": dataset,
            "model_type": modeltype,
            "accuracy": metrics.get("eval_accuracy", 0),
            "f1": metrics.get("eval_f1", 0),
            "training_time_s": js.get("training_time", 0),
        })
    out = Path("results") / "comparisons" / "multidataset_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {out} with {len(rows)} rows")

if __name__ == "__main__":
    main()

