"""Screen clean RAG trajectories for multi-retrieval attack candidates."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUPPORTED_DATASETS = ("2wikimultihopqa", "hotpotqa")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS,
                        default="2wikimultihopqa")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--run-name", default="baseline_20")
    args = parser.parse_args()
    args.work_dir = args.work_dir or ROOT / "experiments" / args.dataset / "smoke_v1"
    if Path(args.run_name).name != args.run_name:
        raise ValueError("run-name must be a filename stem")

    data, results = args.work_dir / "data", args.work_dir / "results"
    supports = {row["id"]: set(row["support_ids"])
                for row in map(json.loads, (data / "qrels.jsonl").open(encoding="utf-8"))}
    source = results / f"{args.run_name}.jsonl"
    output = results / f"{args.run_name}.screening.jsonl"
    summary_path = results / f"{args.run_name}.screening.summary.json"
    if output.exists() or summary_path.exists():
        raise FileExistsError("screening output already exists")

    records = []
    for row in map(json.loads, source.open(encoding="utf-8")):
        state = row.get("state", {})
        retrievals = sum(step["action"] == "retrieve" for step in state.get("steps", []))
        seen = set(row.get("support_seen", []))
        reasons = []
        if row.get("error"):
            reasons.append("error")
        if not row.get("exact_match"):
            reasons.append("wrong_final_answer")
        if retrievals < 2:
            reasons.append("fewer_than_two_retrievals")
        if not supports[row["id"]] <= seen:
            reasons.append("missing_support_document")
        if state.get("stop_reason") != "model":
            reasons.append("not_model_stopped")
        records.append({
            "id": row["id"], "question": row["question"],
            "gold_answer": row["gold_answer"], "answer": row.get("answer"),
            "exact_match": bool(row.get("exact_match")),
            "retrieval_steps": retrievals, "support_seen": sorted(seen),
            "support_total": len(supports[row["id"]]),
            "stop_reason": state.get("stop_reason"),
            "eligible": not reasons, "exclusion_reasons": reasons,
        })

    with output.open("x", encoding="utf-8") as file:
        for row in records:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": str(source),
        "criteria": "exact match; >=2 retrievals; all support docs seen; model stopped; no error",
        "total": len(records),
        "correct": sum(row["exact_match"] for row in records),
        "at_least_two_retrievals": sum(row["retrieval_steps"] >= 2 for row in records),
        "automatic_candidates": [row["id"] for row in records if row["eligible"]],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(summary['automatic_candidates'])}/{len(records)} automatic candidates: {output}")


if __name__ == "__main__":
    main()
