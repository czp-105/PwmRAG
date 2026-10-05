"""Run a clean ComoRAG baseline on an existing experiment index."""

import argparse
import json
import re
import string
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from threading import Lock

from run_rag_smoke import DEFAULT_DATASET, ROOT, SUPPORTED_DATASETS, read_jsonl, sha256
from rag.config import COMORAG_MODE
from rag.parallel import ordered_parallel_map


def default_work_dir(dataset: str) -> Path:
    return ROOT / "experiments" / dataset / "expanded_100_full_v1"


def normalize_answer(answer: str) -> str:
    """Match the answer normalization used by DeepRAG and ComoRAG evaluators."""
    unpunctuated = answer.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", unpunctuated).split())


def f1_score(prediction: str, gold: str) -> float:
    predicted_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    common = sum((Counter(predicted_tokens) & Counter(gold_tokens)).values())
    if not common:
        return 0.0
    precision = common / len(predicted_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--model", "--model-path", dest="model", required=True)
    parser.add_argument("--backend", choices=("transformers", "vllm"), default="transformers")
    parser.add_argument("--model-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--question-id")
    parser.add_argument("--max-questions", type=int, default=1)
    parser.add_argument("--run-name", default="qwen3_8b_smoke")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--min-probe-iterations", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--check-only", action="store_true", help="Validate inputs without loading models")
    args = parser.parse_args()
    args.work_dir = args.work_dir or default_work_dir(args.dataset)

    if Path(args.run_name).name != args.run_name or args.run_name in ("", ".", ".."):
        parser.error("run-name must be a filename stem")
    workers = args.workers or (8 if args.backend == "vllm" else 1)
    if (args.max_questions < 1 or args.top_k < 1 or args.max_iterations < 0
            or not 0 <= args.min_probe_iterations <= args.max_iterations
            or args.max_new_tokens < 1 or workers < 1):
        parser.error("positive limits required and 0 <= min-probe-iterations <= max-iterations")
    if args.backend == "transformers" and workers != 1:
        parser.error("transformers backend requires --workers 1")

    work, data = args.work_dir, args.work_dir / "data"
    corpus = data / "corpus.jsonl"
    index_manifest = json.loads((work / "index/manifest.json").read_text(encoding="utf-8"))
    if index_manifest["corpus_sha256"] != sha256(corpus):
        raise ValueError("corpus changed after E5 indexing")
    queries = read_jsonl(data / "queries.jsonl")
    if args.question_id:
        queries = [row for row in queries if row["_id"] == args.question_id]
    queries = queries[:args.max_questions]
    if not queries:
        parser.error("no questions selected")
    answers = {row["id"]: row["answer"] for row in read_jsonl(data / "answers.jsonl")}
    supports = {row["id"]: set(row["support_ids"]) for row in read_jsonl(data / "qrels.jsonl")}
    results = work / "results/comorag"
    output = results / f"{args.run_name}.jsonl"
    summary_path = results / f"{args.run_name}.summary.json"
    if output.exists() or summary_path.exists():
        raise FileExistsError(f"run name already used: {args.run_name}")
    if args.check_only:
        print(f"Ready: {len(queries)} question(s); {[row['_id'] for row in queries]}")
        print(f"Results will be written to {output}")
        return

    # Match the import order used by the existing DeepRAG smoke runner.
    import zstandard  # noqa: F401
    from rag import load_comorag, load_e5, run_comorag

    retriever = load_e5(index_dir=str(work / "index"), corpus_path=str(corpus),
                        model_name=index_manifest["model"], device=args.device)
    model = load_comorag(args.model, max_new_tokens=args.max_new_tokens,
                         backend=args.backend, base_url=args.model_url)
    results.mkdir(parents=True, exist_ok=True)
    correct = any_support = all_support = errors = 0
    total_f1 = 0.0
    search_lock = Lock()

    def process(query):
        qid = query["_id"]
        row = {"id": qid, "question": query["text"], "gold_answer": answers[qid]}

        def search(probe, k):
            with search_lock:
                return retriever.search(probe, k)

        try:
            state = run_comorag(query["text"], search, model.encode,
                                model.answer, model.make_probes, model.fuse,
                                top_k=args.top_k, max_iterations=args.max_iterations,
                                min_probe_iterations=args.min_probe_iterations)
            seen = {doc.id for item in state.retrievals for doc in item.documents}
            short = re.search(r"<answer short>(.*?)</answer short>", state.final_answer, re.S)
            answer = short.group(1).strip() if short else ""
            match = bool(answer) and normalize_answer(answer) == normalize_answer(answers[qid])
            score = f1_score(answer, answers[qid])
            row.update({"answer": answer, "exact_match": match, "f1": score,
                        "support_seen": sorted(seen & supports[qid]), "state": asdict(state)})
            return (row, match, score, bool(seen & supports[qid]),
                    supports[qid] <= seen, state.stop_reason, False)
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            return row, False, 0.0, False, False, None, True

    stop_counts = Counter()
    with output.open("x", encoding="utf-8") as file:
        for row, match, score, any_hit, all_hit, stop_reason, failed in ordered_parallel_map(
                process, queries, workers):
            correct += match
            total_f1 += score
            any_support += any_hit
            all_support += all_hit
            errors += failed
            if stop_reason:
                stop_counts[stop_reason] += 1
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
            print(f"{row['id']}: {row.get('answer', row.get('error'))}", flush=True)

    sample = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    summary = {"dataset": sample.get("dataset", Path(sample["source"]).name),
               "workflow": "comorag_sequential_single_probe_clean",
               "comorag_mode": COMORAG_MODE,
               "source": sample["source"], "seed": sample["seed"],
               "sample_documents": sample["documents"], "sample_dir": str(data),
               "index_dir": str(work / "index"), "retriever_model": index_manifest["model"],
               "generator_model": args.model, "questions": len(queries),
               "question_ids": [row["_id"] for row in queries], "correct": correct,
               "f1": total_f1 / len(queries),
               "any_support_seen_questions": any_support,
               "all_support_seen_questions": all_support, "stop_reasons": dict(stop_counts),
               "errors": errors, "top_k": args.top_k, "max_iterations": args.max_iterations,
               "min_probe_iterations": args.min_probe_iterations,
               "max_new_tokens": args.max_new_tokens}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Results: {output}\nSummary: {summary_path}")
    if errors == len(queries):
        raise RuntimeError("all ComoRAG questions failed; inspect the JSONL errors")


if __name__ == "__main__":
    main()
