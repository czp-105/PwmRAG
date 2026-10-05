"""Prepare and run a small DeepRAG experiment with real E5 retrieval."""

import argparse
import hashlib
import json
import random
import re
from dataclasses import asdict
from pathlib import Path
from threading import Lock

from rag.parallel import ordered_parallel_map


ROOT = Path(__file__).resolve().parent
SUPPORTED_DATASETS = ("2wikimultihopqa", "hotpotqa")
DEFAULT_DATASET = "2wikimultihopqa"


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(args):
    source, work = args.source, args.work_dir
    if work.exists():
        raise FileExistsError(f"experiment directory already exists: {work}")
    queries = read_jsonl(source / "queries.jsonl")
    answers = {row["id"]: row["answer"] for row in read_jsonl(source / "answers.jsonl")}
    corpus = read_jsonl(source / "corpus.jsonl")
    corpus_ids = {row["_id"] for row in corpus}
    qrels = {}
    with (source / "qrels/test.tsv").open(encoding="utf-8") as file:
        next(file)
        for line in file:
            qid, doc_id, _ = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, []).append(doc_id)

    eligible = [row for row in queries if row["_id"] in answers
                and len(qrels.get(row["_id"], [])) == 2
                and set(qrels[row["_id"]]) <= corpus_ids]
    if args.questions < 1 or args.questions > len(eligible):
        raise ValueError(f"questions must be between 1 and {len(eligible)}")
    rng = random.Random(args.seed)
    selected = rng.sample(eligible, args.questions)
    support_ids = {doc_id for row in selected for doc_id in qrels[row["_id"]]}
    if args.corpus_size < len(support_ids) or args.corpus_size > len(corpus):
        raise ValueError(f"corpus_size must be between {len(support_ids)} and {len(corpus)}")
    distractors = [row["_id"] for row in corpus if row["_id"] not in support_ids]
    selected_ids = support_ids | set(rng.sample(distractors, args.corpus_size - len(support_ids)))

    data = work / "data"
    data.mkdir(parents=True)
    (work / "index").mkdir()
    (work / "results").mkdir()
    write_jsonl(data / "queries.jsonl", selected)
    write_jsonl(data / "answers.jsonl", [{"id": row["_id"], "answer": answers[row["_id"]]} for row in selected])
    write_jsonl(data / "corpus.jsonl", (row for row in corpus if row["_id"] in selected_ids))
    write_jsonl(data / "qrels.jsonl", [{"id": row["_id"], "support_ids": qrels[row["_id"]]} for row in selected])
    (data / "manifest.json").write_text(json.dumps({
        "dataset": args.dataset, "source": str(source), "seed": args.seed,
        "questions": len(selected),
        "documents": len(selected_ids), "selection": "two-support questions plus random distractors",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(selected)} questions and {len(selected_ids)} documents in {work}")


def build_index(args):
    import torch
    from src.models.encoder import SimpleEncoder

    data = args.work_dir / "data/corpus.jsonl"
    index = args.work_dir / "index/embedding-shard-00000.pt"
    if index.exists():
        raise FileExistsError(f"index already exists: {index}")
    corpus = read_jsonl(data)
    encoder = SimpleEncoder(args.e5_model, device=args.device, batch_size=args.batch_size)
    embeddings = encoder.encode_corpus(corpus)
    if embeddings.shape[0] != len(corpus):
        raise ValueError("E5 embedding count does not match corpus")
    temporary = index.with_suffix(".pt.tmp")
    torch.save(embeddings, temporary)
    temporary.replace(index)
    (args.work_dir / "index/manifest.json").write_text(json.dumps({
        "model": args.e5_model,
        "corpus_sha256": sha256(data),
        "documents": len(corpus),
        "embedding_dim": embeddings.shape[1],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Indexed {len(corpus)} documents at {index}")


def run_smoke(args):
    # The shared base environment needs zstandard loaded before the Qwen model stack.
    import zstandard  # noqa: F401
    from rag import load_deeprag, load_e5, run_deeprag

    work = args.work_dir
    data = work / "data"
    manifest = json.loads((work / "index/manifest.json").read_text(encoding="utf-8"))
    corpus_path = data / "corpus.jsonl"
    if manifest["corpus_sha256"] != sha256(corpus_path):
        raise ValueError("corpus changed after E5 indexing")
    if args.max_questions < 1:
        raise ValueError("max_questions must be positive")
    queries = read_jsonl(data / "queries.jsonl")
    if args.question_id:
        queries = [row for row in queries if row["_id"] == args.question_id]
    queries = queries[:args.max_questions]
    if not queries:
        raise ValueError("no questions selected")
    answers = {row["id"]: row["answer"] for row in read_jsonl(data / "answers.jsonl")}
    supports = {row["id"]: set(row["support_ids"]) for row in read_jsonl(data / "qrels.jsonl")}
    results = work / "results"
    output = results / f"{args.run_name}.jsonl"
    summary_path = results / f"{args.run_name}.summary.json"
    if output.exists() or summary_path.exists():
        raise FileExistsError(f"run name already used: {args.run_name}")

    retriever = load_e5(index_dir=str(work / "index"), corpus_path=str(corpus_path),
                        model_name=manifest["model"], device=args.device)
    workers = args.workers or (8 if args.backend == "vllm" else 1)
    if args.backend == "transformers" and workers != 1:
        raise ValueError("transformers backend requires --workers 1")
    model = load_deeprag(
        args.deeprag_model, max_new_tokens=args.max_new_tokens,
        backend=args.backend, base_url=args.model_url,
    )
    correct = retrieved = any_support = all_support = errors = 0
    search_lock = Lock()

    def process(query):
        qid = query["_id"]
        row = {"id": qid, "question": query["text"], "gold_answer": answers[qid]}

        def search(text, k):
            with search_lock:
                return retriever.search(text, k)

        try:
            state = run_deeprag(query["text"], model.decide, search,
                                model.answer_step, model.answer_final,
                                top_k=args.top_k, max_steps=args.max_steps)
            found = {doc.id for step in state.steps for doc in step.documents}
            short = re.search(r"<answer short>(.*?)</answer short>", state.final_answer, re.S)
            answer = short.group(1).strip() if short else ""
            match = answer.casefold() == answers[qid].strip().casefold()
            used_retrieval = any(step.action == "retrieve" for step in state.steps)
            row.update({"answer": answer, "exact_match": match,
                        "support_seen": sorted(found & supports[qid]), "state": asdict(state)})
            return row, match, used_retrieval, bool(found & supports[qid]), supports[qid] <= found, False
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            return row, False, False, False, False, True

    with output.open("x", encoding="utf-8") as file:
        for row, match, used_retrieval, any_hit, all_hit, failed in ordered_parallel_map(
                process, queries, workers):
            correct += match
            retrieved += used_retrieval
            any_support += any_hit
            all_support += all_hit
            errors += failed
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
            print(f"{row['id']}: {row.get('answer', row.get('error'))}", flush=True)

    sample = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    summary = {"dataset": sample.get("dataset", Path(sample["source"]).name),
               "source": sample["source"],
               "seed": sample["seed"], "sample_documents": sample["documents"],
               "sample_dir": str(data), "index_dir": str(work / "index"),
               "retriever_model": manifest["model"], "generator_model": args.deeprag_model,
               "questions": len(queries), "question_ids": [row["_id"] for row in queries],
               "correct": correct, "retrieved_questions": retrieved,
               "any_support_seen_questions": any_support,
               "all_support_seen_questions": all_support,
               "errors": errors, "top_k": args.top_k, "max_steps": args.max_steps,
               "max_new_tokens": args.max_new_tokens}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Results: {output}\nSummary: {summary_path}")
    if errors == len(queries):
        raise RuntimeError("all smoke questions failed; inspect the JSONL errors")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "index", "run"))
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--questions", type=int, default=20)
    parser.add_argument("--corpus-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--e5-model", default="intfloat/e5-large-v2")
    parser.add_argument("--deeprag-model", default="xinyan233333/DeepRAG-7b")
    parser.add_argument("--backend", choices=("transformers", "vllm"), default="transformers")
    parser.add_argument("--model-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-questions", type=int, default=2)
    parser.add_argument("--question-id")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--run-name", default="smoke")
    args = parser.parse_args()
    args.dataset = args.dataset or (args.source.name if args.source else DEFAULT_DATASET)
    args.source = args.source or ROOT / "datasets" / args.dataset
    args.work_dir = args.work_dir or ROOT / "experiments" / args.dataset / "smoke_v1"
    if args.workers is not None and args.workers < 1:
        parser.error("workers must be positive")
    {"prepare": prepare, "index": build_index, "run": run_smoke}[args.stage](args)


if __name__ == "__main__":
    main()
