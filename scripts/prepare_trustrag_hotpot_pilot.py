"""Prepare a small TrustRAG HotpotQA pilot without loading its 2.1 GB corpus."""

import argparse
import json
import random
import zipfile
from pathlib import Path


def write_jsonl(path: Path, rows) -> None:
    with path.open("x", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trust-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--questions", type=int, default=10)
    parser.add_argument("--corpus-size", type=int, default=9811)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    attacks_path = args.trust_root / "results/adv_targeted_results/hotpotqa.json"
    attacks = json.loads(attacks_path.read_text(encoding="utf-8"))
    selected = list(attacks.values())[:args.questions]
    selected_ids = {row["id"] for row in selected}
    if len(selected) != args.questions or len(selected_ids) != args.questions:
        raise ValueError("not enough unique TrustRAG questions")

    qrels = {qid: [] for qid in selected_ids}
    with zipfile.ZipFile(args.trust_root / "datasets/hotpotqa.zip") as archive:
        with archive.open("hotpotqa/qrels/test.tsv") as file:
            next(file)
            for raw_line in file:
                qid, doc_id, _ = raw_line.decode().rstrip("\n").split("\t")
                if qid in qrels:
                    qrels[qid].append(doc_id)
    if any(not ids for ids in qrels.values()):
        raise ValueError("selected question is missing qrels")

    support_ids = {doc_id for ids in qrels.values() for doc_id in ids}
    distractor_count = args.corpus_size - len(support_ids)
    if distractor_count < 0:
        raise ValueError("corpus-size is smaller than the supporting-document set")
    rng = random.Random(args.seed)
    supports, distractors = {}, []
    seen = 0
    with (args.trust_root / "datasets/hotpotqa/corpus.jsonl").open(
            encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if row["_id"] in support_ids:
                supports[row["_id"]] = row
                continue
            seen += 1
            if len(distractors) < distractor_count:
                distractors.append(row)
            else:
                index = rng.randrange(seen)
                if index < distractor_count:
                    distractors[index] = row
    missing = support_ids - supports.keys()
    if missing:
        raise ValueError(f"missing {len(missing)} supporting documents from corpus")

    data = args.output / "data"
    data.mkdir(parents=True)
    (args.output / "index").mkdir()
    (args.output / "results").mkdir()
    write_jsonl(data / "queries.jsonl", (
        {"_id": row["id"], "text": row["question"]} for row in selected))
    write_jsonl(data / "answers.jsonl", (
        {"id": row["id"], "answer": row["correct answer"]} for row in selected))
    write_jsonl(data / "targets.jsonl", (
        {"question_id": row["id"], "target_answer": row["incorrect answer"]}
        for row in selected))
    write_jsonl(data / "qrels.jsonl", (
        {"id": row["id"], "support_ids": qrels[row["id"]]} for row in selected))
    write_jsonl(data / "corpus.jsonl", [*supports.values(), *distractors])
    (data / "manifest.json").write_text(json.dumps({
        "dataset": "hotpotqa_trustrag", "source": str(args.trust_root),
        "seed": args.seed, "questions": len(selected),
        "documents": len(supports) + len(distractors),
        "selection": "first TrustRAG attack targets plus all supports and reservoir distractors",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(selected)} questions and "
          f"{len(supports) + len(distractors)} documents in {args.output}")


if __name__ == "__main__":
    main()
