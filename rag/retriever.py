from pathlib import Path

from .state import Document


class E5Adapter:
    """Convert the existing E5 retriever's dict results to trajectory documents."""

    def __init__(self, retriever):
        self.retriever = retriever

    def search(self, query: str, k: int) -> list[Document]:
        return [
            Document(
                id=str(hit["id"]),
                title=hit.get("title", ""),
                text=hit["contents"],
                score=float(hit["score"]),
                is_poisoned=hit.get("is_poisoned", False),
            )
            for hit in self.retriever.search(query, k=k)
        ]


def load_e5(
    *,
    index_dir: str,
    corpus_path: str,
    poisoned_index_dir: str | None = None,
    poisoned_corpus_path: str | None = None,
    model_name: str = "intfloat/e5-large-v2",
    device: str | None = None,
    encoder=None,
) -> E5Adapter:
    """Initialize the copied E5 implementation after checking its input artifacts."""
    if bool(poisoned_index_dir) != bool(poisoned_corpus_path):
        raise ValueError("poisoned index and corpus must be supplied together")
    for path in (corpus_path, poisoned_corpus_path):
        if path and not Path(path).is_file():
            raise FileNotFoundError(f"corpus not found: {path}")
    for path in (index_dir, poisoned_index_dir):
        if path and not any(Path(path).glob("embedding-shard-*.pt")):
            raise FileNotFoundError(f"E5 index shards not found: {path}")

    from src.e5_retriever import E5_Retriever

    options = {"model_name": model_name, "encoder": encoder}
    if device is not None:
        options["device"] = device
    return E5Adapter(
        E5_Retriever(
            index_dir=index_dir,
            corpus_path=corpus_path,
            poisoned_index_dir=poisoned_index_dir,
            poisoned_corpus_path=poisoned_corpus_path,
            **options,
        )
    )
