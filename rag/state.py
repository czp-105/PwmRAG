from dataclasses import dataclass, field
from typing import Literal


Action = Literal["retrieve", "parametric", "finish"]


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    title: str = ""
    score: float = 0.0
    is_poisoned: bool = False


@dataclass(frozen=True)
class Decision:
    action: Action
    query: str = ""


@dataclass
class Step:
    query: str
    action: Literal["retrieve", "parametric"]
    documents: list[Document]
    intermediate_answer: str


@dataclass
class State:
    question: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str = ""
    stop_reason: str = ""


ComoRAGMemoryKind = Literal["veridical", "fusion"]


@dataclass(frozen=True)
class ComoRAGMemoryNode:
    probe: str
    kind: ComoRAGMemoryKind
    contents: tuple[str, ...] = ()
    cue: str = ""
    source_ids: tuple[str, ...] = ()
    poisoned_source_ids: tuple[str, ...] = ()


@dataclass
class ComoRAGMemoryPool:
    main: list[ComoRAGMemoryNode] = field(default_factory=list)
    temporary: list[ComoRAGMemoryNode] = field(default_factory=list)

    def stage(self, node: ComoRAGMemoryNode) -> None:
        self.temporary.append(node)

    def commit(self) -> None:
        self.main.extend(self.temporary)
        self.temporary.clear()

    def by_kind(self, kind: ComoRAGMemoryKind, *, temporary: bool = False) -> list[ComoRAGMemoryNode]:
        nodes = self.temporary if temporary else self.main
        return [node for node in nodes if node.kind == kind]

    def probes(self) -> list[str]:
        return list(dict.fromkeys(node.probe for node in self.main if node.probe))


@dataclass(frozen=True)
class ComoRAGRetrieval:
    probe: str
    documents: tuple[Document, ...]


@dataclass
class ComoRAGState:
    question: str
    memory: ComoRAGMemoryPool = field(default_factory=ComoRAGMemoryPool)
    retrievals: list[ComoRAGRetrieval] = field(default_factory=list)
    attempts: list[str] = field(default_factory=list)
    final_answer: str = ""
    stop_reason: str = ""
