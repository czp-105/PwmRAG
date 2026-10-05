from collections.abc import Callable

from .state import (
    ComoRAGMemoryNode, ComoRAGMemoryPool, ComoRAGRetrieval, ComoRAGState,
    Decision, Document, State, Step,
)


def run_deeprag(
    question: str,
    decide: Callable[[State], Decision],
    search: Callable[[str, int], list[Document]],
    answer_step: Callable[[State, Decision, list[Document]], str],
    answer_final: Callable[[State], str],
    *,
    top_k: int = 5,
    max_steps: int = 6,
) -> State:
    """Run a multi-step retrieval/parametric-answer loop using supplied components."""
    if not question.strip() or top_k < 1 or max_steps < 1:
        raise ValueError("question must be nonempty; top_k and max_steps must be positive")

    state = State(question=question)
    for _ in range(max_steps):
        decision = decide(state)
        if decision.action == "finish":
            state.stop_reason = "model"
            break
        if decision.action not in ("retrieve", "parametric") or not decision.query.strip():
            raise ValueError(f"invalid decision: {decision}")

        documents = search(decision.query, top_k) if decision.action == "retrieve" else []
        answer = answer_step(state, decision, documents)
        state.steps.append(Step(decision.query, decision.action, documents, answer))
    else:
        state.stop_reason = "max_steps"

    state.final_answer = answer_final(state)
    return state


def run_comorag(
    question: str,
    search: Callable[[str, int], list[Document]],
    encode: Callable[[str, list[Document], str], list[ComoRAGMemoryNode]],
    answer: Callable[[str, ComoRAGMemoryPool], str],
    make_probes: Callable[[str, ComoRAGMemoryPool], list[str]],
    fuse: Callable[[str, ComoRAGMemoryPool], ComoRAGMemoryNode | None],
    *,
    top_k: int = 5,
    max_iterations: int = 5,
    min_probe_iterations: int = 0,
) -> ComoRAGState:
    """Retry answering after '*' by committing memory and probing for evidence."""
    if (not question.strip() or top_k < 1 or max_iterations < 0
            or not 0 <= min_probe_iterations <= max_iterations):
        raise ValueError(
            "question must be nonempty; top_k positive; "
            "0 <= min_probe_iterations <= max_iterations")

    state = ComoRAGState(question)

    def retrieve(probe: str) -> None:
        documents = search(probe, top_k)
        state.retrievals.append(ComoRAGRetrieval(probe, tuple(documents)))
        for node in encode(probe, documents, question):
            state.memory.stage(node)

    retrieve(question)
    for iteration in range(max_iterations + 1):
        response = answer(question, state.memory).strip()
        if not response:
            raise ValueError("ComoRAG produced an empty answer")
        state.attempts.append(response)
        if response != "*" and iteration >= min_probe_iterations:
            state.final_answer = response
            state.stop_reason = "answered"
            break
        if iteration == max_iterations:
            state.final_answer = response
            state.stop_reason = "max_iterations" if response == "*" else "answered"
            break

        probes = [probe.strip() for probe in make_probes(question, state.memory) if probe.strip()][:1]
        state.memory.commit()
        if not probes:
            state.final_answer = response
            state.stop_reason = "no_probes"
            break
        for probe in probes:
            retrieve(probe)
        fused = fuse(question, state.memory)
        if fused is not None:
            state.memory.stage(fused)

    return state


# Keep the original entry point for existing DeepRAG callers.
run = run_deeprag
