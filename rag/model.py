"""Model adapters for the DeepRAG and ComoRAG workflows."""

import json
import re
import runpy
import urllib.request
from pathlib import Path
from string import Template

from .state import ComoRAGMemoryNode, ComoRAGMemoryPool, Decision, Document, State


INSTRUCTION = (
    "Instruction: You are a helpful Retrieve-Augmented Generation (RAG) model. "
    "Your task is to answer questions by logically decomposing them into clear "
    'sub-questions and iteratively addressing each one. Use "Follow up:" to '
    'introduce each sub-question and "Intermediate answer:" to provide answers. '
    "For each sub-question, decide whether you can provide a direct answer or if "
    'additional information is required. If additional information is needed, state, '
    '"Let\'s search the question in Wikipedia." and then use the retrieved '
    "information to respond comprehensively. If a direct answer is possible, "
    "provide it immediately without searching.\n\n"
)
SEARCH_LINE = "Let's search the question in Wikipedia."
END_MARKERS = ("<|eot_id|>", "<|im_end|>")
COMORAG_QA_TEMPLATE = runpy.run_path(
    Path(__file__).resolve().parents[1]
    / "static/ComoRAG/src/comorag/prompts/templates/rag_qa_narrativeqa.py"
)["prompt_template"]
COMORAG_CUE_TEMPLATE = runpy.run_path(
    Path(__file__).resolve().parents[1]
    / "static/ComoRAG/src/comorag/prompts/templates/memory_fusion.py"
)["prompt_template"]
COMORAG_PROBE_TEMPLATE = runpy.run_path(
    Path(__file__).resolve().parents[1]
    / "static/ComoRAG/src/comorag/prompts/templates/agent_probe.py"
)["prompt_template"]
COMORAG_FUSION_TEMPLATE = runpy.run_path(
    Path(__file__).resolve().parents[1]
    / "static/ComoRAG/src/comorag/prompts/templates/node_fusion.py"
)["prompt_template"]


class VLLMGenerator:
    """Use a vLLM OpenAI-compatible completion endpoint."""

    def __init__(self, model_name: str, base_url: str, tokenizer=None):
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt, max_length, stop_words=None, *, temperature=0,
                 top_p=1.0, seed=None):
        payload = json.dumps({
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "stop": stop_words,
            "include_stop_str_in_output": True,
        }).encode()
        request = urllib.request.Request(
            f"{self.base_url}/completions", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
        try:
            return result["choices"][0]["text"], None, None
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError(f"invalid vLLM response: {result!r}") from error


def _context(documents: list[Document]) -> str:
    return "\n".join(
        f"[{i}] Title: {doc.title} Text: {doc.text}"
        for i, doc in enumerate(documents, 1)
    )


def _history(state: State) -> str:
    text = ""
    for step in state.steps:
        text += f"\nFollow up: {step.query}\n"
        if step.action == "retrieve":
            text += f"{SEARCH_LINE}\nContext:\n{_context(step.documents)}\n"
        text += f"Intermediate answer: {step.intermediate_answer}"
    return text


def _clean(text: str) -> str:
    for marker in ("Follow up:", "Intermediate answer:", "So the final answer", *END_MARKERS):
        text = text.split(marker, 1)[0]
    return text.strip()


class DeepRAGModel:
    """Adapt a generator to DeepRAG's prompt protocol and workflow callbacks."""

    def __init__(self, generator, max_new_tokens: int = 256):
        self.generator = generator
        self.max_new_tokens = max_new_tokens

    def _generate(self, state: State, assistant: str, stop_words: list[str]) -> str:
        prompt = self.generator.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": INSTRUCTION + "Question: " + state.question},
                {"role": "assistant", "content": assistant},
            ],
            tokenize=False,
        ).strip("\n")
        for marker in END_MARKERS:
            prompt = prompt.removesuffix(marker)
        output, _, _ = self.generator.generate(
            prompt, max_length=self.max_new_tokens, stop_words=stop_words
        )
        return output.strip()

    def decide(self, state: State) -> Decision:
        # DeepRAG lets the model emit "Follow up:" after an existing history.
        prefix = _history(state) + "\n\n" if state.steps else "Follow up:"
        output = self._generate(
            state, prefix, [*END_MARKERS, "Intermediate answer:", "So the final", "Let"]
        )
        if output.startswith("So the final"):
            return Decision("finish")
        # Prefer the explicit follow-up if the model emits preamble before it.
        follow_up = output.rsplit("Follow up:", 1)[-1].strip()
        first_line, _, rest = follow_up.partition("\n")
        retrieve = (output.rstrip().endswith("Let") or SEARCH_LINE in first_line
                    or rest.lstrip().startswith(SEARCH_LINE))
        query = first_line.split(SEARCH_LINE, 1)[0].removesuffix("Let").strip()
        if not query or query.startswith("So the final"):
            raise ValueError(f"DeepRAG produced no follow-up question: {output!r}")
        return Decision("retrieve" if retrieve else "parametric", query)

    def answer_step(self, state: State, decision: Decision, documents: list[Document]) -> str:
        prefix = _history(state) + ("\n" if state.steps else "") + f"Follow up: {decision.query}\n"
        if decision.action == "retrieve":
            prefix += f"{SEARCH_LINE}\nContext:\n{_context(documents)}\n"
        prefix += "Intermediate answer: "
        output = self._generate(state, prefix, [*END_MARKERS, "Follow up:", "So the final answer"])
        answer = _clean(output)
        if not answer:
            raise ValueError("DeepRAG produced no intermediate answer")
        return answer

    def answer_final(self, state: State) -> str:
        output = self._generate(
            state, _history(state) + "\n\nSo the final answer is: <answer long>",
            [*END_MARKERS, "Follow up:"],
        )
        answer = "<answer long>" + _clean(output.removeprefix("<answer long>"))
        if not output:
            raise ValueError("DeepRAG produced no final answer")
        return answer


class ComoRAGModel:
    """Encode retrieved passages and supply ComoRAG answer/probe/fusion callbacks."""

    def __init__(self, generator, max_new_tokens: int = 512):
        self.generator = generator
        self.max_new_tokens = max_new_tokens

    def _generate_messages(self, messages: list[dict[str, str]]) -> str:
        prompt = self.generator.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        output, _, _ = self.generator.generate(
            prompt, max_length=self.max_new_tokens, stop_words=list(END_MARKERS)
        )
        for marker in END_MARKERS:
            output = output.split(marker, 1)[0]
        output = re.sub(r"^\s*<think>.*?</think>\s*", "", output, flags=re.S).strip()
        if not output:
            raise ValueError("ComoRAG model returned empty output")
        return output

    def encode(self, probe: str, documents: list[Document], question: str | None = None
               ) -> list[ComoRAGMemoryNode]:
        if not documents:
            return []
        contents = tuple(f"Title: {doc.title} Text: {doc.text}" for doc in documents)
        cue_query = probe if not question or question == probe else f"{question} {probe}"
        cue = self._generate_messages([
            {"role": item["role"], "content": Template(item["content"]).substitute(
                query=cue_query, content="\n".join(contents))}
            for item in COMORAG_CUE_TEMPLATE
        ])
        return [ComoRAGMemoryNode(
            probe=probe, kind="veridical", contents=contents, cue=cue,
            source_ids=tuple(doc.id for doc in documents),
            poisoned_source_ids=tuple(doc.id for doc in documents if doc.is_poisoned),
        )]

    def answer(self, question: str, memory: ComoRAGMemoryPool) -> str:
        historical = memory.by_kind("veridical")
        current = memory.by_kind("veridical", temporary=True)
        evidence = [node.cue for node in historical if node.cue]
        evidence.extend("\n".join(node.contents) for node in current if node.contents)
        context = ["### Detail Chunks\n" + "\n".join(evidence)] if evidence else []
        fused = memory.by_kind("fusion") + memory.by_kind("fusion", temporary=True)
        if fused:
            context.append("### Historical Information\n" + "\n".join(
                f"Probe: {node.probe}\nFinding: {node.cue}" for node in fused
            ))
        prompt_user = "\n\n".join(context + [f"Question: {question}\nThought: "])
        output = self._generate_messages([
            {"role": item["role"],
             "content": Template(item["content"]).substitute(prompt_user=prompt_user)}
            for item in COMORAG_QA_TEMPLATE
        ])
        match = re.search(r"(?m)^### Final Answer\s*\.?\s*:?\s*", output)
        if not match:
            raise ValueError(f"ComoRAG answer lacks final-answer heading: {output!r}")
        lines = output[match.end():].strip().splitlines()
        if not lines or not lines[0].strip():
            raise ValueError("ComoRAG produced no final answer")
        answer = lines[0].strip()
        if answer in {"*", "...", "…"}:
            return "*"
        long_answer = output[:match.start()].strip() or answer
        short_answer = answer
        return (f"<answer long>{long_answer}</answer long>"
                f"<answer short>{short_answer}</answer short>")

    def make_probes(self, question: str, memory: ComoRAGMemoryPool) -> list[str]:
        nodes = memory.by_kind("veridical", temporary=True)
        previous_probes = list(dict.fromkeys(memory.probes() + [node.probe for node in nodes]))
        context = "### Detail Chunks\n" + "\n".join(node.cue for node in nodes if node.cue)
        context += f"\n\nQuestion: {question}\nThought: "
        output = self._generate_messages([
            {"role": item["role"], "content": Template(item["content"]).substitute(
                query=question, context=context,
                previous_probes="\n".join(previous_probes))}
            for item in COMORAG_PROBE_TEMPLATE
        ])
        try:
            probes = json.loads(output)
        except json.JSONDecodeError as error:
            if (error.pos == len(output) and output.lstrip().startswith("{")
                    and not output.endswith("}")):
                try:
                    probes = json.loads(output + "}")
                except json.JSONDecodeError:
                    raise ValueError(f"ComoRAG probes are not JSON: {output!r}") from error
            else:
                raise ValueError(f"ComoRAG probes are not JSON: {output!r}") from error
        if not isinstance(probes, dict):
            raise ValueError("ComoRAG probes must be a JSON object")
        return [value.strip() for key, value in sorted(probes.items())
                if re.fullmatch(r"probe_[1-3]", key) and isinstance(value, str) and value.strip()]

    def fuse(self, question: str, memory: ComoRAGMemoryPool) -> ComoRAGMemoryNode | None:
        nodes = [node for node in memory.main + memory.temporary if node.cue]
        if not nodes:
            return None
        content = "\n\n".join(
            f"Node {i}:\nNote: {node.cue}" for i, node in enumerate(nodes, 1))
        cue = self._generate_messages([
            {"role": item["role"], "content": Template(item["content"]).substitute(
                query=question, content=content)}
            for item in COMORAG_FUSION_TEMPLATE
        ])
        return ComoRAGMemoryNode(
            probe=question, kind="fusion", cue=cue,
            source_ids=tuple(dict.fromkeys(doc_id for node in nodes for doc_id in node.source_ids)),
            poisoned_source_ids=tuple(dict.fromkeys(
                doc_id for node in nodes for doc_id in node.poisoned_source_ids
            )),
        )


def load_generator(model_name: str, device: str = "auto", *,
                   backend: str = "transformers", base_url: str | None = None):
    """Load the shared local Transformers backend for workflows and attack generation."""
    if backend == "vllm":
        return VLLMGenerator(model_name, base_url or "http://127.0.0.1:8000/v1")
    if backend != "transformers":
        raise ValueError("backend must be 'transformers' or 'vllm'")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    class Generator:
        def __init__(self):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map=device, torch_dtype="auto"
            ).eval()

        def generate(self, prompt, max_length, stop_words=None, *, temperature=0,
                     top_p=1.0, seed=None):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            sampling = {}
            if temperature > 0:
                sampling = {"do_sample": True, "temperature": temperature, "top_p": top_p}
                if seed is not None:
                    sampling["generator"] = torch.Generator(
                        device=inputs["input_ids"].device).manual_seed(seed)
            with torch.inference_mode():
                tokens = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    **(sampling or {"do_sample": False}),
                    stop_strings=stop_words,
                    tokenizer=self.tokenizer,
                )
            text = self.tokenizer.decode(
                tokens[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False
            )
            return text, None, None

    return Generator()


def load_deeprag(model_name: str, max_new_tokens: int = 256,
                 device: str = "auto", *, backend: str = "transformers",
                 base_url: str | None = None) -> DeepRAGModel:
    """Load a DeepRAG checkpoint locally through Transformers (no training)."""
    return DeepRAGModel(load_generator(
        model_name, device, backend=backend, base_url=base_url), max_new_tokens)


def load_comorag(model_name: str, max_new_tokens: int = 512,
                 device: str = "auto", *, backend: str = "transformers",
                 base_url: str | None = None) -> ComoRAGModel:
    """Load a generator for the ComoRAG prompt workflow (no training)."""
    return ComoRAGModel(load_generator(
        model_name, device, backend=backend, base_url=base_url), max_new_tokens)
