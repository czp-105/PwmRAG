import unittest
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from rag import Decision, DeepRAGModel, Document, E5Adapter, load_deeprag, load_e5, run


class WorkflowTest(unittest.TestCase):
    def test_vllm_generator_uses_completion_api_and_keeps_stop_text(self):
        from rag import model as model_module

        self.assertTrue(hasattr(model_module, "VLLMGenerator"))
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                requests.append((self.path, json.loads(self.rfile.read(length))))
                body = json.dumps({"choices": [{"text": "answer STOP"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            tokenizer = object()
            generator = model_module.VLLMGenerator(
                "Qwen/Qwen3-8B", f"http://127.0.0.1:{server.server_port}/v1",
                tokenizer=tokenizer,
            )
            output, tokens, logprobs = generator.generate("PROMPT", 17, ["STOP"])
            generator.generate(
                "SAMPLED", 23, temperature=1.0, top_p=0.9, seed=42)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

        self.assertEqual((output, tokens, logprobs), ("answer STOP", None, None))
        self.assertIs(generator.tokenizer, tokenizer)
        self.assertEqual(requests, [
            ("/v1/completions", {
                "model": "Qwen/Qwen3-8B", "prompt": "PROMPT", "max_tokens": 17,
                "temperature": 0, "top_p": 1.0, "seed": None, "stop": ["STOP"],
                "include_stop_str_in_output": True,
            }),
            ("/v1/completions", {
                "model": "Qwen/Qwen3-8B", "prompt": "SAMPLED", "max_tokens": 23,
                "temperature": 1.0, "top_p": 0.9, "seed": 42, "stop": None,
                "include_stop_str_in_output": True,
            }),
        ])

    def test_retrieve_parametric_then_finish(self):
        searches = []
        decisions = [
            Decision("retrieve", "first query"),
            Decision("parametric", "second query"),
            Decision("finish"),
        ]

        def search(query, k):
            searches.append((query, k))
            return [Document("poison-1", "false fact", is_poisoned=True)]

        state = run(
            "question",
            lambda state: decisions[len(state.steps)],
            search,
            lambda state, decision, docs: f"{decision.action}:{decision.query}",
            lambda state: " | ".join(step.intermediate_answer for step in state.steps),
            top_k=3,
        )

        self.assertEqual(searches, [("first query", 3)])
        self.assertEqual([step.action for step in state.steps], ["retrieve", "parametric"])
        self.assertEqual(state.steps[0].documents[0].id, "poison-1")
        self.assertTrue(state.steps[0].documents[0].is_poisoned)
        self.assertEqual(state.steps[1].documents, [])
        self.assertEqual(state.stop_reason, "model")
        self.assertEqual(state.final_answer, "retrieve:first query | parametric:second query")

    def test_step_limit_and_invalid_decision(self):
        base = (lambda query, k: [], lambda state, decision, docs: "ok", lambda state: "done")
        state = run("question", lambda state: Decision("parametric", "q"), *base, max_steps=2)
        self.assertEqual((len(state.steps), state.stop_reason), (2, "max_steps"))
        with self.assertRaises(ValueError):
            run("question", lambda state: Decision("retrieve"), *base)

    def test_e5_adapter_preserves_rank_and_poison_labels(self):
        class FakeE5:
            def search(self, query, k):
                self.call = (query, k)
                return [
                    {"id": "clean-1", "title": "A", "contents": "fact", "score": 0.9, "is_poisoned": False},
                    {"id": "poison-1", "title": "B", "contents": "false fact", "score": 0.8, "is_poisoned": True},
                ][:k]

        backend = FakeE5()
        adapter = E5Adapter(backend)
        state = run(
            "question",
            lambda state: Decision("retrieve", "subquery") if not state.steps else Decision("finish"),
            adapter.search,
            lambda state, decision, docs: docs[0].text,
            lambda state: state.steps[-1].intermediate_answer,
            top_k=2,
        )

        self.assertEqual(backend.call, ("subquery", 2))
        self.assertEqual([doc.id for doc in state.steps[0].documents], ["clean-1", "poison-1"])
        self.assertEqual([doc.is_poisoned for doc in state.steps[0].documents], [False, True])
        self.assertEqual(state.steps[0].documents[1].score, 0.8)

    def test_missing_index_fails_before_model_load(self):
        with self.assertRaises(FileNotFoundError):
            load_e5(index_dir="/nonexistent/e5_index", corpus_path="/nonexistent/corpus.jsonl")
        with self.assertRaises(ValueError):
            load_e5(index_dir="unused", corpus_path="unused", poisoned_index_dir="unused")

    def test_deeprag_model_protocol(self):
        class FakeGenerator:
            tokenizer = None

            def __init__(self):
                self.outputs = iter([
                    "Who directed Spitfire?\nLet",
                    "John Cromwell\nFollow up:",
                    "Follow up: When did John Cromwell die?\nIntermediate answer:",
                    "1979",
                    "So the final answer is:",
                    "John Cromwell died in 1979.</answer long><answer short>1979</answer short>",
                ])
                self.prompts = []
                self.tokenizer = self

            def apply_chat_template(self, messages, tokenize=False):
                self.prompts.append(messages)
                return messages[0]["content"] + "\n" + messages[1]["content"] + "<|im_end|>"

            def generate(self, prompt, max_length, stop_words):
                self.prompts[-1].append(prompt)
                return next(self.outputs), None, None

        generator = FakeGenerator()
        model = DeepRAGModel(generator)
        searches = []

        def search(query, k):
            searches.append(query)
            return [Document("1", "John Cromwell directed Spitfire.", title="Spitfire", is_poisoned=True)]

        state = run("When did the director of Spitfire die?", model.decide, search,
                    model.answer_step, model.answer_final)

        self.assertEqual(searches, ["Who directed Spitfire?"])
        self.assertEqual([step.action for step in state.steps], ["retrieve", "parametric"])
        self.assertIn("Context:\n[1] Title: Spitfire Text: John Cromwell directed Spitfire.",
                      generator.prompts[1][1]["content"])
        self.assertIn("Intermediate answer: John Cromwell", generator.prompts[2][1]["content"])
        self.assertTrue(generator.prompts[2][1]["content"].endswith("\n\n"))
        self.assertNotIn("is_poisoned", generator.prompts[1][1]["content"])
        self.assertEqual(state.final_answer,
                         "<answer long>John Cromwell died in 1979.</answer long><answer short>1979</answer short>")
        self.assertEqual(state.stop_reason, "model")

    def test_local_generator_passes_stop_words_to_transformers(self):
        import torch

        class Inputs(dict):
            def to(self, device):
                return self

        class Tokenizer:
            def __call__(self, prompt, return_tensors):
                return Inputs(input_ids=torch.tensor([[1, 2]]))

            def decode(self, tokens, skip_special_tokens):
                return "query Let"

        class Model:
            device = "cpu"

            def eval(self):
                return self

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return torch.tensor([[1, 2, 3]])

        tokenizer, backend = Tokenizer(), Model()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             patch("transformers.AutoModelForCausalLM.from_pretrained",
                   return_value=backend) as model_loader:
            adapter = load_deeprag("unused", device="cpu")

        model_loader.assert_called_once_with("unused", device_map="cpu", torch_dtype="auto")

        output, _, _ = adapter.generator.generate("prompt", 32, ["Let", "So the final"])
        self.assertEqual(output, "query Let")
        self.assertEqual(backend.kwargs["stop_strings"], ["Let", "So the final"])
        self.assertIs(backend.kwargs["tokenizer"], tokenizer)
        self.assertEqual(backend.kwargs["max_new_tokens"], 32)

        adapter.generator.generate(
            "prompt", 64, temperature=1.0, top_p=0.9, seed=42)
        self.assertTrue(backend.kwargs["do_sample"])
        self.assertEqual(backend.kwargs["temperature"], 1.0)
        self.assertEqual(backend.kwargs["top_p"], 0.9)
        self.assertEqual(backend.kwargs["generator"].initial_seed(), 42)

    def test_decision_prefers_explicit_follow_up_after_preamble(self):
        class Generator:
            class Tokenizer:
                def apply_chat_template(self, messages, tokenize=False):
                    return messages[1]["content"] + "<|im_end|>"

            tokenizer = Tokenizer()

            def generate(self, prompt, max_length, stop_words):
                return "The context says John Cromwell.\nFollow up: When did John Cromwell die?\nLet", None, None

        from rag import State
        decision = DeepRAGModel(Generator()).decide(State("Question"))
        self.assertEqual(decision, Decision("retrieve", "When did John Cromwell die?"))


if __name__ == "__main__":
    unittest.main()
