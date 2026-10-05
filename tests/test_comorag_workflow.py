import unittest

from rag.model import ComoRAGModel
from rag.state import ComoRAGMemoryNode, Document
from rag.workflow import run_comorag
from run_rag_comorag import f1_score, normalize_answer


class ComoRAGWorkflowTest(unittest.TestCase):
    def test_encode_uses_original_comorag_memory_fusion_prompt(self):
        class Generator:
            class Tokenizer:
                def apply_chat_template(inner_self, messages, **kwargs):
                    inner_self.messages = messages
                    inner_self.kwargs = kwargs
                    return "prompt"

            tokenizer = Tokenizer()

            def generate(self, prompt, max_length, stop_words):
                return "- Key Finding: answer", None, None

        generator = Generator()
        nodes = ComoRAGModel(generator).encode(
            "question", [Document("1", "evidence", title="title")])

        messages = generator.tokenizer.messages
        self.assertEqual([item["role"] for item in messages], ["system", "user"])
        self.assertIn("Provide a structured analysis with up to 5 key findings",
                      messages[0]["content"])
        self.assertEqual(messages[1]["content"],
                         "Questions:\nquestion\n\nContent:\nTitle: title Text: evidence"
                         "\n\nYour Response: ")
        self.assertEqual(nodes[0].cue, "- Key Finding: answer")
        self.assertFalse(generator.tokenizer.kwargs["enable_thinking"])

    def test_probe_and_fusion_use_original_comorag_prompts(self):
        class Generator:
            def __init__(self):
                self.tokenizer = self
                self.outputs = iter(['{"probe_1": "next"}', "fused"])
                self.messages = []

            def apply_chat_template(self, messages, **kwargs):
                self.messages.append(messages)
                return "prompt"

            def generate(self, prompt, max_length, stop_words):
                return next(self.outputs), None, None

        generator = Generator()
        memory = __import__("rag").ComoRAGMemoryPool()
        memory.main.append(ComoRAGMemoryNode(
            probe="old", kind="veridical", contents=("old evidence",), cue="old finding"))
        memory.stage(ComoRAGMemoryNode(
            probe="question", kind="veridical", contents=("current evidence",),
            cue="current finding"))
        model = ComoRAGModel(generator)

        self.assertEqual(model.make_probes("question", memory), ["next"])
        self.assertEqual(model.fuse("question", memory).cue, "fused")

        probe_messages, fusion_messages = generator.messages
        self.assertIn("Entity Priority Principle", probe_messages[0]["content"])
        self.assertIn("Original Query:\nquestion", probe_messages[1]["content"])
        self.assertIn("### Detail Chunks\ncurrent finding", probe_messages[1]["content"])
        self.assertNotIn("current evidence", probe_messages[1]["content"])
        self.assertNotIn("old evidence", probe_messages[1]["content"])
        self.assertNotIn("old finding", probe_messages[1]["content"])
        self.assertIn("Previous probes:\nold\nquestion", probe_messages[1]["content"])
        self.assertIn("narrative synthesis specialist", fusion_messages[0]["content"])
        self.assertIn("Previous Analysis:\nNode 1:\nNote: old finding",
                      fusion_messages[1]["content"])
        self.assertIn("Node 2:\nNote: current finding",
                      fusion_messages[1]["content"])

    def test_answer_uses_original_comorag_narrativeqa_prompt(self):
        class Generator:
            class Tokenizer:
                def apply_chat_template(inner_self, messages, **kwargs):
                    inner_self.messages = messages
                    inner_self.kwargs = kwargs
                    return "prompt"

            tokenizer = Tokenizer()

            def generate(self, prompt, max_length, stop_words):
                return "reasoning\n### Final Answer\nanswer", None, None

        generator = Generator()
        memory = __import__("rag").ComoRAGMemoryPool()
        memory.stage(ComoRAGMemoryNode(
            probe="question", kind="veridical", contents=("evidence",)))

        ComoRAGModel(generator).answer("question", memory)

        messages = generator.tokenizer.messages
        self.assertEqual([item["role"] for item in messages],
                         ["system", "user", "assistant", "user"])
        self.assertIn("Use the shortest possible answer taken from the text",
                      messages[0]["content"])
        self.assertIn("### Final Answer\n1862.", messages[2]["content"])
        self.assertIn("### Detail Chunks\nevidence", messages[3]["content"])
        self.assertTrue(messages[3]["content"].endswith("Question: question\nThought: "))
        self.assertFalse(generator.tokenizer.kwargs["enable_thinking"])

    def test_probe_parser_repairs_one_missing_closing_brace(self):
        class Generator:
            class Tokenizer:
                def apply_chat_template(self, messages, **kwargs):
                    return "prompt"

            tokenizer = Tokenizer()

            def generate(self, prompt, max_length, stop_words):
                return ('{"probe_1": "first", "probe_2": "second"', None, None)

        probes = ComoRAGModel(Generator()).make_probes(
            "question", __import__("rag").ComoRAGMemoryPool())

        self.assertEqual(probes, ["first", "second"])

    def test_normalized_answer_ignores_terminal_punctuation(self):
        self.assertEqual(normalize_answer("Württemberg Mausoleum."),
                         normalize_answer("Württemberg Mausoleum"))

    def test_f1_gives_partial_credit_for_more_specific_answer(self):
        self.assertEqual(f1_score("Providence, Rhode Island.", "Providence"), 0.5)

    def test_model_callbacks_complete_answer_or_probe_loop(self):
        class FakeGenerator:
            def __init__(self):
                self.tokenizer = self
                self.outputs = iter([
                    "Augusta's father was William I.",
                    "The evidence does not give the burial place.\n### Final Answer\n*",
                    '{"probe_1": "Where was William I buried?"}',
                    "William I was buried in Württemberg Mausoleum.",
                    "Augusta's father was William I.",
                    "William I was buried in Württemberg Mausoleum.\n### Final Answer\nWürttemberg Mausoleum<|im_end|>",
                ])
                self.prompts = []

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                                    enable_thinking=True):
                self.enable_thinking = enable_thinking
                self.prompts.append(messages)
                return messages[-1]["content"]

            def generate(self, prompt, max_length, stop_words):
                return next(self.outputs), None, None

        generator = FakeGenerator()
        model = ComoRAGModel(generator)
        question = "Where was Augusta's father buried?"

        def search(probe, k):
            if probe == question:
                return [Document("poison-1", "Augusta's father was William I.",
                                 title="Augusta", is_poisoned=True)]
            return [Document("clean-2", "William I is buried in Württemberg Mausoleum.",
                             title="William I")]

        state = run_comorag(question, search, model.encode, model.answer,
                            model.make_probes, model.fuse)

        tagged = ("<answer long>William I was buried in Württemberg Mausoleum.</answer long>"
                  "<answer short>Württemberg Mausoleum</answer short>")
        self.assertEqual(state.final_answer, tagged)
        self.assertFalse(generator.enable_thinking)
        self.assertEqual(state.attempts, ["*", tagged])
        self.assertEqual([item.probe for item in state.retrievals],
                         [question, "Where was William I buried?"])
        self.assertEqual(state.memory.by_kind("fusion", temporary=True)[0].poisoned_source_ids,
                         ("poison-1",))
        self.assertIn("### Historical Information", generator.prompts[-1][-1]["content"])
        self.assertNotIn("poison-1", generator.prompts[-1][-1]["content"])

    def test_insufficient_answer_triggers_probe_and_memory_fusion(self):
        searches = []

        def search(query, k):
            searches.append((query, k))
            return [Document(query, f"evidence for {query}", is_poisoned=query == "father")]

        def encode(probe, documents, question):
            return [ComoRAGMemoryNode(
                probe=probe, kind="veridical", contents=tuple(doc.text for doc in documents),
                source_ids=tuple(doc.id for doc in documents),
                poisoned_source_ids=tuple(doc.id for doc in documents if doc.is_poisoned),
            )]

        def answer(question, memory):
            if not memory.by_kind("fusion", temporary=True):
                return "*"
            self.assertEqual(memory.probes(), [question])
            self.assertEqual(memory.by_kind("veridical", temporary=True)[0].poisoned_source_ids,
                             ("father",))
            return "Württemberg Mausoleum"

        def fuse(question, memory):
            self.assertEqual(len(memory.main), 1)
            self.assertEqual(len(memory.temporary), 1)
            return ComoRAGMemoryNode(probe=question, kind="fusion", cue="father is William I")

        def make_probes(question, memory):
            self.assertEqual(len(memory.main), 0)
            self.assertEqual(len(memory.by_kind("veridical", temporary=True)), 1)
            return ["father"]

        question = "Where was Augusta's father buried?"
        state = run_comorag(question, search, encode, answer,
                            make_probes, fuse, top_k=2)

        self.assertEqual(searches, [(question, 2), ("father", 2)])
        self.assertEqual(state.attempts, ["*", "Württemberg Mausoleum"])
        self.assertEqual(state.stop_reason, "answered")
        self.assertEqual(state.final_answer, "Württemberg Mausoleum")
        self.assertEqual([item.probe for item in state.retrievals], [question, "father"])
        self.assertEqual(len(state.memory.main), 1)

    def test_only_first_probe_is_retrieved_each_iteration(self):
        searches = []

        def search(query, k):
            searches.append(query)
            return [Document(query, f"evidence for {query}")]

        encode = lambda probe, docs, question: [ComoRAGMemoryNode(probe, "veridical")]
        answers = iter(["*", "answer"])
        state = run_comorag(
            "question", search, encode, lambda question, memory: next(answers),
            lambda question, memory: ["next hop", "unrelated branch"],
            lambda question, memory: None,
        )

        self.assertEqual(searches, ["question", "next hop"])
        self.assertEqual(state.final_answer, "answer")

    def test_min_probe_iterations_forces_a_retrieval_before_accepting_answer(self):
        searches = []
        answers = iter(["premature", "final"])
        state = run_comorag(
            "question",
            lambda query, k: searches.append(query) or [Document(query, "evidence")],
            lambda probe, docs, question: [ComoRAGMemoryNode(probe, "veridical")],
            lambda question, memory: next(answers),
            lambda question, memory: ["next hop", "unused"],
            lambda question, memory: None,
            min_probe_iterations=1,
        )

        self.assertEqual(searches, ["question", "next hop"])
        self.assertEqual(state.attempts, ["premature", "final"])
        self.assertEqual(state.final_answer, "final")

    def test_answer_now_and_iteration_limit(self):
        calls = []
        search = lambda query, k: [Document("1", "evidence")]
        encode = lambda probe, docs, question: [ComoRAGMemoryNode(probe, "veridical")]
        probes = lambda question, memory: calls.append("probe") or ["next"]
        fuse = lambda question, memory: None

        answered = run_comorag("q", search, encode, lambda question, memory: "answer",
                                probes, fuse)
        limited = run_comorag("q", search, encode, lambda question, memory: "*",
                               probes, fuse, max_iterations=0)
        self.assertEqual((answered.stop_reason, limited.stop_reason),
                         ("answered", "max_iterations"))
        self.assertEqual(calls, [])
        self.assertEqual(len(limited.retrievals), 1)


if __name__ == "__main__":
    unittest.main()
