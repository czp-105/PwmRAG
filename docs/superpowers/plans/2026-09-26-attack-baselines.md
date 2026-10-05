# PwmRAG Attack Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add faithful, dynamically targeted implementations of `naive`, `ignore`, `completion_real`, `completion_realcmb`, and `poisonedRAG` to PwmRAG.

**Architecture:** Keep experiment orchestration in `rag/generate_attack_doc.py` and put the migrated prompt constants and pure method helpers in one new `rag/attack_methods.py`. Preserve the existing case/JSONL/index contract; static methods return five texts without loading a model, while `poisonedRAG` uses the shared generator with the reference sampling parameters and receives only the original question plus dynamic target.

**Tech Stack:** Python 3, standard-library `random`/`json`/`hashlib`, Transformers, OpenAI-compatible vLLM endpoint, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-26-attack-baselines-design.md`

## Global Constraints

- Migrate the reference prompts verbatim; only replace hardcoded `KILL YOURSELF` with the selected question's `target_answer` and map source fields to PwmRAG fields.
- The five methods are valid only for `attack_goal: targeted_hijacking`.
- Never pass `step.query`, `retrieved_documents`, `state_update`, memory cues, or intermediate answers to the `poisonedRAG` model prompt.
- Each real retrieve step receives exactly five poison documents with globally unique IDs.
- Keep `generation_results.jsonl`, resume behavior, manifest fields, case validation, indexing, and evaluation interfaces compatible.
- Do not add dependencies or a class/plugin framework.
- The PwmRAG root is not a Git checkout, so execution cannot create the per-task commits normally requested by this workflow.

## Review Focus

- A target containing quotes or newlines must be inserted as text without corrupting document JSON serialization; covered in Task 2 target-substitution tests.
- Resume and reordered question selection must reproduce static randomized templates per question and step; covered in Task 2 deterministic-RNG tests.
- A trajectory with no retrieve steps must remain `skipped` and must not load the attacker model; covered in Task 3 integration tests.
- `poisonedRAG` must generate once per question and reuse its five texts across that question's retrieve steps without leaking step data; covered in Task 3 privacy/call-count tests.
- Existing `kidnap_baseline` and `evidence_progression` behavior must remain unchanged; covered by the full regression run in Task 4.

---

### Task 1: Support reference sampling in the shared generator

**Files:**
- Modify: `rag/model.py:44-69,278-308`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: existing `generator.generate(prompt, max_length, stop_words=None)` calls.
- Produces: `generate(prompt, max_length, stop_words=None, *, temperature=0, top_p=1.0, seed=None)` for both backends; existing calls retain greedy defaults.

- [ ] **Step 1: Write failing backend-option tests**

Add tests asserting that a vLLM request contains supplied `temperature`, `top_p`, and `seed`, and that the existing positional three-argument call still uses `temperature == 0`.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python -m unittest tests.test_pipeline -v`

Expected: FAIL because `generate` rejects the new keyword arguments or omits them from the request.

- [ ] **Step 3: Extend both generator implementations minimally**

Update both `generate` signatures with keyword-only `temperature`, `top_p`, and `seed`. The vLLM adapter forwards them in its JSON request. The Transformers adapter keeps `do_sample=False` at temperature zero; otherwise it enables sampling, passes temperature/top-p, and uses a locally seeded `torch.Generator` when a seed is supplied.

- [ ] **Step 4: Run the focused tests**

Run: `python -m unittest tests.test_pipeline -v`

Expected: PASS.

### Task 2: Add faithful prompt and static-method helpers

**Files:**
- Create: `rag/attack_methods.py`
- Modify: `tests/test_attack_interface.py`

**Interfaces:**
- Consumes: method name, dynamic target string, original question, and a caller-owned `random.Random`.
- Produces:
  - `STATIC_ATTACK_METHODS: frozenset[str]`
  - `MIGRATED_ATTACK_METHODS: frozenset[str]`
  - `build_static_attack_texts(method: str, target: str, rng: random.Random) -> list[str]`
  - `build_poisonedrag_messages(question: str, target: str) -> list[list[dict[str, str]]]`
  - `clean_poisonedrag_output(text: str) -> str`

- [ ] **Step 1: Write failing prompt-fidelity tests**

Add tests for all four static methods asserting five nonempty outputs, exact reference boundary markers/newlines, absence of literal `KILL YOURSELF`, and presence of a dynamic target containing both a quote and a newline. For `ignore` and `completion_realcmb`, instantiate two RNGs with the same seed and assert identical output.

- [ ] **Step 2: Write a failing `poisonedRAG` prompt privacy test**

Assert that `build_poisonedrag_messages` returns five equal message lists containing the exact reference system/user wording and the dynamic question/target. Place distinct secret markers in hypothetical query, retrieved documents, and state update, and assert none occur in serialized messages.

- [ ] **Step 3: Run focused tests and verify failure**

Run: `python -m unittest tests.test_attack_interface -v`

Expected: FAIL because `rag.attack_methods` does not exist.

- [ ] **Step 4: Migrate exact constants and helpers**

Copy the five reference terminal instructions and `IGNORE_ATTACK_SENTENCES["test"]` verbatim into `rag/attack_methods.py`. Implement only the five public interfaces above, preserving the source concatenation order for each method. `clean_poisonedrag_output` removes only a leading `Document: ` and surrounding whitespace, then rejects empty output.

- [ ] **Step 5: Run focused tests**

Run: `python -m unittest tests.test_attack_interface -v`

Expected: PASS for the new helper tests and all existing interface tests.

### Task 3: Integrate the five strategies into generation and configuration

**Files:**
- Modify: `rag/config.py:8-88`
- Modify: `rag/generate_attack_doc.py:25-461`
- Modify: `tests/test_attack_interface.py`

**Interfaces:**
- Consumes: Task 2 helper interfaces and Task 1 generator sampling keywords.
- Produces:
  - five new legal `document_strategy` values;
  - existing `generate_cases(args) -> None` support for every migrated method;
  - unchanged case/result schema with method-specific `document_strategy` and document metadata.

- [ ] **Step 1: Write failing config validation tests**

For each migrated method, assert that `load_experiment_config` accepts `targeted_hijacking` and rejects `refusal` and `incorrect_answer`. Keep the existing exact-key validation assertion.

- [ ] **Step 2: Write failing static generation integration tests**

Use a two-step trajectory and a dynamic target. Assert each static method creates ten documents, loads no generator, emits unique IDs containing method/step/role, and records method/source metadata. Assert a trajectory with no retrieve action remains skipped without loading a model.

- [ ] **Step 3: Write failing `poisonedRAG` integration and privacy tests**

Use a fake generator that records prompts and returns five distinguishable values. Assert it is called exactly five times per question, with `max_length=1024`, `temperature=1.0`, `top_p=0.9`, and deterministic per-variant seeds; two retrieve steps reuse the five results to create ten uniquely identified documents. Assert secret step-query/document/state markers do not occur in recorded prompts.

- [ ] **Step 4: Run focused tests and verify failure**

Run: `python -m unittest tests.test_attack_interface -v`

Expected: FAIL because the new strategies are not configured or dispatched.

- [ ] **Step 5: Extend configuration validation**

Add the five values to `DOCUMENT_STRATEGIES`, define a named set for migrated target-only strategies, and reject their combination with any attack goal other than `targeted_hijacking` inside `load_experiment_config`.

- [ ] **Step 6: Add shared document assembly and deterministic RNG helpers**

In `rag/generate_attack_doc.py`, add a helper that maps five texts to the existing document shape using IDs derived from sanitized question ID, step number, method, and role. Seed a local RNG from `SEED`, question ID, step number, and method through SHA-256 rather than Python's randomized `hash()`.

- [ ] **Step 7: Dispatch static methods without loading a model**

Add prompt-version entries and route the four static strategies through `build_static_attack_texts`. Update generator-loading conditions so a run containing only these methods never calls `load_generator`. Make `--check-only` print their five sample documents.

- [ ] **Step 8: Dispatch `poisonedRAG` once per question**

Build its five original message lists from only `trajectory["question"]` and target, render them with the tokenizer, and call the generator five times using the reference `max_tokens=1024`, `temperature=1.0`, and `top_p=0.9`. Derive a stable seed per question and variant. Clean outputs once, then attach the same five texts to every real retrieve step using the shared assembler.

- [ ] **Step 9: Run focused tests**

Run: `python -m unittest tests.test_attack_interface -v`

Expected: PASS.

### Task 4: Update experiment documentation and run regression checks

**Files:**
- Modify: `doc/experiments.md:81-110`
- Test: `tests/test_attack_interface.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: final configuration and CLI behavior from Task 3.
- Produces: current examples for all five strategies and an explicit data-exposure statement.

- [ ] **Step 1: Replace stale generation commands**

Document the current `--config`, `--trajectories`, `--targets`, `--output-dir`, `--check-only`, backend, and resume interface. Add one YAML/CLI example showing that the selected method lives in `document_strategy` and requires `targeted_hijacking` plus per-question targets.

- [ ] **Step 2: Document method provenance and exposure boundary**

List the five migrated KidnapRAG baselines and state that their original prompts are preserved with dynamic target substitution. Explicitly state that `topicattack`/`paradox` are excluded and that `poisonedRAG` receives neither step queries nor step content.

- [ ] **Step 3: Run the complete short test suite**

Run: `python -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 4: Run syntax compilation**

Run: `python -m compileall -q rag tests`

Expected: exit status 0 with no output.

- [ ] **Step 5: Inspect the final diff-equivalent file list**

Because the project root has no Git metadata, list modified files explicitly and verify no files outside this plan changed.
