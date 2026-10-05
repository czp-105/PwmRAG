"""主实验变量与固定系统参数。"""

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .attack_methods import MIGRATED_ATTACK_METHODS


WORKFLOWS = {"deeprag", "comorag"}
DOCUMENT_STRATEGIES = {
    "kidnap_baseline", "evidence_progression",
} | MIGRATED_ATTACK_METHODS
ATTACK_GOALS = {"refusal", "targeted_hijacking", "incorrect_answer"}
TARGETED_GOALS = {"targeted_hijacking"}

# 主实验统一控制量：修改实验条件时只改这里，不写入外部 config。
RETRIEVER_MODEL = "intfloat/e5-large-v2"
TOP_K = 5
MAX_STEPS = 5
MAX_ITERATIONS = 5
MIN_PROBE_ITERATIONS = 1
VICTIM_MAX_NEW_TOKENS = 512
ATTACKER_MAX_NEW_TOKENS = 2048
DOCUMENTS_PER_STAGE = 5
TEMPERATURE = 0
SEED = 42
EXPECTED_QUESTIONS = 100
EVALUATION_VERSION = "paired_outcome_retrieval_v1"
ERROR_POLICY = "count_as_attack_failure"
COMORAG_MODE = "sequential_single_probe_accumulated_memory_v3"

CONFIG_KEYS = {
    "workflow", "document_strategy", "attack_goal",
    "victim_model", "victim_device", "attacker_model", "attacker_device",
}
KIDNAP_CONFIG_KEYS = CONFIG_KEYS | {"kidnap_chain_length"}


@dataclass(frozen=True)
class ExperimentConfig:
    workflow: str
    document_strategy: str
    attack_goal: str
    victim_model: str
    victim_device: str
    attacker_model: str
    attacker_device: str
    kidnap_chain_length: int | None = None

    def resolved(self) -> dict:
        """返回写入结果的完整实验条件。"""
        config = {
            **asdict(self),
            "retriever_model": RETRIEVER_MODEL,
            "top_k": TOP_K,
            "max_steps": MAX_STEPS,
            "max_iterations": MAX_ITERATIONS,
            "min_probe_iterations": MIN_PROBE_ITERATIONS,
            "victim_max_new_tokens": VICTIM_MAX_NEW_TOKENS,
            "attacker_max_new_tokens": ATTACKER_MAX_NEW_TOKENS,
            "documents_per_stage": DOCUMENTS_PER_STAGE,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "expected_questions": EXPECTED_QUESTIONS,
            "evaluation_version": EVALUATION_VERSION,
            "error_policy": ERROR_POLICY,
        }
        if self.workflow == "comorag":
            config["comorag_mode"] = COMORAG_MODE
        return config


def load_experiment_config(path: Path) -> ExperimentConfig:
    """读取主实验变量的 YAML config。"""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must be a YAML mapping")
    expected = (KIDNAP_CONFIG_KEYS if payload.get("document_strategy") == "kidnap_baseline"
                else CONFIG_KEYS)
    if set(payload) != expected:
        raise ValueError(f"config must contain exactly {sorted(expected)}")
    if payload["workflow"] not in WORKFLOWS:
        raise ValueError(f"workflow must be one of {sorted(WORKFLOWS)}")
    if payload["document_strategy"] not in DOCUMENT_STRATEGIES:
        raise ValueError(f"document_strategy must be one of {sorted(DOCUMENT_STRATEGIES)}")
    if payload["attack_goal"] not in ATTACK_GOALS:
        raise ValueError(f"attack_goal must be one of {sorted(ATTACK_GOALS)}")
    if (payload["document_strategy"] in MIGRATED_ATTACK_METHODS
            and payload["attack_goal"] != "targeted_hijacking"):
        raise ValueError(
            f"{payload['document_strategy']} requires attack_goal targeted_hijacking")
    if payload.get("kidnap_chain_length") is not None and (
            not isinstance(payload["kidnap_chain_length"], int)
            or isinstance(payload["kidnap_chain_length"], bool)
            or not 0 <= payload["kidnap_chain_length"] <= 4):
        raise ValueError("kidnap_chain_length must be an integer from 0 to 4")
    for key in ("victim_model", "victim_device", "attacker_model", "attacker_device"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
        payload[key] = payload[key].strip()
    return ExperimentConfig(**payload)


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
