import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class BaselineScriptTest(unittest.TestCase):
    def test_target_answer_replaces_missing_targets_file(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            config = work / "config.yaml"
            config.write_text("document_strategy: poisonedRAG\n", encoding="utf-8")
            trajectories = work / "trajectories.jsonl"
            trajectories.write_text(json.dumps({"question_id": "q1"}) + "\n",
                                    encoding="utf-8")
            result = subprocess.run(
                ["bash", str(root / "static/test_all_baselines.sh"), "--dry-run"],
                cwd=root,
                env={**os.environ, "BASE_DIR": str(work),
                     "CONFIG_TEMPLATE": str(config), "TRAJECTORIES": str(trajectories),
                     "TARGET_ANSWER": "I don't know"},
                text=True, capture_output=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("python -m rag.attack experiment"), 6)


if __name__ == "__main__":
    unittest.main()
