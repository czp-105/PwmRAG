#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset="${DATASET:-2wikimultihopqa}"
experiment="${EXPERIMENT:-expanded_100_full_v1}"
base="${BASE_DIR:-$root/experiments/$dataset/$experiment}"
trajectories="${TRAJECTORIES:-$base/attacks/comorag_qwen3_8b_clean_trajectories.jsonl}"
output_root="${OUTPUT_ROOT:-$base/attacks/baselines_comorag_qwen3_8b_32b_v1}"
methods=(naive ignore completion_real completion_realcmb poisonedRAG kidnap)
dry_run=false

template="${CONFIG_TEMPLATE:-}"
if [[ -z "$template" ]]; then
  for candidate in \
    "$base/configs/poisonedrag_comorag_qwen3_8b_32b.yaml" \
    "$base/configs/kidnap_comorag_qwen3_8b_32b.yaml"; do
    [[ -f "$candidate" ]] && { template="$candidate"; break; }
  done
fi
[[ -n "$template" ]] || {
  echo "no config template found under $base/configs; set CONFIG_TEMPLATE" >&2
  exit 1
}

targets="${TARGETS:-$base/attacks/targets.jsonl}"
if [[ -z "${TARGETS:-}" && ! -f "$targets" ]]; then
  targets="$base/attacks/full100_v1/targets.jsonl"
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi

for path in "$template" "$trajectories"; do
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 1; }
done

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
if [[ -n "${TARGET_ANSWER:-}" ]]; then
  targets="$tmp/targets.jsonl"
  python - "$trajectories" "$targets" "$TARGET_ANSWER" <<'PY'
import json
import sys

source, output, target = sys.argv[1:]
with open(source, encoding="utf-8") as rows, open(output, "w", encoding="utf-8") as results:
    for line in rows:
        row = json.loads(line)
        results.write(json.dumps({"question_id": row["question_id"],
                                  "target_answer": target}, ensure_ascii=False) + "\n")
PY
fi
[[ -f "$targets" ]] || { echo "missing required file: $targets" >&2; exit 1; }
cd "$root"

run() {
  printf '%q ' "$@"
  printf '\n'
  "$dry_run" || "$@"
}

for method in "${methods[@]}"; do
  strategy="$method"
  [[ "$method" == "kidnap" ]] && strategy="kidnap_baseline"
  config="$tmp/$method.yaml"
  awk -v method="$strategy" -v chain="${KIDNAP_CHAIN_LENGTH:-2}" '
    /^document_strategy:/ { print "document_strategy: " method; next }
    /^kidnap_chain_length:/ {
      if (method == "kidnap_baseline") print "kidnap_chain_length: " chain
      found_chain = 1
      next
    }
    { print }
    END {
      if (method == "kidnap_baseline" && !found_chain)
        print "kidnap_chain_length: " chain
    }
  ' "$template" > "$config"

  run python -m rag.attack experiment \
    --config "$config" \
    --work-dir "$base" \
    --trajectories "$trajectories" \
    --targets "$targets" \
    --output-dir "$output_root/$method" \
    --backend vllm \
    --victim-url "${VICTIM_URL:-http://127.0.0.1:8000/v1}" \
    --attacker-url "${ATTACKER_URL:-http://127.0.0.1:8001/v1}" \
    --workers "${WORKERS:-8}" \
    --device "${DEVICE:-cuda:0}" \
    --run-name "paired_${method}_comorag_v1" \
    --resume
done
