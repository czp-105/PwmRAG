### 1

cd /home/zpc/projects/RAG/PwmRAG
conda activate pwmrag
./scripts/serve_vllm.sh victim

### 2

python run_rag_comorag.py \
  --work-dir experiments/2wikimultihopqa/expanded_100_full_v1/test20 \
  --model Qwen/Qwen3-8B \
  --backend vllm \
  --model-url http://127.0.0.1:8000/v1 \
  --workers 8 \
  --max-questions 20 \
  --top-k 5 \
  --max-iterations 2 \
  --max-new-tokens 1024 \
  --run-name qwen3_8b_vllm_test20

python run_rag_comorag.py \
  --work-dir experiments/2wikimultihopqa/expanded_100_full_v1 \
  --model Qwen/Qwen3-32B \
  --backend vllm \
  --model-url http://127.0.0.1:8000/v1 \
  --workers 10 \
  --max-questions 100 \
  --top-k 5 \
  --max-iterations 5 \
  --max-new-tokens 1024 \
  --run-name qwen3_32b_comorag_original_prompt_100

### 3 ComoRAG Kidnap

分别在两个终端启动 victim 和 attacker：

```bash
cd /home/zpc/projects/RAG/PwmRAG
conda activate bmt
CUDA_VISIBLE_DEVICES=0 ./scripts/serve_vllm.sh victim Qwen/Qwen3-8B
```

```bash
cd /home/zpc/projects/RAG/PwmRAG
conda activate bmt
CUDA_VISIBLE_DEVICES=1 ./scripts/serve_vllm.sh attacker Qwen/Qwen3-32B
```

导出 ComoRAG clean trajectory：

```bash
cd /home/zpc/projects/RAG/PwmRAG
conda activate pwmrag
python -m rag.attack trace \
  --config experiments/2wikimultihopqa/expanded_100_full_v1/configs/kidnap_comorag_qwen3_8b_32b.yaml \
  --clean-results experiments/2wikimultihopqa/expanded_100_full_v1/results/comorag/qwen3_8b_comorag_original_prompt_100.jsonl \
  --output experiments/2wikimultihopqa/expanded_100_full_v1/attacks/comorag_qwen3_8b_clean_trajectories.jsonl
```

运行攻击实验：

```bash

python -m rag.attack experiment \
  --config experiments/2wikimultihopqa/expanded_100_full_v1/configs/kidnap_comorag_qwen3_8b_32b.yaml \
  --trajectories experiments/2wikimultihopqa/expanded_100_full_v1/attacks/comorag_qwen3_8b_clean_trajectories.jsonl \
  --targets experiments/2wikimultihopqa/expanded_100_full_v1/attacks/full100_v1/targets.jsonl \
  --output-dir experiments/2wikimultihopqa/expanded_100_full_v1/attacks/kidnap_comorag_qwen3_8b_32b_v1 \
  --backend vllm \
  --victim-url http://127.0.0.1:8000/v1 \
  --attacker-url http://127.0.0.1:8001/v1 \
  --workers 8 \
  --device cuda:0 \
  --run-name paired_kidnap_comorag_v1 \
  --resume
```

cd /home/zpc/projects/RAG/PwmRAG
./static/test_all_baselines.sh
