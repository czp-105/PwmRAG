# HotpotQA PoisonedRAG / ComoRAG

项目已按 seed 42 准备 100 道题，并保留完整的 9,811 篇语料。

## 1. 构建 E5 索引

```bash
cd /home/zpc/projects/RAG/PwmRAG
conda activate pwmrag
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python run_rag_smoke.py index \
  --dataset hotpotqa \
  --work-dir experiments/hotpotqa/expanded_100_full_v1 \
  --device cuda
```

## 2. 运行 ComoRAG clean baseline

先启动 victim：

```bash
cd /home/zpc/projects/RAG/PwmRAG
conda activate bmt
CUDA_VISIBLE_DEVICES=0 ./scripts/serve_vllm.sh victim Qwen/Qwen3-8B
```

再运行 100 题：

```bash
cd /home/zpc/projects/RAG/PwmRAG
conda activate pwmrag
python run_rag_comorag.py \
  --dataset hotpotqa \
  --work-dir experiments/hotpotqa/expanded_100_full_v1 \
  --model Qwen/Qwen3-8B \
  --backend vllm \
  --model-url http://127.0.0.1:8000/v1 \
  --workers 8 \
  --max-questions 100 \
  --top-k 5 \
  --max-iterations 5 \
  --max-new-tokens 512 \
  --run-name qwen3_8b_comorag_clean_100
```

## 3. 导出 clean trajectory

```bash
python -m rag.attack trace \
  --config experiments/hotpotqa/expanded_100_full_v1/configs/poisonedrag_comorag_qwen3_8b_32b.yaml \
  --clean-results experiments/hotpotqa/expanded_100_full_v1/results/comorag/qwen3_8b_comorag_clean_100.jsonl \
  --output experiments/hotpotqa/expanded_100_full_v1/attacks/comorag_qwen3_8b_clean_trajectories.jsonl
```

## 4. 运行 PoisonedRAG

另一个终端启动 attacker：

```bash
cd /home/zpc/projects/RAG/PwmRAG
conda activate bmt
CUDA_VISIBLE_DEVICES=1 ./scripts/serve_vllm.sh attacker Qwen/Qwen3-32B
```

先检查输入：

```bash
python -m rag.attack experiment \
  --work-dir experiments/hotpotqa/expanded_100_full_v1 \
  --config experiments/hotpotqa/expanded_100_full_v1/configs/poisonedrag_comorag_qwen3_8b_32b.yaml \
  --trajectories experiments/hotpotqa/expanded_100_full_v1/attacks/comorag_qwen3_8b_clean_trajectories.jsonl \
  --targets experiments/hotpotqa/expanded_100_full_v1/attacks/targets.jsonl \
  --output-dir experiments/hotpotqa/expanded_100_full_v1/attacks/poisonedrag_comorag_qwen3_8b_32b_v1 \
  --backend vllm \
  --check-only
```

检查通过后运行：

```bash
python -m rag.attack experiment \
  --work-dir experiments/hotpotqa/expanded_100_full_v1 \
  --config experiments/hotpotqa/expanded_100_full_v1/configs/poisonedrag_comorag_qwen3_8b_32b.yaml \
  --trajectories experiments/hotpotqa/expanded_100_full_v1/attacks/comorag_qwen3_8b_clean_trajectories.jsonl \
  --targets experiments/hotpotqa/expanded_100_full_v1/attacks/targets.jsonl \
  --output-dir experiments/hotpotqa/expanded_100_full_v1/attacks/poisonedrag_comorag_qwen3_8b_32b_v1 \
  --backend vllm \
  --victim-url http://127.0.0.1:8000/v1 \
  --attacker-url http://127.0.0.1:8001/v1 \
  --workers 8 \
  --device cuda:0 \
  --run-name paired_poisonedrag_comorag_v1 \
  --resume
```

## 5. 运行 HotpotQA 全部 baseline

```bash
cd /home/zpc/projects/RAG/PwmRAG
conda activate pwmrag
DATASET=hotpotqa EXPERIMENT=expanded_100_full_v1 \
  ./static/test_all_baselines.sh
```

```bash
BASE_DIR=/home/zpc/projects/RAG/PwmRAG/experiments/hotpotqa/expanded_100_full_v1 \
CONFIG_TEMPLATE=/home/zpc/projects/RAG/PwmRAG/experiments/hotpotqa/expanded_100_full_v1/configs/poisonedrag_comorag_qwen3_32b_32b.yaml \
TRAJECTORIES=/home/zpc/projects/RAG/PwmRAG/experiments/hotpotqa/expanded_100_full_v1/attacks/comorag_qwen3_32b_sequential_v3_clean_trajectories.jsonl \
TARGETS=/home/zpc/projects/RAG/PwmRAG/experiments/hotpotqa/expanded_100_full_v1/attacks/targets.jsonl \
OUTPUT_ROOT=/home/zpc/projects/RAG/PwmRAG/experiments/hotpotqa/expanded_100_full_v1/attacks/baselines_comorag_qwen3_32b_32b_sequential_v3 \
VICTIM_URL=http://127.0.0.1:8001/v1 \
ATTACKER_URL=http://127.0.0.1:8001/v1 \
WORKERS=8 \
DEVICE=cuda:0 \
KIDNAP_CHAIN_LENGTH=0 \
./static/test_all_baselines.sh
```