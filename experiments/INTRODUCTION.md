# Experiments

本目录保存 PwmRAG 的可复现实验数据、检索索引、clean 推理结果和投毒攻击结果。当前主要 victim 是 ComoRAG，研究目标是评估针对 multi-step RAG 的定向回答劫持与文档投毒。

## 实验概览

| 目录 | 数据集 | 问题数 | 语料文档数 | 主要内容 |
| --- | --- | ---: | ---: | --- |
| `2wikimultihopqa/expanded_100_full_v1/` | 2WikiMultiHopQA | 100 | 6,119 | clean baseline、Kidnap 和五种基础攻击方法 |
| `hotpotqa/expanded_100_full_v1/` | HotpotQA | 100 | 9,811 | 不同 victim/target/ComoRAG 模式下的完整 baseline 对比 |

数据集均使用固定随机种子 `42`。`expanded_100_full_v1` 表示选择 100 个问题，并保留支持文档和扩展的随机干扰文档。

## 单个实验目录

每个实验通常包含以下目录：

- `data/`：实验输入。
  - `queries.jsonl`：问题。
  - `answers.jsonl`：标准答案。
  - `qrels.jsonl`：问题与支持文档的对应关系。
  - `corpus.jsonl`：检索语料。
  - `manifest.json`：数据来源、样本数、随机种子和构造方式。
- `index/`：clean 语料的 E5 检索索引及其 manifest。
- `configs/`：保留的实验配置。
- `results/`：未投毒的 clean 推理结果。
- `attacks/`：攻击文档、投毒索引、逐样本攻击结果和汇总指标。
- `test.md`：该实验使用过的运行命令和操作记录。

## Clean 结果

Clean 结果位于各实验的 `results/` 下：

- `*.jsonl`：逐问题输出和推理记录。
- `*.summary.json`：汇总指标。
- `results/comorag/`：ComoRAG victim 的结果。

主要位置：

- `2wikimultihopqa/expanded_100_full_v1/results/`
- `hotpotqa/expanded_100_full_v1/results/comorag/`

## 攻击方法与结果

当前出现的基础攻击方法包括：

- `naive`
- `ignore`
- `completion_real`
- `completion_realcmb`
- `poisonedRAG`
- `kidnap`

此外还保留了早期自研方法 `fmp_frontier` 的实验结果。

一次完整攻击结果通常位于：

```text
attacks/<实验组>/<方法>/
```

其中：

- `config.yaml`：victim、attacker、检索器、步数和随机种子。
- `manifest.json`：实际生成数、错误数、攻击目标和输出位置。
- `generation_results.jsonl`：为每个问题生成的投毒文档。
- `shared_poison_corpus.jsonl`：加入攻击文档后的共享语料。
- `shared_poison_index/`：投毒语料对应的检索索引。
- `paired_<method>_comorag_v1.jsonl`：clean 与 poisoned 的逐样本配对结果。
- `paired_<method>_comorag_v1.summary.json`：攻击成功率、检索命中和回答变化等汇总指标。

查看攻击效果时，应优先读取 `paired_*.summary.json`；分析具体成功或失败案例时，再读取对应的 `paired_*.jsonl`。

## 各数据集的攻击实验

### 2WikiMultiHopQA

正式实验位于 `2wikimultihopqa/expanded_100_full_v1/`。

- victim：主要为 `Qwen/Qwen3-8B` ComoRAG。
- attacker：主要为 `Qwen/Qwen3-32B`。
- 基础方法结果：`attacks/baselines_comorag_qwen3_8b_32b_v1/`。
- Kidnap 完整结果：`attacks/kidnap_comorag_qwen3_8b_32b_v1/`。
- clean trajectory：`attacks/comorag_qwen3_8b_clean_trajectories.jsonl`。

该数据集中实际进入 ComoRAG 攻击评估的有效样本通常为 99 条，具体以各方法的 `manifest.json` 为准。

### HotpotQA

正式实验位于 `hotpotqa/expanded_100_full_v1/`，共 100 条问题。

主要实验组：

- `baselines_comorag_qwen3_8b_32b_v1/`：8B victim、32B attacker 的初始 target 实验。
- `baselines_comorag_qwen3_8b_32b_fuck_you_v1/`：固定字符串 target 实验。
- `baselines_comorag_qwen3_32b_32b_idk_v1/`：32B victim 的拒答目标实验。
- `baselines_comorag_qwen3_32b_32b_sequential_v3/`：32B victim 与 sequential accumulated-memory ComoRAG。
- `poisonedrag_comorag_qwen3_8b_32b_fuck_you_v1/`：PoisonedRAG 单方法复现实验。
- `fmp_frontier_fuck_you_v1/`：FMP frontier 探索实验。

baseline 实验组通常同时包含六种方法：`naive`、`ignore`、`completion_real`、`completion_realcmb`、`poisonedRAG` 和 `kidnap`。

## 阅读结果的建议顺序

1. 查看 `data/manifest.json`，确认数据规模和来源。
2. 查看 `results/**/*.summary.json`，确认 clean 性能。
3. 查看攻击方法的 `manifest.json`，确认实际运行样本数和错误数。
4. 查看 `paired_*.summary.json`，比较攻击指标。
5. 查看 `paired_*.jsonl`，分析具体成功、失败和检索传播案例。
