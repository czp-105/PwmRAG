# multi-step RAG 

# 多步检索与工作记忆的知识投毒实验

## 研究问题与威胁模型

研究静态毒知识能否影响多步 RAG 对**证据进展与充分性**的判断：模型何时继续检索、何时认为可以回答，以及如何利用已保存的中间信息。区别于 KidnapRAG 以诱导特定查询链为主的设计，这里关注检索信息进入中间答案或工作记忆后，对后续检索、停止和答案的持续影响。这是待验证的机制假设，不以给文档角色换名作为创新点。

采用与 KidnapRAG 相同的定向、黑盒文档投毒场景：攻击者知道目标问题及希望诱导的结果，只能在问题运行前发布少量可检索的毒文档。毒文档在该次问答的所有轮次中持续可检索；攻击者不能改用户问题、模型、提示词、检索器、工作记忆，也不能在运行中按轮切换文档或查看内部状态。研究者可记录轨迹用于评估，但这些信息不是攻击者的运行时权限。逐题独立构建语料是评测隔离，不是动态激活毒文档。

## 攻击目的

- **拒绝回答／无法完成**：使原本可回答的问题最终明确拒答、无可用答案，或耗尽检索预算。显式拒答、无 probe、预算耗尽分别统计；ComoRAG 中途的 `*` 只表示继续检索，不算最终拒答。
- **定向回答劫持**：使最终答案成为预先指定的错误目标；特别关注证据尚不足时的过早停止。
- **非定向错误结果**：最终答案偏离正确答案，但不要求命中指定目标；与定向成功分开报告。

候选机制是“虚假未完成”（有证据却持续检索或无法作答）、“虚假完成”（过早相信证据链闭合）和“错误信息累积”（中间信息影响后续查询及最终答案）。这些是结果与轨迹的解释框架，不能仅凭毒文档命中或记忆来源标签就断言因果机制。

## 实验对照与指标

先分别获得 DeepRAG 和简化 ComoRAG 的**干净基线**；DeepRAG 的结果不能充当 ComoRAG 基线。对预先选定的目标题，使用相同模型、干净语料、检索配置、步数预算和解码设置，分别运行 clean 与 poisoned 条件。毒文档在 poisoned 条件中预先加入语料，整次运行保持不变；不采用“首次检索后移除文档”或“直接修改记忆”的攻击设置。先用少量题调试，再用预先固定的筛题规则评估，避免只报告成功样例。

主要结果按**全部预先确定的目标题**计算：拒答 ASR、指定目标答案命中 ASR，以及 clean 正确样本上的错误答案 ASR；同时报告 clean/poisoned 的规范化 EM、平均 token F1 及其下降幅度。原始答案继续保留以便复核，运行错误不计为攻击成功。

检索暴露指标参考 KidnapRAG：题级毒文档命中率、检索动作级毒文档命中率、全部 top-k 结果中的毒文档占比，以及首次命中轮次与排名。这些指标只说明毒文档被检索，不自动等同于最终攻击成功。当前主评估不汇总查询变化、记忆编码、支持文档覆盖或停止原因；完整 workflow state 仍保存在原始结果中。

DeepRAG 会把原文与 `Intermediate answer` 一同保留在后续提示词；简化 ComoRAG 在后续回答中主要使用 cue 与融合信息。因此两者可作为不同上下文保留机制的比较，但不应把跨系统差异直接归因于工作记忆，模型与基线能力等条件也须明确报告。


## 创新点
首先，这项工作揭示了多步 RAG 特有的安全风险：更复杂的检索和记忆机制不一定更安全，反而可能为攻击者提供持续影响系统状态的通道。
1. 面向证据充分性判断的攻击
2. Evidence Progression 攻击机制
3. 面向工作记忆的持续性投毒


## method
1. Target Query Profiling：目标问题分析
2. Poisoned Evidence Construction：构造“证据缺口型”投毒知识
① Evidence Conflict：制造证据冲突
② Evidence Dependency：制造额外验证条件
③ Verification Direction：植入下一步验证方向
3. Sufficiency Disruption → Retrieval Steering：从“证据不足”转化成检索偏转
4. Multi-Round Poisoning → Attack Outcome：利用后续检索实现三类结果
A. Refusal：持续制造“证据仍然不足”
B. Targeted Hijacking：先制造不足，再让攻击者证据完成闭环
C. Incorrect Answer：把补证据过程引入错误知识空间


## 2WikiMultiHopQA 真实检索 smoke test

`run_rag_smoke.py` 按 `--dataset` 从源数据中固定抽取两跳题、对应支持文档和随机干扰文档；产物写入对应的 `experiments/<dataset>/`，源数据目录不修改。已有 2Wiki 样本为 20 题、1000 篇文档；该结果只说明链路可运行，不代表完整数据集性能。

首次建立新实验目录时运行（将 GPU 编号改为当时空闲的编号）：

```bash
python run_rag_smoke.py prepare --dataset 2wikimultihopqa --questions 20 --corpus-size 1000 --seed 42
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 python run_rag_smoke.py index --dataset 2wikimultihopqa --device cuda
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 python run_rag_smoke.py run --dataset 2wikimultihopqa --question-id 931a85420bdd11eba7f7acde48001122 --max-questions 1 --run-name retrieval_check
```

当前目录已经准备并建立索引，因此再次测试只需执行 `run`，并换一个未使用的 `--run-name`；重新抽样时使用新的 `--work-dir`。working memory 与投毒尚未接入。

`baseline_20.jsonl` 的干净基线已完成。运行 `python select_rag_baseline.py --dataset 2wikimultihopqa --run-name baseline_20` 会生成机械筛选结果；20 题中 5 题同时满足答对、至少两次检索和两篇支持文档均命中。逐步人工复核后，`results/baseline_20.shortlist.jsonl` 保留 2 个关系链较可信的首批目标，原因见 `results/baseline_20.manual_review.md`。此规模仅用于后续攻击流程调试，不能作为成功率估计。

`expanded_100_full_v1/results/baseline_100.jsonl` 是另一组 DeepRAG 干净基线：100 题中严格匹配 46 题、至少两次真实检索 38 题、两篇支持文档均出现 44 题；19 题同时满足这三项，但其中仅 6 题还满足“无参数知识步骤，且第二次检索带来新的支持文档”。这些机械条件不保证推理链正确，仍需人工复核；该结果也不是 ComoRAG 的攻击前基线。

## ComoRAG + 本地 Qwen3-8B 干净 smoke test

`run_rag_comorag.py` 复用所选数据集 `expanded_100_full_v1` 的数据与 E5 索引，不重建语料或索引；结果单独写入该目录的 `results/comorag/`，不会覆盖 DeepRAG 基线。

在项目根目录运行（按实际空闲 GPU 调整 `CUDA_VISIBLE_DEVICES`；使用包含 `torch`、`transformers`、`zstandard` 的环境）：

```bash
python run_rag_comorag.py --dataset 2wikimultihopqa --model-path /home/zpc/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 --question-id a8d938b80bda11eba7f7acde48001122 --run-name qwen3_8b_smoke_a8_tags_v4 --check-only
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python run_rag_comorag.py --dataset 2wikimultihopqa --model-path /home/zpc/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 --question-id a8d938b80bda11eba7f7acde48001122 --run-name qwen3_8b_smoke_a8_tags_v4
```

`qwen3_8b_smoke_a8_tags_v4` 已完成：首次回答 `*`，继续检索后得到 `Württemberg Mausoleum.`；两篇支持文档均被检索到。模型沿用简洁的 `### Final Answer` 协议：证据不足输出 `*`，否则标题后的第一行是短答案；适配器再将标题前的解释与短答案整理为 `<answer long>`、`<answer short>`，供轨迹记录与评测抽取，不要求模型直接生成标签。若模型输出省略号占位符，则按证据不足继续检索。后续 ComoRAG 结果的 `exact_match` 直接采用 DeepRAG/ComoRAG 原评测器的去冠词、标点与空白规范化，不再另记 `normalized_exact_match`。已生成的 v4 及更早结果保留当时的严格 `exact_match`，不会改写；与它们或旧 DeepRAG smoke 结果比较时需注意口径差异。

## 静态文档投毒接口（DeepRAG / ComoRAG）

实验分为两个入口：`rag.attack trace` 从真实 clean 结果导出统一轨迹，
`rag.generate_attack_doc` 再生成攻击案例；`rag.attack experiment` 可串联文档生成、
共享毒索引、clean/poisoned 配对运行和汇总。没有真实检索步骤的轨迹会被跳过。

除 `kidnap_baseline` 和 `evidence_progression` 外，当前还支持从 KidnapRAG 原脚本迁移的
`naive`、`ignore`、`completion_real`、`completion_realcmb` 和 `poisonedRAG`。
这五种基线保留原 prompt，只将硬编码答案替换为每题 `target_answer`，因此仅允许
`attack_goal: targeted_hijacking`。前四种为本地模板，不加载攻击者模型；
`poisonedRAG` 只向攻击者模型提供原问题和目标答案，不提供步骤 query、
`retrieved_documents`、`state_update`、memory cue 或 intermediate answer。因需要暴露逐步
检索内容，`topicattack` 和 `paradox` 不在本实验实现。

`generation_results.jsonl` 每行记录状态；成功项的 `case` 字段直接包含 `question_id`、可选的 `target_answer` 和非空 `documents`（每篇有 `_id/title/text`），不再创建逐题目录。攻击者可以离线观察 clean 轨迹，但在线阶段只能向检索库置入文档，不能修改 query、memory 或 workflow。单独的 `case.json` 只保留给 `index/run` 调试入口。

五种迁移基线的 config 使用普通主配置字段（不包含 `kidnap_chain_length`），例如：

```yaml
workflow: deeprag
document_strategy: poisonedRAG  # 或 naive/ignore/completion_real/completion_realcmb
attack_goal: targeted_hijacking
victim_model: xinyan233333/DeepRAG-7b
victim_device: cuda:0
attacker_model: Qwen/Qwen3-8B
attacker_device: cuda:1
```

在项目根目录运行；以下路径需按实验目录替换：

```bash
# 1. 从真实 clean 结果提取轨迹
python -m rag.attack trace --config experiments/.../config.yaml --clean-results experiments/.../clean.jsonl --output experiments/.../clean_trajectories.jsonl

# 2. 检查 prompt/静态文档，然后生成全部案例
python -m rag.generate_attack_doc --config experiments/.../config.yaml --trajectories experiments/.../clean_trajectories.jsonl --targets experiments/.../targets.jsonl --output-dir experiments/.../naive --check-only
python -m rag.generate_attack_doc --config experiments/.../config.yaml --trajectories experiments/.../clean_trajectories.jsonl --targets experiments/.../targets.jsonl --output-dir experiments/.../naive

# 3. 或者一次执行生成、共享索引、配对运行与汇总
python -m rag.attack experiment --config experiments/.../config.yaml --trajectories experiments/.../clean_trajectories.jsonl --targets experiments/.../targets.jsonl --output-dir experiments/.../naive --run-name paired --check-only
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 python -m rag.attack experiment --config experiments/.../config.yaml --trajectories experiments/.../clean_trajectories.jsonl --targets experiments/.../targets.jsonl --output-dir experiments/.../naive --run-name paired
```

完整 `experiment` 会像 KidnapRAG 一样，将全部案例文档合并到 `shared_poison_corpus.jsonl`，只构建一次 `shared_poison_index/`，并让所有问题复用固定的 clean/poisoned 检索器；不会逐题或逐轮动态插入文档。运行结果逐题追加到一个 `<run-name>.jsonl`，最终只写一份 `<run-name>.summary.json`。单题 `index/run` 仅用于调试。不同 workflow、攻击方法或配置不得混合汇总。


### 攻击目标
破坏这种多轮检索、带工作记忆的 rag 机制的可用性，结果分为“拒绝回答“，“定向劫持“，“结果错误“
### 攻击方法
误导这个机制的 rag 对““证据是否足够”和“记忆中保存了什么”“的判断，然后后面我们通过一些具体的流程说清楚这个点
