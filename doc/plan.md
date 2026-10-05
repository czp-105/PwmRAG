## 当前实现

`rag/` 是多步 RAG 实验的实现目录，而不是 DeepRAG 的专属目录。`model.py` 放模型协议适配；`workflow.py` 分别封装 `run_deeprag` 和 `run_comorag` 两套控制流程，旧入口 `run` 保留为 DeepRAG 别名。两套流程的状态放在 `state.py`，ComoRAG 的记忆池只在对应流程中使用。

`rag/state.py` 保存共享文档类型及两套流程各自的状态；DeepRAG 状态包含子查询、动作、文档和中间答案，ComoRAG 状态包含检索尝试及临时/主记忆池。`rag/workflow.py` 分别实现 DeepRAG 决策循环和 ComoRAG 的 `*`→probe→融合循环。`rag/retriever.py` 薄适配现有 E5，保留文档 ID、顺序、分数及投毒来源标签，加载前检查语料和索引文件。短测试用模拟模型和检索器检查控制流，不需 GPU。

`rag/model.py` 分别提供 DeepRAG 与 ComoRAG 模型适配器。DeepRAG 使用 `Follow up:`/检索触发语/`Intermediate answer:` 协议；例如 `model = load_deeprag("xinyan233333/DeepRAG-7b")` 后，调用 `run_deeprag(question, model.decide, retriever.search, model.answer_step, model.answer_final)`。当前 ComoRAG 实验固定为 E5 原文检索 → 临时记忆编码 → 回答或 `*` → 提交记忆、生成 probe → 再检索、融合记忆；调用 `run_comorag(question, retriever.search, model.encode, model.answer, model.make_probes, model.fuse)`。记忆节点保留文档 ID 和投毒来源。语义摘要、时间线与 OpenIE 图索引不纳入当前实验；如需比较它们的影响，再作为独立实验条件增加，不能把当前实现称为完整 ComoRAG 复现。

已通过 `run_rag_smoke.py` 为可选数据集建立独立 E5 索引并运行真实检索；数据、索引、结果目录及复现命令见 `doc/experiments.md`。ComoRAG 的真实模型运行与投毒攻击尚未测试；DeepRAG 的训练与树搜索不在本项目当前范围内。

`run_rag_comorag.py` 已提供独立的本地 Qwen3-8B 干净 smoke 入口，复用所选数据集和 E5 索引，使用单独的 ComoRAG 结果目录；目前只完成无模型输入校验及短时单测，真实推理结果仍待用户运行。

当前定向黑盒文档投毒的威胁模型、攻击目标、clean/poisoned 对照及指标定义见 `doc/experiments.md`。文档中的实验设计尚未全部实现，不应与现有 `rag/` 组件的完成状态混同。
