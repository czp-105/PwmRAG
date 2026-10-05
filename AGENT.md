# 项目协作规范

1. 对于耗时较长的实验任务（如模型下载、索引构建、批量推理或评测），不要代替用户运行或持续等待。给出可直接复制的运行指令、工作目录和预期输出位置，由用户自行执行并反馈结果。短时的代码检查与测试可以直接运行。

2. 攻击实验分为两个明确入口：`rag/attack.py` 负责 `trace`（保存真实 clean 轨迹）、`index`（准备毒索引）、`run`（运行单题 clean/poisoned 并评估）和 `eval`（跨题汇总）；`rag/generate_attack_doc.py` 单独读取已保存轨迹并批量生成攻击案例。不要为每个 RAG workflow 复制攻击或生成脚本，具体 workflow 通过数据字段和参数适配。

3. 参照 KidnapRAG `attack_react.py` 的实验主线组织代码：先明确数据、生成模型、检索模型、设备、检索深度及迭代预算等系统参数，再启动模型和检索器、接入毒文档、执行问答、保存轨迹并计算指标。不要照搬其顶层加载模型、硬编码路径或覆盖结果文件的做法；模型等耗时资源只在实际运行阶段加载。

4. clean 与 poisoned 必须使用同一题目、模型和推理参数，唯一区别是 poisoned 检索器额外加载经校验的毒索引。每次运行应保留 workflow、参数、案例/语料哈希、答案、错误和检索轨迹；不同 workflow、攻击方法或配置的结果不得混合汇总。

5. 攻击代码采用简短中文注释，按“参数与数据 → 系统初始化 → 攻击接入/运行 → 评估与保存”标出阶段；注释解释实验假设、攻击边界和指标含义，不逐行复述代码。区分毒文档被检索、进入记忆、最终回答被影响，不能把前一阶段自动当成攻击成功。

6. 攻击文档必须基于受害系统实际 clean 运行产生的 trajectory，不把模型模拟的 query 或 memory 当作真实轨迹。统一轨迹至少保存原问题、每步 action/query、检索文档、状态更新、观察到的下一动作/查询、停止原因和最终答案；DeepRAG 的状态更新对应 `intermediate_answer`，ComoRAG 对应 `memory cue`。

7. 文档生成参考 KidnapRAG，输入固定为每步真实 `search_query` 与对应的 `search_intent`（本项目使用该步记录的 `intermediate_answer` 或 `memory cue` 作为可观察推理内容），攻击方法和目标答案只作为生成约束。第一版 few-shot 输出按 `search`、`memory`、`judge`、`output` 四种作用标记，所有文档标题固定使用来源 query，并记录来源步骤。examples 与 prompt 集中放在 `rag/generate_attack_doc.py`，生成器一次加载模型并统一处理已保存轨迹；完整案例逐题追加到同一个 `generation_results.jsonl`，不创建逐题目录。没有真实检索步骤的题目应跳过。
