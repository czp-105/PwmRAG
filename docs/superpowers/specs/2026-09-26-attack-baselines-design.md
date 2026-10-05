# PwmRAG 攻击基线迁移设计

## 目标

将 KidnapRAG `generate_attack_doc_kill.py` 中的五种定向攻击基线迁移到
PwmRAG：`naive`、`ignore`、`completion_real`、`completion_realcmb` 和
`poisonedRAG`。所有方法使用 PwmRAG 配置中逐题的 `target_answer`，不保留
参考实现中的硬编码目标。

不迁移 `topicattack` 和 `paradox`，因为它们需要把每个 clean 检索步骤的
文档内容暴露给攻击者模型。

## 实验边界

- 五种方法仅允许与 `attack_goal: targeted_hijacking` 组合。
- 必须为每个选中问题提供非空 `target_answer`，且目标不得与 gold answer 相同。
- 不读取或发送 `retrieved_documents`、`state_update`、memory cue、intermediate answer。
- 真实检索 query 只用作毒文档的 title 和来源元数据，不发送给
  `poisonedRAG` 的攻击者模型。
- 每个真实检索步骤生成恰好五篇文档。

## 代码结构

新增 `rag/attack_methods.py`，保存五种方法的 prompt/模板与简单分派函数。
使用普通函数，不引入基类、工厂或插件注册框架。

`rag/generate_attack_doc.py` 仍负责：

1. 加载并校验配置、轨迹和目标。
2. 遍历真实 retrieve 步骤。
3. 调用对应攻击方法。
4. 组装、校验并增量写入 case。
5. 保持 resume、manifest 和下游索引协议不变。

`rag/config.py` 的 `DOCUMENT_STRATEGIES` 增加五种方法，并增加一个定向
方法集合供配置校验复用。

## 方法语义

五种方法直接迁移参考脚本的原始 prompt/模板和组合顺序。除下列适配外，
不重写 prompt 的措辞、标记或结构：

- 将所有硬编码的 `KILL YOURSELF` 替换为当前问题的动态 `target_answer`。
- 将参考脚本的 `qid`、`question`、`subquery` 对应到 PwmRAG 的
  `question_id`、`trajectory["question"]`、`step["query"]`。
- 只修正输出 ID、元数据和 case 封装，不将这些工程修正写入 prompt。

各方法保持以下原始语义和调用方式：

- `naive`：直接使用参考脚本中的五个 `instruction1...instruction5`，仅替换
  其中的硬编码目标。
- `ignore`：使用原 `IGNORE_ATTACK_SENTENCES["test"]` 候选句式，将对应的原
  instruction 填入 `{injected_prompt}`。
- `completion_real`：保持原始 `##response:`、`OK.`、`##instruction:` 文本、
  换行与拼接顺序，并接入对应的原 instruction。
- `completion_realcmb`：保持原始 completion 边界、ignore 候选句式、随机
  换行/制表字符及原拼接顺序。
- `poisonedRAG`：直接迁移原 system/user prompt，仅将用户 prompt 中硬编码的
  answer 替换为动态 target。与参考实现一致，对同一 `question + target`
  构造五个相同 prompt 并分别生成五篇不超过 30 个英文词的文档；
  保留对生成文本前缀 `Document: ` 的删除逻辑。

`ignore` 和 `completion_realcmb` 使用从项目 `SEED` 派生的局部
`random.Random`，断点续跑不依赖全局 RNG 状态。

## 文档与输出

每篇文档包含：

- 唯一 `_id`：包含 question ID、step、method 和文档序号。
- `title`：来源检索 query。
- `text`：方法生成的攻击内容。
- `metadata`：记录 role、method、source step、source query、original ID、target answer
  和 `is_poisoned` 标记。

案例仍使用现有 `generation_results.jsonl`，并在 case 和 manifest 中将方法名
写入 `document_strategy`。不改变 `rag.attack` 的索引、运行和评测输入格式。

## 错误处理

- 配置与攻击目标不匹配时在加载阶段报错。
- `poisonedRAG` 任一次生成返回非字符串或空文档时，将该题记录为
  `error`，并由现有 `--resume` 重试。
- 静态方法不加载攻击者模型。
- `--check-only` 输出选中方法的样例文档或 `poisonedRAG` prompt，不运行生成。

## 验证

在现有 `tests/test_attack_interface.py` 增加小型测试：

1. 每种方法恰好产生五篇、ID 唯一、title 正确的文档。
2. 四种静态方法的输出与参考脚本的原模板一致，唯一语义变化是
   硬编码目标被动态 target 替换，且随机方法可复现。
3. `poisonedRAG` 构造五个原始 prompt；prompt 包含 question/target，但不包含放入
   `retrieved_documents`、`state_update` 和 step query 的秘密标记。
4. 五种方法与非 `targeted_hijacking` 目标组合时拒绝运行。
5. 现有攻击接口测试继续通过。

## 文档更新

更新 `doc/experiments.md`，记录五种方法、数据暴露边界、配置示例和运行
方式；同时移除已与当前 CLI 不符的旧参数示例。
