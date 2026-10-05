# A-Mem：统一评测框架中的核心方法复现

本实现参考作者的 [WujiangXu/A-mem](https://github.com/WujiangXu/A-mem/tree/0c8039f28fdcc08189a23c07a3437d9d2482f9c2)，
固定提交 `0c8039f28fdcc08189a23c07a3437d9d2482f9c2`，采用原始 JSON 版
`memory_layer.py`，不是另一个 `A-mem-sys` 包，也不是后来加入的 robust 变体。
代码按框架接口重实现核心算法，复用该版本的分析、演化、查询提示词及 JSON schema。
来源和 MIT 许可证见 [third_party/amem](../third_party/amem/README.md)。

这是**统一评测协议下的 A-Mem baseline**；离线行为测试通过不代表重现论文分数。
比较新方法时，应固定数据、模型、预算和回答协议，并在论文中披露以下适配差异。

## 1. 安装与模型配置

所有命令在仓库根目录运行，Python 3.12 或 3.13：

```powershell
uv sync --locked --extra dev --extra amem
uv run --extra amem langmem-eval --list-methods
```

列表应包含 `amem` 和 `langmem`。可选组 `amem` 提供本地 SentenceTransformers，
不在列出方法时加载模型。首次实际运行可能需要下载模型权重；断网运行需提前缓存权重，
设置 `EMBEDDING_LOCAL_FILES_ONLY=true`。测试使用替身，不需要模型权重。

### 矩池云：A16 / NVIDIA 510.54 驱动

当前依赖固定为 PyTorch 2.7.1、Sentence Transformers 5.1.2、Transformers 4.57.6。
Linux x86_64 从官方 `cu118` 索引安装 `torch==2.7.1+cu118`，使用 CUDA 11.8
运行库；Windows 开发环境使用 CPU 构建。Python 范围限制为 `>=3.12,<3.14`，
以匹配该 PyTorch 的 wheel。`training` 组也固定 PyTorch 版本，避免一起安装时升级回 CUDA 13。

NVIDIA 510.54 满足 CUDA 11.x 小版本兼容的基础驱动条件，但不代表所有 CUDA 功能
都支持；矩池云上的实际 GPU 运算和 MiniLM 编码仍需验证。本次依赖调整未进行云端 GPU 实测。
参考 [NVIDIA 兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

将更新后的 `pyproject.toml` 和 `uv.lock` 一起同步到服务器，再运行：

```bash
cd /mnt/LangMemEval
# /mnt 挂载盘不支持软链接时，将环境放到主目录。
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/langmemeval"
uv sync --locked --python 3.12 --extra dev --extra amem
```

不要手动编辑锁文件中的版本，也不要只通过 pip 替换 torch；后续同步会按锁文件恢复。
运行以下命令确认项目环境，而不是当前 Conda 环境中的另一个 torch：

```bash
uv run --locked --extra amem python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('Available:', torch.cuda.is_available()); x = torch.ones((2, 2), device='cuda'); print('GPU:', torch.cuda.get_device_name(0)); print('GPU result:', (x @ x).cpu())"
```

预期版本分别为 `2.7.1+cu118`、`11.8`，可用性为 `True`，矩阵结果全为 2。
这只是 GPU 基础检查；之后仍需验证 MiniLM 和 A-Mem。设置 `.env` 中
`EMBEDDING_DEVICE=cuda`，或者在评测命令中传 `--embedding-device cuda`。
`nvidia-smi` 仍显示 CUDA 11.6 是正常的，它与 PyTorch 自带运行库版本含义不同。
本机 CPU 开发环境不代表上述云端检查已经通过。

编辑已有 `.env`；尚无配置文件时从 `.env.example` 复制。不要覆盖已有密钥。

```dotenv
OPENAI_API_KEY=你的聊天服务密钥
OPENAI_BASE_URL=https://你的服务商/v1
LLM_MODEL=服务商实际支持的聊天模型ID

EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
AMEM_RESPONSE_FORMAT=json_schema
```

`OPENAI_BASE_URL` 填 API 根地址，不能带 `/chat/completions`。
`--llm-model` 优先于 `LLM_MODEL`；两者未设时框架默认 `gpt-4.1`。
同一个聊天模型用于记忆分析、演化、检索关键词生成和最终回答。
优先级为命令行 > `.env` > 进程环境变量 > 默认值。`AML_*` 只控制赛事服务。
embedding 配置现在由 A-Mem 和 LangMem 共用，详见 [统一配置](configuration.md)。

默认使用原作者代码的 MiniLM embedding；通过 `EMBEDDING_REVISION` 固定模型仓库的
commit，可进一步固定权重。本实现没有自动选择最新 revision 或声称已固定权重版本。

若聊天服务不支持严格 JSON schema，可显式设 `AMEM_RESPONSE_FORMAT=json_object`；
若连 JSON 模式也不支持，可设 `prompt`。三种模式都检查输出结构；演化 JSON 无法解析时
按上游策略跳过本次演化并记录，其他结构错误仍显式失败，不生成替代记忆。对于支持此选项的千问服务，可添加：

```dotenv
LLM_EXTRA_BODY={"enable_thinking":false}
```

该附加参数用于 A-Mem、LangMem 的聊天请求和统一回答请求，**不自动应用于裁判或原生 baseline 内部的 LLM**。
不支持的参数会导致服务报错；不要将密钥或模型名写入 `LLM_EXTRA_BODY`。

如果使用 embedding API，无需 `amem` 可选依赖，用 `uv sync --locked --extra dev` 安装即可。
替换以上 embedding 配置，例如：

```dotenv
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=服务商实际支持的embedding模型ID
EMBEDDING_BASE_URL=https://你的embedding服务商/v1
EMBEDDING_API_KEY=你的embedding密钥
# 仅在服务商支持 dimensions 参数时设置；否则删除或注释。
# EMBEDDING_DIMS=1024
EMBEDDING_BATCH_SIZE=10
```

API 模式发送字符串列表，默认每批最多 10 条。聊天与 embedding 的地址、密钥独立配置，
不会自动拿聊天服务调用 embedding。使用其他 embedding 模型属于改变模型配置的实验，
不能称为与原论文完全相同的设置。

## 2. 先跑最小样例

```powershell
uv run --extra amem langmem-eval --systems amem --data-file examples/amem_smoke.json --num-samples 1 --top-k 10 --context-tokens 6000 --skip-judge --output-dir results/amem-smoke
```

样例含 1 段对话、3 条历史发言、2 道题。正常情况下会有 6 次写入 LLM 调用、
2 次查询改写及 2 次回答调用，另有 embedding；重试另计。
**这个命令会调用配置的真实模型并可能产生费用**；`--skip-judge` 仅跳过裁判。
纯离线验收用第 5 节命令。

API embedding 模式下可以省略命令中的 `--extra amem`，避免安装本地模型依赖。
`.env` 修改后直接重启命令，无需改 Python 文件。

运行 LoCoMo：

```powershell
uv run --extra amem langmem-eval --systems amem --benchmark locomo --num-samples 1 --top-k 10 --context-tokens 6000 --max-output-tokens 256 --skip-judge --output-dir results/amem-locomo
```

`--num-samples` 指对话数量，不是问题数量；一段完整对话也可能写入很多笔记。
首次调通请使用小样例。正式比较时用同一份 `--data-file` 和相同协议运行所有方法，
不要因某个方法失败而删题。去掉 `--skip-judge` 前另行确认裁判模型与服务兼容。

## 3. 写入、检索与差异

1. 每条历史发言形成一条笔记，保留时间、说话人、原文和来源 ID，LLM 提取 keywords/context/tags。
2. 使用原文检索最近的 5 个旧笔记，由 LLM 决定 `strengthen` 和 `update_neighbor`。
   前者建立新笔记到旧笔记的有向连接、更新新标签；后者更新旧笔记的 context/tags，保留原文。
3. 每累计 100 次 `should_evolve=true` 重建索引。此前旧笔记的向量仍使用先前元数据，
   保留上游的延迟重建语义；新笔记即时加入索引。
4. 提问时先由 LLM 生成关键词，cosine 检索 top-k 种子笔记，再展开其已有链接。
   检索不会修改记忆，也不会在写入和检索时读取参考答案。
5. 将完整的“种子笔记及其关联笔记”作为一组证据交给统一 Token 预算选择器，再调用回答模型。

重要适配差异：

| 项目 | 本项目行为 |
| --- | --- |
| 回答和评分 | 使用现有统一协议，不复用上游按题型处理的回答脚本；不将 gold 答案选项传入模型 |
| 检索预算 | `--top-k` 是种子数，每个种子最多附带 k 个关联笔记；最终受 `--context-tokens` 约束 |
| 证据选择 | 整组放入或舍弃，不字符截断；大组可能被跳过；不同组内重复笔记保留 |
| 链接计数 | 修复上游关联展开中可能返回 k+1 条的边界问题，最多 k 条 |
| 链接身份 | 仅将候选邻居索引映射为稳定笔记 UUID；过滤非候选编号并记录，不因此中断写入；这是区别于上游直接保存所有编号的健壮性策略 |
| 邻居更新长度 | 与固定上游一致：按候选邻居顺序更新 `min(邻居数, 返回标签条数)` 条；忽略多余条目，缺少 context 时保留旧值；长度不匹配写入日志 |
| 失败处理 | 演化 JSON 解码失败时跳过本次演化，保留分析后的新笔记；所有截断响应先尝试解析；分析/查询非法 JSON、实际使用字段的缺失/类型错误、非法向量及网络错误仍显式失败；单条笔记原子提交 |
| 索引重建 | 重用相同 embedding 实例、批量重编码，省略重建前会被丢弃的新笔记单独编码；不重载模型 |
| 模型接口 | OpenAI 兼容聊天；可选远程 embedding；并非移植上游所有 Ollama/sglang 控制器 |
| 持久化和赛事 | 每段对话使用独立内存状态；未接入 AML 的持久化 Add/Search 服务，也未实现断点恢复 |

`AMEM_NEIGHBOR_K=5`、`AMEM_EVOLUTION_THRESHOLD=100`、`AMEM_TEMPERATURE=0.7`、
`AMEM_MAX_OUTPUT_TOKENS=1000` 是默认记忆配置。
`EVAL_TEMPERATURE` / `--answer-temperature` 和 `EVAL_MAX_OUTPUT_TOKENS` / `--max-output-tokens`
单独控制最终回答。更改参数需记录为实验配置，尤其不要将记忆生成预算与回答预算混为一谈。

### 邻居更新列表超过候选数

固定上游 `memory_layer.py:844–858` 使用 `min(len(indices), len(new_tags_neighborhood))`
限制更新循环，并不会因为数组过长而终止整段对话。当前已恢复此行为：

- 给出 5 个邻居、返回 7 组标签时，只按候选顺序使用前 5 组标签及可用的对应 context。
- 只返回 3 组标签时，只更新前 3 个邻居；其他邻居保持原状。
- 某个已更新邻居没有对应的新 context 时，保留其旧 context；多余的 context 不使用。
- 没有邻居时，不更新任何旧记忆。

这是上游的按位置容错策略，不能保证模型生成的内容在语义上一定对应正确。
新增 `amem.neighbor_updates.length_mismatch` 日志记录候选数、两个数组的实际长度、
应用及忽略的条目数，不记录记忆正文，也不因此增加 LLM 调用。
保留原始提示词和 JSON Schema，不增加动态长度约束或纠正重试。

原适配版本会对过长数组抛出 `ValueError`，比上游更严格。
本次修复将实现标识更新为 `amem_original_json_unified_v2`，结果的 `method_config`
增加 `neighbor_update_policy=upstream_positional_prefix`，便于区分修复前后的实验。
v2 保留了单条笔记的原子写入和非法链接报错；v4 将非候选链接改为过滤，详见下文。
这些适配不表示整个框架与上游评测协议完全一致。

### 演化输出截断与 JSON 解析失败

v3 引入演化解析失败回退，当前实现标识为 `amem_original_json_unified_v4`。默认 `AMEM_MAX_OUTPUT_TOKENS=1000`
沿用固定上游 OpenAI 控制器的单次输出上限，并不表示 1000 对所有模型都是最佳预算。
处理流程对齐上游 `process_memory` 的 JSON 解码失败策略：

- 响应 `finish_reason=length` 时记录截断，但仍尝试解析；完整且通过实际使用字段校验的 JSON 可以正常应用。v4 将此规则扩展到分析和查询改写。
- 按上游方式提取第一个 `{` 到最后一个 `}` 之间的内容，不补全或猜测被截断的 JSON。
- 仅 JSON 解码失败时跳过这次演化：不改旧邻居、不增加成功演化计数，使用已完成分析的新笔记进行向量编码并保存；后续消息和问答继续。
- 不增加纠正请求，不自动提高预算；SDK 原有网络重试策略不变。
- 实际使用字段缺失/类型错误、网络与 embedding 错误仍显式报错；分析和查询改写不回退无效 JSON。若跳过演化后新笔记编码失败，仍不提交该笔记。

日志新增 `amem.response.truncated`、`amem.evolution.skipped` 和 `amem.evolution.stats`。
截断和跳过事件携带原调用的 `request_id`、`response_id`、`finish_reason`，以及会话/消息位置；
精简终端不会逐条打印这些事件，完整文件日志仍保留。日志不保存响应正文。

结果中每题的 `answer_trace.method_config` 增加：

```json
{
  "evolution_failure_policy": "upstream_skip_invalid_json",
  "evolution_stats": {"attempts": 24, "truncated": 1, "skipped_invalid_json": 1}
}
```

以上数字仅为示例。统计按 conversation 独立累计，并复制到每题 trace；汇总时不要将同一对话
重复计算。`truncated` 只数演化响应返回 `length` 的次数，`skipped_invalid_json` 包括未截断但
JSON 无效的响应；两者不一定相等。统计记录发生过的尝试，即使后续编码失败也不会回滚，
与 `snapshot().evolution_count` 的成功演化次数含义不同。若写入阶段因其他错误终止，
最新统计仍可在日志中查看；未进入问答的失败题不含此方法 trace。

调试当前模型的预算可在命令末尾添加 `--amem-max-output-tokens 2048`，或在 `.env` 中设置
`AMEM_MAX_OUTPUT_TOKENS=2048`；命令行优先。它不同于最终回答的 `--max-output-tokens`。
建议在固定开发子集对比 1000/2048 的截断率、跳过率、问答效果、成本和耗时，再冻结正式实验配置。
2048 是待验证的工程选项，不是论文保证的最佳值。

矩池云端可执行新增离线回归（无真实模型请求）：

```bash
PYTHON_DOTENV_DISABLED=1 uv run --locked --extra dev python -m pytest -q tests/test_amem_output_validation.py tests/test_amem_evolution_fallback.py tests/test_amem_contract.py tests/test_amem_extra.py tests/test_amem_neighbor_updates.py tests/test_llm_diagnostics.py
```

### v4：过滤非候选链接与校验审查

模型返回 `suggested_connections` 中的整数如果不在当前候选邻居中，会被过滤。即使该编号
对应某条已存在的记忆，也不会关联；负数、越界编号同样过滤。合法编号的顺序和重复项保留，
不会将编号猜测为候选列表的相对位置。全部被过滤时，新笔记仍保存，合法标签更新和邻居更新仍执行。
没有候选时不会建立链接。此规则不增加模型请求，默认输出预算仍为 1000。

固定上游 `memory_layer.py:833–835` 直接保存返回编号，`893–894` 检索时直接用作下标。
因此 `link_policy=filter_non_candidates` 是本适配器的健壮性改进，不是与上游完全相同的行为。

日志 `amem.links.filtered` 记录 `candidate_indices`、`returned_indices`、`kept_indices`、
`filtered_indices` 及原请求 ID；数字使用当前会话记忆列表的全局索引，不记录记忆正文。
`answer_trace.method_config` 增加 `link_policy`、`unknown_action_policy`、
`response_validation_policy` 和 `output_adjustments`：

```json
{
  "filtered_link_responses": 1,
  "filtered_links": 3,
  "ignored_action_responses": 0,
  "ignored_actions": 0
}
```

数字为示例；这些是按 conversation 累计的响应/条目数，重复非法编号按出现次数统计，
即使后续编码失败也保留已观察到的计数。每题复制同一份累计值，汇总时不能逐题相加。
写入阶段失败时可从事件日志查看，未进入问答的结果不含方法 trace。

本次审查覆盖 A-Mem 模型边界、写入/检索、参数检查及共用 embedding 返回值检查：

| 检查 | v4 处理及依据 |
| --- | --- |
| 非候选链接 | 过滤并继续；不接受负数下标，不猜测模型意图 |
| 未知演化动作 | 忽略未知动作，照常执行 `strengthen` / `update_neighbor`；上游仅处理这两个分支，不为其他动作报错；`amem.actions.ignored` 记录数量 |
| 额外 JSON 字段 | 忽略算法不使用的顶层字段，`amem.response.extra_fields_ignored` 仅记录数量；不记录额外字段名/内容 |
| 未执行动作的字段 | `should_evolve=false` 时只要求该布尔值；为 true 时要求动作数组及所选已知动作需要的字段。未用字段缺失或类型错误不阻断，和上游按分支读取字段的方式一致 |
| 标记截断但 JSON 完整 | 分析、演化、查询均先解析及校验，不仅凭 `length` 报错；不补全不完整 JSON |
| 邻居更新数组长度不一致 | 保留 v2 的上游位置前缀规则 |
| 实际使用字段缺失/类型错误 | 保留错误，包括字符串形式的编号、布尔值充当编号；不强制转换；错误消息包含字段路径便于定位 |
| 分析无效 JSON、查询无效 JSON 或空关键词 | 保留错误，不捏造元数据或静默改用原问题；演化 JSON 解码失败沿用 v3 回退 |
| 向量形状、维度、非有限值、零向量及返回索引 | 保留，避免余弦检索出错或数据错配 |
| 非法输入消息、非正预算/检索数量、错误端点及网络故障 | 保留，不能通过忽略错误构造可信的评测结果 |

严格 JSON 请求中的原始提示词和 Schema 保持不变；以上是返回后的本地处理策略，
不保证服务商一定遵守请求 Schema。`should_evolve=true` 仍按上游增加演化计数，
即便所有链接被过滤或动作全部未知；该计数不是“实际改变了多少条记忆”。
无需新增配置。离线回归已随新策略更新：原先“非法整数链接必须报错”的验收断言
改为“错误类型链接必须报错”，另有专门测试覆盖非法整数过滤、继续检索和结果统计。

## 4. 查看结果与继续开发

输出目录保存固定问题 manifest、逐题 progress.jsonl、完整 results.json 及汇总。
每题的 `answer_trace` 包含：

- `method_config`：固定上游提交、实现版本、提示词 hash、模型及公开配置、实际向量维度。
- `retrieval_trace`：改写后的查询、种子 ID 和各证据组的笔记 ID。
- `context`、`token_count`、`selected_indices`、`dropped_indices`：实际回答上下文及选择记录。
- `messages`：实际回答提示词；空上下文弃答时不调用回答器。

成本在 `cost_accounting` 中区分 write/retrieve/answer/judge。
本地 embedding 单独记录调用次数，Token 和计算费用为未知，**不等于零成本**。
远程服务按可观测 SDK usage 统计；未配置价格的模型费用仍为未知。
失败会在运行日志中保留完整异常堆栈；逐题结果仍记录阶段和异常类型。

### 慢请求诊断

终端默认使用**精简模式**：保留运行概要、进度、会话结果落盘提示和简短错误定位。
逐次请求、完整 `usage`、等待心跳和完整 traceback 仍全部写入 `run_*.log`，
不会因终端精简而丢失诊断信息。模型下载/加载进度及第三方组件自身的输出不受此选项统一控制。

在原运行命令后添加以下参数即可切换终端详细程度：

```bash
# 精简终端输出（默认），完整日志保存在文件中
--log-mode concise

# 在终端也显示完整阶段、请求和异常日志
--log-mode full
```

也可以在 `.env` 中设置 `EVAL_LOG_MODE=concise` 或 `EVAL_LOG_MODE=full`。
优先级为命令行 > `.env` > 进程环境变量 > 默认 `concise`。
`--show-config` 的 `logging` 字段以及结果文件的 `console_log_mode` / `file_log_mode`
会记录最终设置。此选项只控制终端详细程度，文件日志始终完整。

交互终端默认显示两层进度，无需新增参数：上层是 conversation 数量，下层依次显示
`Write memory`（已成功保存的消息数 / 历史消息总数）和 `Answer questions`
（已处理题数 / 总题数，包含失败题）。A-Mem 按条推进写入，其他统一后端按 session 完成时推进。
进度条还显示用时、估算剩余时间，以及 `analyze`、`evolve`、`retrieve`、`answer`、
`judge`、`waiting` 等当前状态；预计剩余时间会随模型响应速度波动。
回答阶段的 `errors` 表示已处理题中的回答或裁判错误数，不是答错题数。
失败或中断不会把未完成的写入补成 100%；终端进度条不写入日志文件，
stderr 不连接交互终端时自动禁用。原生 baseline 内部阶段未统一接入，保留外层对话进度。

无需增加参数，继续使用原来的运行命令。日志保存在 `--output-dir` 指定目录中的
`run_<run_tag>.log`，启动时会打印具体路径（精简模式显示 `Log: ...`）。
新增诊断覆盖 A-Mem 的分析、演化、查询改写，以及统一适配器的最终回答；
不代表已经覆盖其他 baseline 自己的客户端、裁判或远程服务内部的调用。

同一次逻辑请求的所有诊断行共享 `request_id`，同时保留会话、消息/问题序号。
`parent_stage` 可区分 `amem.analyze`、`amem.evolve`、`amem.query.rewrite` 和 `answer.api`。

| 日志事件/字段 | 含义 |
| --- | --- |
| `llm.request` | 模型、输出预算、SDK 最大重试次数、实际传入的顶层或嵌套 `enable_thinking` 布尔值；传入不等于服务端一定采用 |
| `llm.http.start` | 一次 SDK HTTP 发送开始，`attempt` 从 1 计数 |
| `llm.http.response` | 本次发送耗时、HTTP 状态码、服务商请求 ID、`retry-after` / `retry-after-ms` |
| `llm.http.error` | 本次发送的异常类型，例如 `ReadTimeout`，以及耗时 |
| `llm.retry` | 确实开始下一次尝试时记录；包含上次状态码/异常和两次尝试间隔 |
| `llm.response` | 整次调用耗时、完整 `usage`、输出长度、尝试次数及重试次数 |
| `llm.error` | SDK 最终失败时的异常类型、总耗时、已发生的尝试次数及重试次数 |
| `visible_output_chars` | 原始 `message.content` 的 Unicode 字符数，未去空白或解析 JSON；不是 Token 数 |
| `reasoning_chars` | 返回的推理文本字符数；支持 `reasoning_content`、`reasoning` 和可识别的 `reasoning_details` 文本 |
| `reasoning_chars_by_field` | 分字段记录长度，不把可能重复的推理字段相加；`reasoning_field` 表示选用了哪个字段 |
| `usage` | SDK 解析后的完整服务商用量对象，包括返回的推理、缓存用量和扩展字段；不自行估算 |
| `http_attempts` / `retry_count` | SDK HTTP 发送次数 / 额外尝试次数；不修改 SDK 原有重试策略或成本统计 |

长度 `null` 表示没有返回可识别文本，**不代表没有思考**；空字符串的长度为 `0`。
如果服务商把 `<think>` 内容直接放进 `content`，它仍算在 `visible_output_chars` 中，
这里不会猜测和拆分正文。Token 数以服务商 `usage` 为准，其准确性仍需要服务商确认。
注入未接入 HTTP 诊断的自定义客户端时，尝试/重试次数为 `null`，避免误报成没有重试。

重试间隔包含 SDK 退避等待和少量本地处理，不是精确的睡眠时长；服务商内部重试不可见。
当前为非流式调用，HTTP 耗时包含网络和等待完整响应的时间，不能拆出首 Token 延迟、
服务端排队和生成各自耗时。`waiting` 是心跳日志，不是额外 API 请求。

例如同一个 `request_id` 出现 `429 → llm.retry → 200`，说明确实遇到了限流重试；
只有一次尝试却耗时 200 秒，说明这 200 秒没有发生客户端 SDK 重试。
若正文很短而 `usage.completion_tokens_details.reasoning_tokens` 很高，
可以确认服务商报告的主要输出用量属于推理部分。

这些新增结构化诊断不会记录提示词、输出/推理正文、密钥或完整 HTTP 请求头。
现有异常堆栈仍可能包含服务商错误信息；原有结果文件中的回答上下文/提示词记录继续保留。

在矩池云环境可运行离线诊断测试（无需模型、网络或密钥）：

```bash
uv run --locked --extra dev python -m pytest -q tests/test_llm_diagnostics.py tests/test_diagnostics.py
```

代码位置：

- `src/langmem_eval/methods/amem/backend.py`：会话状态、写入/检索编排和原子提交。
- `src/langmem_eval/methods/amem/evolution.py`：纯笔记/链接更新；返回候选状态、诊断与统计，不调用模型。
- `src/langmem_eval/methods/amem/responses.py`：纯解析和运行时校验；显式区分 analysis/evolution/query，避免根据 Schema 内容猜测阶段。
- `src/langmem_eval/configuration.py`：命令行、dotenv 的统一解析和共享 embedding 设置。
- `src/langmem_eval/embeddings.py`：A-Mem 与 LangMem 共用的本地/API embedding 客户端。
- `src/langmem_eval/methods/amem/config.py`：A-Mem 算法设置和公开配置记录。
- `src/langmem_eval/methods/amem/memory.py`：笔记结构与序列化。
- `src/langmem_eval/methods/amem/clients.py`：LLM 请求、响应解析入口及诊断；模型客户端可注入。
- `src/langmem_eval/llm_diagnostics.py`：聊天响应长度、完整用量和逐次 HTTP 重试诊断。
- `src/langmem_eval/methods/amem/prompts.py`：固定上游提示词和 schema。
- `src/langmem_eval/methods/amem/__init__.py`：方法、参数和公开配置回调注册入口。
- `src/langmem_eval/adapter.py`：统一回答和 trace。
- `src/langmem_eval/evaluation.py`：同步逐题评分，沿用框架评分函数与结果字段；无需异步事件循环。

`AMemBackend.snapshot()` 可用于 Python 调试笔记内容；目前不自动将完整笔记库写入结果文件，
检索上下文和实际回答请求会保存。进一步实现新方法时，保留 `amem` baseline，
在 `methods/` 注册另一个名称，接入你自己的 backend，详见 [开发指南](development.md)。

注入 A-Mem controller 时，实现 `complete(prompt, schema, *, purpose="generic") -> dict`；
purpose 为 analysis/evolution/query。输出策略在 `responses.py` 中明确选择，不通过字典相等判断阶段。
注入 embedder 时实现 `encode(texts)`，返回二维向量；共享验证在 `langmem_eval.embeddings.validate_vectors`。
后端只关闭自己创建的资源；注入资源由测试或上层调用方关闭。
这次职责拆分不修改 v4 的提示词、图更新语义、默认预算或模型调用次数。

## 5. 离线验证

```powershell
uv sync --locked --extra dev
.venv/Scripts/python.exe scripts/verify_amem.py
.venv/Scripts/python.exe scripts/verify_amem.py --guard
```

第一个命令对应当前版本的 10 项行为验收（v4 已按新链接策略修订），指标 `amem_contract_pass_rate` 满分为 1；
第二个命令检查现有回归及新增边界测试。脚本禁用 `.env` 加载并注入本地替身，
不使用真实密钥、不下载模型、不调用付费服务。
它们验证工程行为，不评价真实模型的记忆准确率；仍需实际数据集实验才能得出论文结论。
