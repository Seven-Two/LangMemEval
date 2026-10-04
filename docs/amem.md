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
若连 JSON 模式也不支持，可设 `prompt`。三种模式都检查输出结构，失败会计入失败样本，
不会悄悄生成替代记忆。对于支持此选项的千问服务，可添加：

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
| 链接身份 | 将合法的邻居索引映射为稳定笔记 UUID，拒绝越界及非候选邻居 |
| 失败处理 | 非法 JSON、截断响应、非法向量显式失败；一条笔记更新原子提交，整段对话失败保留题目分母 |
| 索引重建 | 重用相同 embedding 实例、批量重编码，省略重建前会被丢弃的新笔记单独编码；不重载模型 |
| 模型接口 | OpenAI 兼容聊天；可选远程 embedding；并非移植上游所有 Ollama/sglang 控制器 |
| 持久化和赛事 | 每段对话使用独立内存状态；未接入 AML 的持久化 Add/Search 服务，也未实现断点恢复 |

`AMEM_NEIGHBOR_K=5`、`AMEM_EVOLUTION_THRESHOLD=100`、`AMEM_TEMPERATURE=0.7`、
`AMEM_MAX_OUTPUT_TOKENS=1000` 是默认记忆配置。
`EVAL_TEMPERATURE` / `--answer-temperature` 和 `EVAL_MAX_OUTPUT_TOKENS` / `--max-output-tokens`
单独控制最终回答。更改参数需记录为实验配置，尤其不要将记忆生成预算与回答预算混为一谈。

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

无需增加参数，继续使用原来的运行命令。日志保存在 `--output-dir` 指定目录中的
`run_<run_tag>.log`，启动时的 `status="log.open"` 会打印具体路径。
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

- `src/langmem_eval/methods/amem/backend.py`：写入演化与检索算法。
- `src/langmem_eval/configuration.py`：命令行、dotenv 的统一解析和共享 embedding 设置。
- `src/langmem_eval/embeddings.py`：A-Mem 与 LangMem 共用的本地/API embedding 客户端。
- `src/langmem_eval/methods/amem/config.py`：A-Mem 算法设置和公开配置记录。
- `src/langmem_eval/methods/amem/memory.py`：笔记结构与序列化。
- `src/langmem_eval/methods/amem/clients.py`：A-Mem 结构化 LLM 输出边界。
- `src/langmem_eval/llm_diagnostics.py`：聊天响应长度、完整用量和逐次 HTTP 重试诊断。
- `src/langmem_eval/methods/amem/prompts.py`：固定上游提示词和 schema。
- `src/langmem_eval/methods/amem/__init__.py`：方法注册入口。
- `src/langmem_eval/adapter.py`：统一回答和 trace。
- `src/langmem_eval/evaluation.py`：同步逐题评分，沿用框架评分函数与结果字段；无需异步事件循环。

`AMemBackend.snapshot()` 可用于 Python 调试笔记内容；目前不自动将完整笔记库写入结果文件，
检索上下文和实际回答请求会保存。进一步实现新方法时，保留 `amem` baseline，
在 `methods/` 注册另一个名称，接入你自己的 backend，详见 [开发指南](development.md)。

## 5. 离线验证

```powershell
uv sync --locked --extra dev
.venv/Scripts/python.exe scripts/verify_amem.py
.venv/Scripts/python.exe scripts/verify_amem.py --guard
```

第一个命令对应固定的 10 项行为验收，指标 `amem_contract_pass_rate` 满分为 1；
第二个命令检查现有回归及新增边界测试。脚本禁用 `.env` 加载并注入本地替身，
不使用真实密钥、不下载模型、不调用付费服务。
它们验证工程行为，不评价真实模型的记忆准确率；仍需实际数据集实验才能得出论文结论。
