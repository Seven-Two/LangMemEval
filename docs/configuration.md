# 统一实验配置

本页适用于 `langmem-eval`。解析顺序固定为：

**命令行显式值 → 指定 `.env` 文件 → 进程环境变量 → 代码默认值。**

没有传参数才回退；显式 `0`、`false` 不会被当作遗漏。
空值会覆盖低优先级的同名设置，必填项不能留空。无效的显式值报错，不偷偷改用默认值。
`.env` 空密钥会覆盖进程中的同名密钥；
若想使用进程环境变量，请删除或注释 `.env` 中对应项。
从仓库根目录运行，默认读取该目录 `.env`。指定其他文件用 `--env-file experiments/qwen.env`。
为避免隐含取值，dotenv 值不执行 `${...}` 插值，请填写实际值。

## 模型参数

| 命令行参数 | `.env` 变量 | 默认值/含义 |
| --- | --- | --- |
| `--llm-model` | `LLM_MODEL` | `gpt-4.1`（示例文件显式设为 gpt-4.1-mini） |
| `--llm-api-key` | `OPENAI_API_KEY` | 聊天密钥；通常只放在 `.env` |
| `--llm-base-url` | `OPENAI_BASE_URL` | OpenAI API 根地址，不带 `/chat/completions` |
| `--llm-extra-body` | `LLM_EXTRA_BODY` | JSON 对象，默认 `{}` |
| `--embedding-provider` | `EMBEDDING_PROVIDER` | `local` 或 `openai`，默认 `local` |
| `--embedding-model` | `EMBEDDING_MODEL` | local 默认 MiniLM；API 模式必须指定 |
| `--embedding-api-key` | `EMBEDDING_API_KEY` | embedding 独立密钥 |
| `--embedding-base-url` | `EMBEDDING_BASE_URL` | embedding 独立 API 根地址 |
| `--embedding-dims` | `EMBEDDING_DIMS` | 正整数或 `auto`；auto 不传 API dimensions 参数 |
| `--embedding-batch-size` | `EMBEDDING_BATCH_SIZE` | 10 |
| `--embedding-device` | `EMBEDDING_DEVICE` | cpu，仅本地模型使用 |
| `--embedding-revision` | `EMBEDDING_REVISION` | 本地权重 revision，可固定模型提交 |
| `--embedding-local-files-only` / `--no-embedding-local-files-only` | `EMBEDDING_LOCAL_FILES_ONLY` | false，仅本地模型使用 |

A-Mem 和 LangMem 使用相同 embedding 配置和实现。LangMem 不再默认使用
text-embedding-3-small；默认和 A-Mem 一样使用本地 MiniLM。这会改变旧 LangMem
实验的模型设置，继续旧实验应显式传入原来的 API 模型和维度。
本地方式安装 `uv sync --locked --extra dev --extra amem`（amem 可选依赖组提供 SentenceTransformers）；
API 方式只需 `uv sync --locked --extra dev`。

LangMem 的向量存储需要提前知道维度：本地模型自动读取维度，API 模式必须设置
`EMBEDDING_DIMS`。设置后会向服务传 `dimensions`，须使用支持该参数的供应商。
A-Mem API 模式可以使用 auto，由实际响应决定维度。
聊天和 embedding 地址、密钥不相互回退；更改聊天服务不会把 embedding 请求发送到那里。

## 完整例子

根目录 `.env`：

```dotenv
OPENAI_API_KEY=你的聊天密钥
OPENAI_BASE_URL=https://你的聊天服务/v1
LLM_MODEL=你的聊天模型ID

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=你的embedding模型ID
EMBEDDING_API_KEY=你的embedding密钥
EMBEDDING_BASE_URL=https://你的embedding服务/v1
EMBEDDING_DIMS=1024
EVAL_SKIP_JUDGE=true
```

先检查最终配置，不会读取评测数据、下载权重或调用模型：

```powershell
uv run langmem-eval --systems amem,langmem --show-config
```

输出中的 `sources` 表示明确配置项来自 cli、.env 还是 environment；未列出的项目使用代码默认值。
密钥显示为 `<redacted>`；附加请求体仅展示已知的采样参数。

只覆盖聊天模型和读取预算，其余从 `.env` 读取：

```powershell
uv run langmem-eval --systems amem,langmem --llm-model 另一个模型ID --context-tokens 4096 --data-file examples/amem_smoke.json --num-samples 1 --skip-judge --output-dir results/comparison
```

这条命令会调用真实模型。一次实验结束或报错后，程序恢复原有进程环境，避免 CLI 覆盖污染后续实验。

## 其他参数

现有回答参数仍使用 `EVAL_*`：例如 `--top-k` 对应 `EVAL_TOP_K`，
`--answer-temperature` 对应 `EVAL_TEMPERATURE`，`--max-output-tokens` 对应 `EVAL_MAX_OUTPUT_TOKENS`。
数据/运行参数同样支持 `.env` 回退：

| CLI | `.env` |
| --- | --- |
| `--systems` | `EVAL_SYSTEMS` |
| `--benchmark` | `EVAL_BENCHMARK` |
| `--split` | `EVAL_SPLIT` |
| `--num-samples` | `EVAL_NUM_SAMPLES` |
| `--data-file` | `EVAL_DATA_FILE` |
| `--output-dir` | `EVAL_RESULTS_DIR` |
| `--skip-judge` / `--no-skip-judge` | `EVAL_SKIP_JUDGE` |
| `--judge-model` | `JUDGE_MODEL` |
| `--longmemeval-judge-model` | `LONGMEMEVAL_JUDGE_MODEL` |

A-Mem 专属算法设置仍保留 `AMEM_` 前缀，例如
`--amem-neighbor-k` / `AMEM_NEIGHBOR_K`、`--amem-evolution-threshold` / `AMEM_EVOLUTION_THRESHOLD`、
`--amem-response-format` / `AMEM_RESPONSE_FORMAT`、`--amem-temperature` / `AMEM_TEMPERATURE`、
`--amem-max-output-tokens` / `AMEM_MAX_OUTPUT_TOKENS`。

完整历史记忆缓存用 `--amem-cache-mode` / `AMEM_CACHE_MODE` 配置，默认 `off`：
`reuse` 命中则复用、未命中则构建保存；`require` 未命中就失败；`refresh` 强制重建。
目录用 `--amem-cache-dir` / `AMEM_CACHE_DIR`，默认相对工作目录的 `data/amem-cache`。
不自动跟随输出目录或 `EVAL_DATA_DIR`。LangMem 当前没有此缓存实现。
完整说明见 [A-Mem 缓存](amem.md#复用已经构建的记忆避免重复写入)。

## 旧配置迁移与范围

把旧 `.env` 中 `AMEM_EMBEDDING_*` 改为 `EMBEDDING_*`，
`AMEM_DEVICE` 改为 `EMBEDDING_DEVICE`，`AMEM_LOCAL_FILES_ONLY` 改为 `EMBEDDING_LOCAL_FILES_ONLY`。
程序发现旧名称会明确提示迁移，不静默忽略。聊天密钥仍用 `OPENAI_API_KEY`，无需改名。

当前内置方法只有 A-Mem 和 LangMem，均使用共享 embedding。
`--show-config` 的 `embedding_scope` 和结果同名字段均标记为 `shared`。
新方法可直接用 `EmbeddingSettings.from_env()` 和 `create_embedder(settings)` 获取共享配置与客户端。
`AML_*` 属于独立的比赛服务，当前请求没有更改其配置协议；不能用它们配置 baseline。
结果中的 `config.models` 保存共享模型配置（已脱敏），便于确认实验实际使用的设置。

所有方法的最终回答都使用 `LLM_MODEL` 和 `EVAL_*`。记忆内部的提示词、输出预算、
采样参数仍属于算法/SDK 配置；共享回答预算不会自动覆盖内部抽取调用。
`LLM_EXTRA_BODY` 用于共享回答器及 A-Mem、LangMem 的聊天调用，不自动应用于裁判。
接口和扩展方式见 [统一方法说明](unified-methods.md)。

## 运行日志与进度

正常评测默认同时向终端和 `--output-dir` 下的 `run_<运行标识>.log` 写入日志，
无需额外参数。日志与同次运行的 manifest、results 文件共用运行标识；结果 JSON
中的 `config.runtime_log` 指向对应日志。`--show-config` 不创建日志、不加载模型。

日志记录时间、阶段、方法和对话编号，以及适用的 session、消息/题目进度和耗时：

- `dataset.load` / `tokenizer.load`：数据和分词器加载。
- `backend.initialize` / `embedding.load`：方法初始化、嵌入模型名称和实际设备。
- `write.plan` / `write.session` / `write.progress`：历史会话和消息数量、完成进度。
- `write.finalize`：批量方法将已暂存的历史真正写入 SDK；进度条 ingest 完成不代表此阶段结束。
- `write.cache.load/save`、`amem.cache.hit/miss/saved/refresh`：完整记忆缓存加载、保存及命中状态。
- A-Mem 的 `amem.write.note`、`amem.analyze`、`amem.neighbors`、`amem.evolve`：
  每条历史消息的分析、邻居检索、演化；记录已保存记忆数量和索引重建。
- `embedding.encode` / `embedding.api`：本地嵌入和远程嵌入批次。
- `question.answer` / `retrieve` / `amem.query.rewrite` / `answer.api`：
  逐题检索、查询改写和回答；`context.selected` 记录候选数、选中数与上下文 Token 数。
- `judge` / `question.result`：裁判阶段与逐题状态、F1。
- `checkpoint.saved` / `results.save` / `method.summary`：落盘位置、覆盖率与汇总状态。

阶段状态为 `start`、`done`、`failed` 或 `stopped`。日志监测线程每 30 秒检查一次，
当前最内层阶段持续超过 30 秒时输出 `waiting` 和已用时间。这表示该调用尚未返回，
不是已确认死锁，也不能区分 SDK 内部的网络等待、重试与服务端排队。
它不设置请求超时、不发起重试、不自动重启实验。

日志逐条刷新，发生异常时保存完整 traceback 和原因链；同一异常沿嵌套阶段传播时只
打印一次堆栈。按 Ctrl+C 中断会记录 `stopped`，不会把该阶段记为完成。
失败题目的计数、覆盖率和运行退出码保持原有行为。

例如，在 Linux 另一个终端查看启动输出中给出的日志路径：

```bash
tail -f results/amem-locomo-first/run_<运行标识>.log
```

日志主动记录阶段元数据，不输出 API 密钥、请求头、请求体或对话/问题正文；实际回答
上下文仍保存在结果 JSON 中。第三方库的原始进度条不是这个日志文件的内容；异常文本
来自第三方时可能包含其请求信息，分享日志前应检查。
所有方法都记录统一写入、检索、回答和裁判阶段，A-Mem 额外提供逐消息子步骤。
第三方 SDK 的内部步骤不一定逐条可见；框架阶段日志不等于完整 SDK 日志。

扩展新方法时，可以直接使用统一日志接口：

```python
from agents_memory.diagnostics import stage, event

with stage("my_method.retrieve", top_k=10):
    records = retrieve_records()
    event("retrieval.ready", count=len(records))
```

不要把密钥、完整配置字典、提示词或记忆正文传给日志字段。

## 新方法参数

每个方法可在自己的注册入口声明 `MethodOption`，自动接入 CLI > `.env` > 进程环境 > 默认值。
无需修改公共配置映射。模板示例为 `--my-method-window` / `MY_METHOD_WINDOW`。
方法提供的无模型配置回调写入 `--show-config` 的 `methods.<name>` 和结果的
`config.method_settings`；A-Mem 现在使用 `methods.amem`，原有 `--amem-*` 参数保持不变。
具体示例见 [开发指南](development.md#为新方法增加参数)。
