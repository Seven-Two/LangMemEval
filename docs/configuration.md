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

## 旧配置迁移与范围

把旧 `.env` 中 `AMEM_EMBEDDING_*` 改为 `EMBEDDING_*`，
`AMEM_DEVICE` 改为 `EMBEDDING_DEVICE`，`AMEM_LOCAL_FILES_ONLY` 改为 `EMBEDDING_LOCAL_FILES_ONLY`。
程序发现旧名称会明确提示迁移，不静默忽略。聊天密钥仍用 `OPENAI_API_KEY`，无需改名。

共享 embedding 适用于本项目注册方法中的 A-Mem、LangMem；其他 MemEval 原生 baseline
仍可能使用各自的 embedding 服务，不应假定它们已全部统一。选择原生 baseline 时显式传入
`--embedding-*` 会报错，避免参数被静默忽略；其 `.env` 仍按各自适配器读取。
新方法可直接用 `EmbeddingSettings.from_env()` 和 `create_embedder(settings)` 获取共享配置与客户端。
`AML_*` 属于独立的比赛服务，当前请求没有更改其配置协议；不能用它们配置 baseline。
结果中的 `config.models` 保存共享模型配置（已脱敏），便于确认实验实际使用的设置。
