# LangMem + MemEval

用于迭代 Agent Memory 方法的整合项目：LangMem 提供记忆组件，MemEval
提供数据加载和评分，新方法通过注册机制接入统一实验流程。

## 快速开始

需要 Python 3.12 或 3.13 和 uv。

```bash
git clone https://github.com/Seven-Two/LangMemEval.git
cd LangMemEval
uv sync --locked --extra dev
uv run pytest -q
uv run langmem-eval --list-methods
```

设置环境变量 `OPENAI_API_KEY` 后运行（会产生 API 费用）：

```bash
uv run langmem-eval --systems langmem --benchmark locomo --num-samples 1 --llm-model gpt-4.1-mini --skip-judge --output-dir results/first-run
```

`--skip-judge` 只跳过裁判调用，不跳过记忆构建、embedding 和回答模型。

统一适配器现支持 `--context-tokens`、`--max-output-tokens`、`--answer-style`、
`--empty-context` 等显式协议配置，逐题保存实际上下文和提示词。
运行前固定题目清单，失败样本保留在结果中；不完整运行保存结果后返回退出码 2。
成本按阶段报告，裁判独立统计；未观测到的费用标记为未知。
原生 baseline 保留自身协议，结果会明确区分，详见 [实验配置与成本记录](docs/development.md)。

根目录 `.env` 可填写 `OPENAI_API_KEY=...`。参数优先级为：命令行 > `.env` > 进程环境变量 > 默认值。
聊天使用 `--llm-model`、`--llm-base-url`，嵌入使用 `--embedding-provider`、`--embedding-model` 等。
A-Mem 和 LangMem 共用 `EMBEDDING_*`；运行 `uv run langmem-eval --systems amem --show-config`
查看脱敏后的最终配置，不调用模型。完整参数表见 [统一配置](docs/configuration.md)。
额外 baseline 按需安装，例如 `uv sync --extra dev --extra mem0`。
可选组：`amem`、`mem0`、`simplemem`、`graphiti`、`memu`、`charts`、`training`。
这些组已纳入统一依赖解析，但未逐个运行；默认安装仅验证 LangMem 评测流程。

## 目录与开发

### A-Mem baseline

已接入固定上游版本的 A-Mem 核心方法：笔记生成、链接、邻居演化和关键词检索，
使用本框架统一回答协议。配置 `.env` 中的 `LLM_MODEL`、聊天 API 和 embedding 后运行：

```powershell
uv sync --locked --extra dev --extra amem
uv run --extra amem langmem-eval --systems amem --data-file examples/amem_smoke.json --num-samples 1 --top-k 10 --skip-judge --output-dir results/amem-smoke
```

上述命令会调用真实模型。配置千问兼容服务、API embedding、离线验收与复现差异见
[A-Mem 使用指南](docs/amem.md)。未运行论文规模实验，不声称复现论文分数。

AML 比赛文本赛道 Add/Search 接入见 [AML 服务指南](docs/aml.md)。
安装 `uv sync --extra dev --extra aml`，配置 `.env` 后运行 `uv run langmem-aml`。

- `src/langmem_eval/`：整合代码与方法注册机制。
- `src/agents_memory/`：数据加载、baseline、评测、成本统计与实验运行器。
- `tests/`：测试；根目录 pyproject.toml 与 uv.lock 统一管理依赖。
- `scripts/`：抽样和离线验收工具；评测使用 `langmem-eval`，实验主体在 `agents_memory.runner`。
- `third_party/memeval/`：上游许可证、来源文档与示例图，不是第二套运行项目。
- 每个方法集中在 `src/langmem_eval/methods/<方法名>/`，包含注册、算法及所需配置/提示词。
- `src/langmem_eval/interfaces.py` 定义共享接口；`examples/method_template/` 可复制为新方法起点。

详见 [使用与注册新方法](docs/development.md)。单个项目采用可编辑安装；
修改代码后重启评测即可，所有命令均在仓库根目录运行。
模块边界、方法参数注册、资源生命周期与论文实验注意事项见 [架构说明](docs/architecture.md)。

源码已统一到根目录 `src/`，不再有嵌套的 `MemEval/` 或 `LangMemEval/` 项目。
从旧布局更新后运行 `uv sync --locked --extra dev`，刷新可编辑安装的包路径；
如使用赛事服务，额外加 `--extra aml`。
数据、模型与结果默认位于当前工作目录的 `data/`、`models/`、`results/`，
可通过 `EVAL_DATA_DIR`、`EVAL_MODELS_DIR`、`EVAL_RESULTS_DIR` 指定。
`--output-dir` 优先于默认结果目录；不会向源码或 site-packages 写入实验数据。

终端默认精简输出，完整阶段、请求用量、重试及异常堆栈保存在输出目录的 `run_*.log`。
在原命令后添加 `--log-mode full` 可在终端也显示完整日志，`--log-mode concise` 切回精简模式。
也可在 `.env` 设置 `EVAL_LOG_MODE=concise` 或 `full`；命令行优先，文件日志始终完整。
交互终端中还会显示对话总进度，以及统一适配器的记忆写入/问题处理进度。
A-Mem 每成功写入一条记忆就推进写入进度；其他统一后端按完成的 session 推进。
问题进度包含已处理的失败题，不能当作准确率；终端错误输出被重定向时自动关闭进度条。

已通过离线集成测试；未运行真实付费模型 benchmark，不提供性能结论。

## 第三方来源与许可证

仓库根目录 MIT LICENSE 适用于本项目原创整合代码。
`src/agents_memory/` 及保留的上游工具来自 [ProsusAI/MemEval](https://github.com/ProsusAI/MemEval)，
基础提交 `807ae6d7d8a5b76f6fe964d5a581d96c036e2ac4`，遵循其
[Apache-2.0 LICENSE](third_party/memeval/LICENSE) 与 [NOTICE](third_party/memeval/NOTICE)。
本地修改了系统发现、实验协议、失败记录和成本统计，并统一了源码与运行时路径。
`src/langmem_eval/evaluation.py` 同样从 MemEval 评分流程改编，沿用 Apache-2.0。
上游原始说明和示例结果保留在 `third_party/memeval/`，不代表本项目实测结果。
LangMem 作为依赖使用，来源 https://github.com/langchain-ai/langmem。

A-Mem 提示词和算法来源为 [WujiangXu/A-mem](https://github.com/WujiangXu/A-mem)，
固定提交、适配说明和 MIT 许可证见 [来源记录](third_party/amem/README.md)。
