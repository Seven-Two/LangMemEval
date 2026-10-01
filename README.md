# LangMem + MemEval

用于迭代 Agent Memory 方法的整合项目：LangMem 提供记忆组件，MemEval
提供数据加载和评分，新方法通过注册机制接入统一实验流程。

## 快速开始

需要 Python >=3.12 和 uv。

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

根目录 `.env` 可填写 `OPENAI_API_KEY=...`，入口会自动加载，已有环境变量优先。
额外 baseline 按需安装，例如 `uv sync --extra dev --extra mem0`。
可选组：`mem0`、`simplemem`、`graphiti`、`memu`、`charts`、`training`。
这些组已纳入统一依赖解析，但未逐个运行；默认安装仅验证 LangMem 评测流程。

## 目录与开发

AML 比赛文本赛道 Add/Search 接入见 [AML 服务指南](docs/aml.md)。
安装 `uv sync --extra dev --extra aml`，配置 `.env` 后运行 `uv run langmem-aml`。

- `src/langmem_eval/`：整合代码与方法注册机制。
- `src/agents_memory/`：数据加载、baseline、评测、成本统计与实验运行器。
- `tests/`：测试；根目录 pyproject.toml 与 uv.lock 统一管理依赖。
- `scripts/`：抽样工具及兼容运行入口，实验主体在 `agents_memory.runner`。
- `third_party/memeval/`：上游许可证、来源文档与示例图，不是第二套运行项目。
- 新方法写在 `src/langmem_eval/methods/`，无需复制适配器。

详见 [使用与注册新方法](docs/development.md)。单个项目采用可编辑安装；
修改代码后重启评测即可，所有命令均在仓库根目录运行。

源码已统一到根目录 `src/`，不再有嵌套的 `MemEval/` 或 `LangMemEval/` 项目。
从旧布局更新后运行 `uv sync --locked --extra dev`，刷新可编辑安装的包路径；
如使用赛事服务，额外加 `--extra aml`。
数据、模型与结果默认位于当前工作目录的 `data/`、`models/`、`results/`，
可通过 `EVAL_DATA_DIR`、`EVAL_MODELS_DIR`、`EVAL_RESULTS_DIR` 指定。
`--output-dir` 优先于默认结果目录；不会向源码或 site-packages 写入实验数据。

已通过离线集成测试；未运行真实付费模型 benchmark，不提供性能结论。

## 第三方来源与许可证

仓库根目录 MIT LICENSE 适用于本项目原创整合代码。
`src/agents_memory/` 及保留的上游工具来自 [ProsusAI/MemEval](https://github.com/ProsusAI/MemEval)，
基础提交 `807ae6d7d8a5b76f6fe964d5a581d96c036e2ac4`，遵循其
[Apache-2.0 LICENSE](third_party/memeval/LICENSE) 与 [NOTICE](third_party/memeval/NOTICE)。
本地修改了系统发现、实验协议、失败记录和成本统计，并统一了源码与运行时路径。
上游原始说明和示例结果保留在 `third_party/memeval/`，不代表本项目实测结果。
LangMem 作为依赖使用，来源 https://github.com/langchain-ai/langmem。
