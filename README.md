# LangMem + MemEval

用于迭代 Agent Memory 方法的整合项目：LangMem 提供记忆组件，MemEval
提供数据加载和评分，新方法通过注册机制接入统一实验流程。

## 快速开始

需要 Python >=3.12 和 uv。

```bash
git clone https://github.com/Seven-Two/LangMemEval.git
cd LangMemEval/LangMemEval
uv sync --locked --extra dev
uv run pytest -q
uv run langmem-eval --list-methods
```

设置环境变量 `OPENAI_API_KEY` 后运行（会产生 API 费用）：

```bash
uv run langmem-eval --systems langmem --benchmark locomo --num-samples 1 --llm-model gpt-4.1-mini --skip-judge --output-dir results/first-run
```

`--skip-judge` 只跳过裁判调用，不跳过记忆构建、embedding 和回答模型。

## 目录与开发

- `LangMemEval/`：整合包、方法注册机制、测试、统一依赖锁文件。
- `MemEval/`：包含本地集成改动的上游源码，作为普通目录纳入版本控制。
- 新方法写在 `LangMemEval/src/langmem_eval/methods/`，无需复制适配器。

详见 [使用与注册新方法](LangMemEval/README.md)。两个本地包采用可编辑安装；
修改代码后重启评测即可。保留并列目录结构以满足相对依赖路径。

已通过离线集成测试；未运行真实付费模型 benchmark，不提供性能结论。

## 第三方来源与许可证

仓库根目录 MIT LICENSE 适用于本项目原创整合代码。
`MemEval/` 来自 [ProsusAI/MemEval](https://github.com/ProsusAI/MemEval)，
基础提交 `807ae6d7d8a5b76f6fe964d5a581d96c036e2ac4`，遵循其
[Apache-2.0 LICENSE](MemEval/LICENSE) 与 [NOTICE](MemEval/NOTICE)。
本地修改了系统发现机制并添加 LangMem 注册入口。
LangMem 作为依赖使用，来源 https://github.com/langchain-ai/langmem。
