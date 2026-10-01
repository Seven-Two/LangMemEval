# LangMem + MemEval

中性的记忆评测整合工程。LangMem 负责记忆抽取、更新和检索；MemEval
负责数据加载、评分和结果输出。本工程不引入额外研究方法。

## 开始使用

在 PowerShell 中运行：

```powershell
cd <仓库目录>
uv sync --extra dev
uv run pytest -q
uv run langmem-eval --help
$env:OPENAI_API_KEY = "你的密钥"
uv run langmem-eval --benchmark locomo --systems langmem --num-samples 1 --llm-model gpt-4.1-mini --skip-judge --output-dir results/locomo-smoke
```

真实评测会调用付费模型及 embedding。`--skip-judge` 只关闭裁判调用。
一个 LoCoMo 样本可能有很多问题。结果为 JSON；先跑小样本核查费用统计。

## 开发入口

- `src/langmem_eval/backend.py`：LangMem API、向量存储与检索。
- `src/langmem_eval/benchmark.py`：归一化会话转换、上下文预算、后端协议。
- `src/langmem_eval/adapter.py`：唯一的评测适配器实现。
- `src/langmem_eval/cli.py`：调用 `agents_memory.runner.main` 的统一入口。
- `src/agents_memory/runner.py`：实验清单、执行、结果落盘与汇总。
- `src/agents_memory/systems/langmem.py`：只有导入语句的发现入口。
- `src/agents_memory/paths.py`：统一数据、模型和结果目录。

根目录的单个项目同时打包 langmem_eval 与 agents_memory，通过 uv 可编辑安装。修改上述源码，重新启动评测即可；无需复制。
两者源码均在根目录 `src/` 下。从旧目录布局更新后先执行
`uv sync --locked --extra dev`（赛事服务再加 `--extra aml`）刷新安装路径。
`scripts/run_full_benchmark.py` 仅保留兼容入口，运行逻辑只有包内的一份。
上游文档、许可证和历史示例图归档到 `third_party/memeval/`；其中配置示例不自动生效。

数据默认写到当前工作目录 `data/`，训练模型到 `models/`，实验结果到 `results/`。
可用 `EVAL_DATA_DIR`、`EVAL_MODELS_DIR`、`EVAL_RESULTS_DIR` 配置绝对或相对路径；
这些目录不依赖源码位置，wheel 安装也不会向 site-packages 写入数据。
LoCoMo 缓存在数据目录根层，LongMemEval 缓存在其 `longmemeval/` 子目录。
`--output-dir` 和 `MEMORY_R1_MM_ADAPTER`/`MEMORY_R1_AA_ADAPTER` 等显式参数仍优先。
LangMem 使用固定版本依赖，不复制第三方源码。依赖解析记录在 uv.lock 中。
如需修改 LangMem 本身，可将其源码克隆到本地，并在 tool.uv.sources 中指定
editable 路径后重新 uv sync。当前优先通过 backend 中的接口组合进行扩展。

## 注册新方法

查看可用方法：`uv run langmem-eval --list-methods`。
在 `src/langmem_eval/methods/my_method.py` 中添加：

```python
from ..registry import register_method
from ..backend import LangMemBackend

@register_method("my_method", architecture="描述你的算法", infrastructure="LangMem")
class MyMethod(LangMemBackend):
    def retrieve(self, query: str, limit: int) -> list[str]:
        candidates = super().retrieve(query, limit * 3)
        # 在这里实现你的重排序或记忆选择；此示例尚无创新算法。
        return candidates[:limit]
```

然后运行：

```powershell
uv run langmem-eval --systems langmem,my_method --num-samples 1 --skip-judge --output-dir results/comparison
```

每次启动自动发现 methods 下非下划线开头的模块，无需修改 MemEval 或复制代码。
装饰器也可注册工厂函数，接受一个 model 参数并返回实现 ingest(session)、
retrieve(query, limit) 的对象；每个样本都会重新创建对象。
模块顶层只做定义与注册，模型初始化和可选依赖导入放进工厂/构造器。
方法名使用小写字母、数字和下划线，以字母开头，不能为 all，也不能与
其他 MemEval 系统重名；重名会明确报错。修改代码后重启评测即可。
通过这个注册机制接入的方法共用 adapter 的回答/评分逻辑和上下文预算；方法元数据写入运行结果。
MemEval 原生 systems 下的其他 baseline 仍使用自身回答逻辑，结果标记为 `adapter_protocol=native`。
不能仅因为使用同一个 runner 就宣称这些 baseline 采用了相同的回答配置。

## 实验约定（评测流程）

按 session 编号依次导入历史，保留说话人、时间和来源 ID；写入完成后才评测。
每个样本独立 namespace/store，测试问题和答案不写回记忆。
默认管理器删除关闭、更新检索 query_limit=5；回答 top_k=20。
读取采用真实分词器计数，逐条选择完整记录，不再截断字符串。
默认 `EVAL_TOP_K=20`、`EVAL_CONTEXT_TOKENS=6000`、`EVAL_TOKENIZER=cl100k_base`。
计数包含记录间的两个换行；超预算记录跳过，继续检查后续记录。不切开正文。
这是记忆段预算，不包含问题、系统提示或输出；不能当作模型总窗口限制。
分词器是显式实验配置，不会声称它与任意供应商模型的内部 tokenizer 相同。
首次使用需要下载公开 tiktoken 分词表，之后可使用本地缓存。
`LANGMEM_TOP_K` 保留为旧环境变量回退；`LANGMEM_MAX_CONTEXT_CHARS` 已弃用，设置它会警告并忽略。

回答协议默认按 benchmark 选择：LoCoMo 输出上限 256 Token，LongMemEval 为 512；
两者默认温度 0.1。LoCoMo 要求简洁且完整，取消 1–5 词限制；LongMemEval 要求完整回答。
这些是本项目的可配置协议，不代表官方论文配置。
默认空上下文直接返回 `None`，不调用回答模型；可选 `answer` 让回答器处理空上下文。
每道题的 `answer_trace` 保存最终 `context`、真实 Token 数、选择/丢弃的候选位置、
完整请求 `messages`、协议和回答结束原因。空上下文直接拒答时 `messages=null`。
结果包含原始证据与提示内容，应按实验数据权限管理。

```powershell
uv run langmem-eval --systems langmem --benchmark locomo --num-samples 1 --context-tokens 4096 --tokenizer cl100k_base --top-k 20 --max-output-tokens 256 --answer-temperature 0.1 --answer-style concise --empty-context abstain --skip-judge --output-dir results/locomo-protocol-v1
```

CLI 优先于环境变量。对应变量为 `EVAL_PROTOCOL`（auto/locomo/longmemeval）、
`EVAL_CONTEXT_TOKENS`、`EVAL_TOKENIZER`、`EVAL_TOP_K`、`EVAL_MAX_OUTPUT_TOKENS`、
`EVAL_TEMPERATURE`、`EVAL_ANSWER_STYLE`（concise/complete）、
`EVAL_EMPTY_CONTEXT`（abstain/answer）、`EVAL_ABSTENTION_TEXT`。
命令行对应 `--protocol`、`--context-tokens`、`--tokenizer`、`--top-k`、
`--max-output-tokens`、`--answer-temperature`、`--answer-style`、`--empty-context`、`--abstention-text`。
协议最终值完整写入结果配置；原生 baseline 不应用这些覆盖项，配置记为 null。
存储为进程内存，退出后不保留；尚未实现断点恢复。

输入是 MemEval 的 conversation/session_N 格式；原始 LongMemEval 数据应走其 loader。
MemEval 的可选依赖缺失会发出警告并跳过对应注册，运行时请显式选择系统。
运行前生成 `manifest_*.json`：固定题目 ID、原始样本 ID、题目、参考答案哈希及数据哈希。
不同方法接收同一份数据的独立副本。结果中的 `sample_id` 是清单内唯一 ID，
原数据标识保存在 `source_sample_id`；原始 `question_id` 保留，用于 LongMemEval 拒答题识别。
会话异常、缺失结果和回答异常都生成失败行，不会从分母中消失；每完成一个会话写入
`*_progress.jsonl`。这支持审计和中断后的检查，目前不自动恢复未完成的调用。

`answer_status` 与 `judge_status` 分别记录回答/裁判是否成功；错误包含阶段和异常类型。
回答执行失败的 F1 按 0 计入固定分母，不允许因空参考答案误判为正确拒答。
裁判失败的原始分数为 null，不能解释为模型答错。
汇总包含 `answer_coverage`、`judge_coverage`、`coverage`、失败计数与 `run_status`。
任一请求的回答或裁判失败都会使运行标记 `incomplete`，runner 保存结果后以退出码 2 结束。
LongMemEval 主 accuracy 在不完整运行中为 null，另提供明确命名的
`longmemeval_accuracy_failures_zero` 作为操作性指标。完整运行退出码为 0。
F1 同时报告 conversation-macro 与 question-micro；std 是会话均值的标准差，并非多次实验误差条。

## 成本记录

`token_usage` 和 `token_breakdown` 只包含观察到的方法调用，不含裁判。
`cost_accounting` 分别报告 method、judge 以及 write/retrieve/answer/judge/unclassified 阶段。
LLM 输入/输出 Token 与 embedding Token 分列；费用未知时 `reported_cost_usd=null`。
`calls_without_usage` 表示未获得 usage 的调用；它们不是免费调用。
裁判模型按实际评测分支记录：LongMemEval 使用 `LONGMEMEVAL_JUDGE_MODEL`（默认 gpt-4o），
其他评测使用 `JUDGE_MODEL`（默认 gpt-5.2）。

自动统计覆盖常见 OpenAI SDK Chat/Responses/Embeddings 的同步、异步入口和返回 usage 的流。
SDK 内部重试不分别计数；其他供应商、远程服务、本地推理与后台线程不保证自动覆盖。
因此始终标记 `coverage=partial`，不将统计称为整个系统总成本。
旧 baseline 的未拆分调用记入 unclassified；统一适配器明确划分写入、检索与回答。
普通工作线程需要显式传播 contextvars；跨进程无法自动传播阶段。

新适配器可用以下接口补充未被自动观察的调用，已被 SDK 记录的调用不要重复上报：

```python
from agents_memory.token_tracker import phase, record_external_usage

with phase("retrieve"):
    response = external_service.search(query)
    record_external_usage(
        provider="external-service", model="retriever-v1", kind="service",
        prompt_tokens=response.input_tokens, completion_tokens=response.output_tokens,
        # 没有供应商报告就不填；不要用 0 代替未知。
    )
```

目前仍需逐个审计原生 baseline 的内部异常吞掉、远程状态隔离和自身协议。
框架可以记录暴露出来的失败，不能检测第三方内部静默失败。

## 验证与来源

离线测试覆盖数据边界、真实 SDK 存储、样本隔离和上游评分；模型输出使用模拟数据。
这不代表在线抽取、模型回答或 token 统计已经验证。

- LangMem：https://github.com/langchain-ai/langmem （0.0.30）
- MemEval：https://github.com/ProsusAI/MemEval
  基础 commit：807ae6d7d8a5b76f6fe964d5a581d96c036e2ac4。
  本地改动：可选依赖缺失处理，以及 langmem 薄入口。
