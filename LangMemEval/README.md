# LangMem + MemEval

中性的记忆评测整合工程。LangMem 负责记忆抽取、更新和检索；MemEval
负责数据加载、评分和结果输出。本工程不引入额外研究方法。

## 开始使用

在 PowerShell 中运行：

```powershell
cd <仓库目录>/LangMemEval
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
- `src/langmem_eval/cli.py`：调用相邻 MemEval 源码中的统一运行器。
- `../MemEval/src/agents_memory/systems/langmem.py`：只有导入语句的发现入口。

两个本地包通过 uv 可编辑安装。修改上述源码，重新启动评测即可；无需复制。
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
所有方法共用 adapter 的回答/评分逻辑和上下文预算；方法元数据写入运行结果。

## 实验约定（评测流程）

按 session 编号依次导入历史，保留说话人、时间和来源 ID；写入完成后才评测。
每个样本独立 namespace/store，测试问题和答案不写回记忆。
默认管理器删除关闭、更新检索 query_limit=5；回答 top_k=20。
`LANGMEM_TOP_K` 和 `LANGMEM_MAX_CONTEXT_CHARS` 可覆盖回答检索条数和字符预算
（默认 20 / 24000）。后者不是 token 预算。
回答温度 0.1、max_tokens=50；修改时需与对照组对齐。
存储为进程内存，退出后不保留；尚未实现断点恢复。

输入是 MemEval 的 conversation/session_N 格式；原始 LongMemEval 数据应走其 loader。
MemEval 的可选依赖缺失会发出警告并跳过对应注册，运行时请显式选择系统。
官方运行器可能跳过失败样本，正式报告必须核对完成样本数。

## 验证与来源

离线测试覆盖数据边界、真实 SDK 存储、样本隔离和上游评分；模型输出使用模拟数据。
这不代表在线抽取、模型回答或 token 统计已经验证。

- LangMem：https://github.com/langchain-ai/langmem （0.0.30）
- MemEval：https://github.com/ProsusAI/MemEval
  基础 commit：807ae6d7d8a5b76f6fe964d5a581d96c036e2ac4。
  本地改动：可选依赖缺失处理，以及 langmem 薄入口。
