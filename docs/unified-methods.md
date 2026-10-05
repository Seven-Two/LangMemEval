# 统一方法接口

当前只内置 **A-Mem 和 LangMem**，都在 `src/langmem_eval/methods/` 注册。
MemEval 提供数据加载、评分、实验运行和结果记录，不再提供另一套 baseline 执行入口。
其他研究方法及其专属训练代码已移除；新增方法应在明确实验需求后单独实现。

## 执行流程

```text
registry.create_backend(name, model)
  → 按顺序 ingest(Session)
  → 可选 finalize()，完成批量写入
  → retrieve(query, limit) 返回完整证据 list[str]
  → 共享 Token 预算选择
  → 共享 AnswerProtocol 和回答模型
  → 共享评分与结果保存
  → 可选 close()，释放后端拥有的资源
```

必需接口只有 `ingest` 和 `retrieve`，无需继承某个具体方法。
方法只接收历史和检索问题，不接收参考答案，不生成最终评测答案。
每段 conversation 创建独立后端；写入失败使当前对话的问题记为失败，后续对话仍继续处理。

`finalize` 是为以后需要批量写入的方法预留的可选钩子，不是第二套方法接口。
目前 A-Mem 和 LangMem 都在 `ingest` 中同步写入，不需要它。
`describe()`、`last_retrieval` 可提供可序列化且不含密钥的配置和检索轨迹。

A-Mem 另实现可选的成对缓存钩子 `restore_cached_memory(sessions)` / `save_cached_memory(sessions)`。
它们只获得历史：加载成功跳过写入，未命中则在完整写入完成后保存，再评测问题。
LangMem 和新方法不实现这些钩子也能正常运行。A-Mem 默认关闭缓存，启用和审计方式见
[缓存说明](amem.md#复用已经构建的记忆避免重复写入)。

## 代码位置

| 模块 | 职责 |
| --- | --- |
| [interfaces.py](../src/langmem_eval/interfaces.py) | 历史数据和记忆后端契约 |
| [registry.py](../src/langmem_eval/registry.py) | 唯一注册表、发现、依赖检查和创建 |
| [adapter.py](../src/langmem_eval/adapter.py) | 统一写入、检索、回答流程 |
| [methods/amem/](../src/langmem_eval/methods/amem/) | A-Mem 的记忆笔记、连接与演化 |
| [methods/langmem/](../src/langmem_eval/methods/langmem/) | LangMem manager、存储与检索 |
| [method_template/](../examples/method_template/) | 教学用扩展模板，不自动注册为 baseline |

旧 `agents_memory/systems/` 和 `system_entries()` 桥接层已删除。
`--systems` 仍是选择方法的 CLI 参数名，不代表保留旧目录或第二个注册表。

## 使用

```bash
uv sync --locked --extra dev --extra amem
uv run langmem-eval --list-methods
uv run langmem-eval --systems amem,langmem --show-config
```

未安装自定义方法时，列表应只有 `amem` 和 `langmem`；`--systems all` 也只选择这两个方法。
模型配置优先级保持 **CLI > .env > 进程环境 > 默认值**。聊天和 embedding 配置见
[统一配置](configuration.md)，A-Mem 的复现范围见 [A-Mem 指南](amem.md)。

配置好 `.env` 后运行以下命令会调用真实模型：

```bash
uv run --extra amem langmem-eval \
  --systems amem,langmem --benchmark locomo --protocol locomo \
  --data-file examples/amem_smoke.json --num-samples 1 \
  --context-tokens 6000 --top-k 20 --max-output-tokens 256 \
  --skip-judge --output-dir results/comparison
```

新结果都标记 `adapter_protocol=unified_v1`，逐题保存实际上下文与请求。
共享回答协议用于控制比较条件，不能直接等同于原论文评测设置。
成本仍按 write/retrieve/answer/judge 记录，覆盖范围为 partial，裁判开销单独统计。

## 后续扩展

复制模板到 `src/langmem_eval/methods/my_method/`，实现 `ingest/retrieve`，
在该包的 `__init__.py` 注册工厂即可，无需修改运行器或评分器。
使用 `MethodOption` 声明 CLI/.env 参数，用 `dependencies/extra` 声明可选 SDK。
额外依赖仍需添加到项目配置并更新锁文件。详细步骤见 [开发指南](development.md#注册新方法)。

修改后运行纯离线验收：

```bash
uv run --locked --extra dev python scripts/verify_amem.py
uv build --offline --wheel
uv run --locked --extra dev python scripts/verify_amem.py --guard
```

测试使用假模型，不验证在线服务、云端 GPU 或论文分数。安装包测试检查只有要求的两个内置方法，
扩展测试验证教学模板仍可接入同一套配置、回答、评分和资源管理流程。
