# 论文实验框架的模块边界与扩展约定

审查日期：2026-10-05。当前仅内置 A-Mem 和 LangMem，使用唯一的记忆接口和统一回答协议。
接口统一不代表严格复现各论文的原始评测设置；算法与实验语义差异须随结果报告。

## 主执行路径

```mermaid
flowchart TD
    CLI[CLI 与配置优先级] --> Registry[方法注册与参数声明]
    CLI --> Runner[运行器：数据清单、方法循环、结果落盘]
    Registry --> Adapter[统一适配器]
    Runner --> Adapter
    Adapter --> History[仅历史 Session]
    History --> Backend[方法：ingest / 可选 finalize / retrieve]
    Backend --> Evidence[完整证据字符串]
    Evidence --> Budget[Token 预算与回答协议]
    Budget --> Scoring[逐题评分]
    Scoring --> Runner
```

方法不接收参考答案，不承担回答和评分；统一适配器不进入方法内部修改记忆。
每段 conversation 创建独立后端，先完成历史写入，再执行问答，退出时释放拥有的资源。
`agents_memory` 保存 MemEval 数据、评分和运行器；`langmem_eval` 保存统一协议与方法；
这两层已有稳定边界，无需为目录数量而再次混合源码。

## 本轮发现和修正

| 目标 | 原有问题 | 当前实现与验收 |
| --- | --- | --- |
| 可扩展 | 新参数需要编辑公共 ENV_FLAGS 和 runner 中的 A-Mem 特判 | MethodOption 与 public_config 在方法包注册；模板复制后直接接入 CLI/.env/完整评测；见 test_architecture.py |
| 低耦合 | 两套注册入口与独立回答器，夹带未要求的 baseline | 仅保留 methods 注册表及 A-Mem、LangMem；SDK 延迟加载；子进程导入守卫检查 |
| 可维护 | A-Mem 的 API、JSON 策略、图更新、日志和状态提交交错 | clients / responses / evolution / backend 分工；纯演化函数只返回候选状态与诊断数据 |
| 可读性 | 凭 Schema 字典相等推断解析阶段，新方法重复 JSON 解码 | 显式 purpose 与 Session.turns()；必要字段错误带路径 |
| 资源生命周期 | 每段对话创建客户端，异常时可能不释放资源 | 可选 close()，managed_backend 保留原错误；后端释放自身创建的资源 |
| 实验审计 | 方法设置可能只有回答 trace 中有，写入失败时缺失 | 注册配置回调在调用模型前验证；config.method_settings 不依赖问答成功 |
| 复用 | A-Mem 和 LangMem 的向量返回校验不同 | 共用 validate_vectors，支持本地/API embedding；LangMem 用公开 dimensions 属性 |

## 新记忆机制放在哪里

从 `examples/method_template` 复制到 `src/langmem_eval/methods/my_method`，方法包自己管理：

```text
my_method/
  __init__.py       # 轻量注册、参数绑定、公开配置回调
  config.py         # 有默认值和校验的 settings
  backend.py        # 每段对话状态、ingest、retrieve、可选 finalize/close
  ...              # 根据实际复杂度添加纯算法、提示词或客户端模块
```

模板的 `--my-method-window` / `MY_METHOD_WINDOW` 是可运行的配置示例，近期历史选择只是教学算法。
研究概率状态、信息价值或记忆更新策略时，把状态转移写成纯函数，输入当前状态和新证据，
返回候选状态；模型调用与存储提交放在 backend。不要通过继承并覆写 A-Mem 私有方法构建新机制。
如果确实研究 A-Mem 变体，可以显式组合它的 `apply_evolution`，并使用新的注册名和实现版本。

可复用的接口：

- `interfaces.Session`：历史聊天消息与 `turns()` 的类型化读取；`HistoricalTurn` 保留说话人、时间、来源。
- `registry.register_method` / `MethodOption`：工厂、算法说明、CLI/env 绑定和无模型的配置审计。
- `embeddings.create_embedder` / `validate_vectors`：embedding 路由及向量有效性检查。
- `agents_memory.diagnostics.event/stage`：阶段与进度；不要在纯算法里打印提示词和响应全文。
- `agents_memory.usage.phase/record_external_usage`：记录阶段与未被 SDK 自动观察的成本。
- `backend.describe()` / `last_retrieval`：方法版本、参数、诊断和检索轨迹，可 JSON 序列化且不包含密钥。

`retrieve()` 返回排序后的完整证据，限额表示证据条数。图方法可把种子及关联记忆作为一条组证据，
但必须在 describe 中声明；共享预算不会截断字符串，也不会假定不同分组策略完全等价。
模型、embedding 和 settings 可注入构造器；注入资源由调用方关闭，后端只关闭自己创建的资源。

## 保持实验含义稳定

- 新方法采用新注册名，算法行为变化更新实现版本。新增方法以明确的实验需求为前提，
  不因整理接口而自动加入其他 baseline；接口说明见 [统一方法说明](unified-methods.md)。
- A-Mem v4 的非候选链接过滤属于明确记录的适配差异；原版并不执行这种过滤。
- 全部方法结果标记 `adapter_protocol=unified_v1`；旧 `native` 结果不代表新的统一协议结果。
- 所有方法使用同一个冻结题目清单。失败计入固定分母，coverage 和 incomplete 必须随分数一起报告。
- 方法运行顺序和配置优先级仍是显式实验条件。当前 CLI 临时覆盖进程环境，只支持顺序执行；
  并行实验使用独立进程和输出目录。需要同进程并发时，应另行设计不可变运行上下文。
- A-Mem 可缓存完整历史状态，复用时保留实际向量矩阵和链接；命中状态及构建 ID 写入回答轨迹。
  这不等于写入中途/逐题进度断点恢复。缓存命中成本不包含上一次构建，应单独报告。
- 当前没有通用训练循环；AML 服务的持久化生命周期与离线 benchmark 分开。
  这些是功能边界，不应在论文中声称已经支持。可在新方法包中加入训练/状态存取组件，
  但训练数据与测试题答案的隔离仍须单独验证。
- 成本报告只覆盖被观察到的调用，仍标记 partial；不声称捕获所有远程服务内部开销。

## 验证入口与证据范围

```bash
# 仅使用脚本化模型和向量；脚本屏蔽真实密钥与 .env。
uv run --locked --extra dev python scripts/verify_amem.py
uv run --locked --extra dev python scripts/verify_amem.py --guard

# 本机缓存已具备构建依赖时，验证可分发安装包。
uv build --offline --wheel
```

测试覆盖：新方法复制注册、参数来源/恢复、结果记录、状态隔离、参考答案隔离、懒加载、
资源释放、纯状态转移不修改输入、A-Mem 行为与容错、真实 LangMem SDK 假模型存储、固定分母、
Token 预算、阶段成本、AML 隔离、日志/进度，以及 wheel 在仓库外导入并显示 CLI 帮助。
这些测试证明工程边界和行为，不证明真实模型准确率、供应商兼容性或论文结论；正式实验仍需
固定数据版本、模型版本和配置，并运行多次实验及消融。

`tests/conftest.py` 默认禁用真实 .env 和外部 socket 连接，仅允许本地事件循环所需的 loopback。
测试使用 fake SDK 或 MockTransport，不通过 loopback 调用真实推理服务。

验证还检查内置方法仅为 A-Mem、LangMem，安装包不包含其他方法或专属训练代码，
并检查新增方法可通过模板注册，无需编辑核心运行器。后续改变算法或接口后需重跑上述命令。

2026-10-05 收敛方法范围后的验证：A-Mem 行为验收 10/10、其余离线回归 139/139 通过；
锁文件离线校验、wheel 构建与仓库外导入通过，75 个本地文档链接有效。
未调用真实模型或下载权重；测试有一条 Starlette/AnyIO 弃用警告。

同日增加 A-Mem 完整记忆缓存后：10 项行为验收和 165 项其他离线回归通过（合计 175 项），
包括 26 项缓存测试；离线 wheel 构建、文档链接和差异空白检查通过。
