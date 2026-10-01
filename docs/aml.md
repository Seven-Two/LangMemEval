# AML 文本赛道适配

本适配提供 `/add`、`/search`、`/health`。平台负责最终回答和评分；
服务只返回记忆证据，不调用本地 MemEval 的回答链。
契约参考：https://agentmemoryleaderboard.ai/api-guide ，2026-09-22。

## 本地启动

在仓库根目录：

```powershell
uv sync --locked --extra dev --extra aml
uv run pytest -q
# 如果已有 .env，请手工合并 .env.example 中的 AML 配置，不要覆盖原文件。
# 填写配置后：
uv run langmem-aml
```

`.env.example` 中的 `AML_API_KEY` 是你服务的鉴权密钥，与模型供应商密钥
以及官方颁发的评测 Key 都不是一回事。应自行生成长随机值。
LLM 默认 gpt-4o-mini，embedding 默认 text-embedding-v4。
embedding 供应商的 OpenAI-compatible 地址、密钥和支持的维度必须明确配置；
1024 只是配置默认值，尚未在线验证供应商兼容性。两种模型分别使用自己的地址与密钥。
默认只监听 127.0.0.1:8000，不会自动公开到互联网。

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
$headers = @{Authorization = 'Bearer 你的AML_API_KEY'}
$body = @{
    request_id = 'local:request:1'
    user_id = 'local:user:1'
    session_id = 'local:session:1'
    messages = @(@{role='user'; content='Alice lives in Berlin'; timestamp=1704067200000})
} | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8000/add -Method Post -Headers $headers -ContentType 'application/json' -Body $body
$query = @{query='Where does Alice live?'; user_id='local:user:1'; top_k=100} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/search -Method Post -Headers $headers -ContentType 'application/json' -Body $query
```

Add/Search 会产生模型费用；health 和离线测试不会。
支持 Bearer、Token、X-Api-Key，health 无需鉴权。文本和代码字符串可以解析，
本版针对文本赛道验证；图片数组会返回 422，多模态未实现。
`options` 可接收但当前基线不使用它改写查询。

## 持久化及重试

SQLite 保存每个用户的完整记忆快照、原始 Add 请求、已完成请求回执和 embedding 缓存。
成功回执与快照在同一事务提交后才返回。相同 request_id + 相同正文返回已有回执；
正文改变返回 409。失败或进程中断时未提交内容回滚，可重试；外部模型调用费用
无法随数据库事务回滚，因此重试可能再次产生费用。
同一 user_id 跨 session 累积，不同用户的记忆与缓存隔离。Search 返回稳定 ID、
内容和可选分数，不超过 top_k。保存的快照不依赖随机 namespace。
修改模型、服务版本或方法需使用新数据库，避免混用旧状态。
正式材料应记录 commit、AML_SYSTEM_VERSION、模型和 embedding 配置。

## 新方法

现有注册器新增可选 `aml_factory`，在方法的装饰器中传入工厂函数即可。
设置 `AML_METHOD=你的注册名`；不支持 AML 持久化协议的方法会在启动时明确报错。
工厂签名：`factory(settings, user_id, snapshot, connection)`。
对象应实现：

- `add(AddRequest)`：只修改当前事务中的工作状态。
- `search(SearchRequest) -> list[Evidence]`：只返回检索证据。
- `snapshot() -> JSON可序列化对象`：包含下一次请求恢复所需的全部算法状态。

可继承 AMLLangMemBackend 修改 add/search，再注册 aml_factory；参考
`src/langmem_eval/methods/langmem.py`。本地 ingest/retrieve 接口仍保留。
自定义方法不能在事务外提交状态或写入其他用户空间，否则破坏重试及隔离保证。

## 容量边界与尚未完成的验证

本版是可持久化的低并发接入基线，不是已经过 Full 压测的服务。
SQLite 全局写锁覆盖模型处理，忙碌请求返回 503；建议 Smoke 并发设为 1。
每次操作恢复该用户全部记忆，embedding 有缓存，但重建索引仍为 O(N)，
数据增长后应迁移到持久化向量数据库与按用户加锁的事务机制。
没有自动清理评测记录，需规划磁盘容量与数据保留。
需要部署 HTTPS、公网可达、网关超时和请求大小限制，随后进行官方 Smoke；
这些部署与在线评测尚未执行。不要直接用开发服务宣称满足正式赛道容量。
