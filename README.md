# AI 预测智能体 v1

一个从 0 到 1 的预测系统原型，采用 `FastAPI + SQLite + 单页前端`，实现了：

- 离散候选题的创建与管理
- 异步预测任务执行
- 单模型多角色的 `Planner / Researcher / Reducer / Judge` 流程
- ReAct 风格精简轨迹摘要持久化
- 概率分布、直接答案、置信依据、冲突说明、反事实脆弱性输出
- 到期后人工结算与 `accuracy / Brier score` 评测

## 启动

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 启动服务

```bash
uvicorn app.main:app --reload
```

3. 打开浏览器

访问 `http://127.0.0.1:8000`

## API

- `POST /api/questions`
- `POST /api/questions/{id}/forecast`
- `GET /api/runs/{run_id}`
- `GET /api/questions/{id}/result`
- `POST /api/questions/{id}/resolve`
- `GET /api/questions/{id}/evaluation`

## 当前默认实现说明

- `LLMClient.generate_structured(...)` 是本地占位实现，接口已固定，可直接替换成真实模型调用。
- `SearchProvider.search(...)` 和 `PageReader.fetch(...)` 是本地可运行的 synthetic adapter，用于保证没有 API key、没有联网时仍能完成整条链路。
- Map 阶段采用单服务内并发任务，不依赖外部队列。


## 使用示例（PowerShell）

在启动服务前配置 API key；若不配置，将使用本地占位实现（无需联网也可跑通流程）。

```powershell
$env:LLM_PROVIDER="deepseek"
$env:DEEPSEEK_API_KEY="你的_deepseek_key"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-chat"

$env:FORECAST_SEARCH_PROVIDER="exa"
$env:FORECAST_CONTENT_PROVIDER="exa"
$env:EXA_API_KEY="你的_exa_key"
```
