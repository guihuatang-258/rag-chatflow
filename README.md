# rag-chatflow

用 **LangGraph + FastAPI + Qdrant** 复现武陵源 Dify Workflow 的初版。

目标不是把 Dify 的每个模板、赋值、JSON 解析节点逐个照搬，而是保留完整业务流程，同时把静态资源、模型调用和检索拆成可维护的代码模块。

## 当前实现

### 文本链

```text
用户问题
  ↓
读取最近 5 个历史问题
  ↓
语言识别 + 翻译/问题改写 + 场景分类
  ↓
scenic ──→ Qdrant 检索 ──→ 有结果 → RAG 回答
   │                         └→ 无结果 → 通用回答
   ├→ chat ───────────────────────────→ 通用回答
   ├→ sensitive ──────────────────────→ 固定拒答并按需翻译
   └→ identity ───────────────────────→ 固定身份话术并按需翻译
```

原 Dify 中“身份问题”和下游“大模型厂商问题”标签不一致的问题在这里统一成 `identity` 枚举。

### 图片链

```text
图片
  ↓
分类：山 / 古建筑 / 动物 / 植物 / 其他
  ↓
山或古建筑 + 有当前景点 scenic_name
  ├→ 用当前景点参考图做 VLM 匹配
  │    ├→ 命中 → JSON 景点解说词 → LLM 润色
  │    └→ 未命中 → VLM 直接识别 → LLM 润色
  └→ 其他情况 → VLM 直接识别 → LLM 润色
```

支持远程图片 URL，也支持 `/v1/chat/upload` 直接上传图片。上传图片会转换成 data URL 发送给 OpenAI-compatible 视觉模型。

## 技术栈

- Python 3.11+
- FastAPI
- LangGraph `StateGraph`
- OpenAI-compatible Chat / Vision / Embedding API
- Qdrant 本地持久化模式
- JSON 景点静态资源
- JSON 会话问题历史

Qdrant 初版无需 Docker 或独立数据库进程，数据直接写入 `.data/qdrant`。后续需要多实例部署时，可以把 Qdrant 切到独立服务；历史状态则可以把 `JsonHistoryStore` 换成 Redis 实现。

## 与原 Dify RAG 的对应关系

原 Workflow 检索参数为：

- top_k: 10
- score threshold: 0.05
- vector weight: 0.6
- keyword weight: 0.4
- rerank: disabled

初版使用 Qdrant 做 dense vector retrieval，再对候选结果执行一个轻量关键词匹配并按 `0.6 vector + 0.4 keyword` 融合排序。这样暂时不需要再引入 Elasticsearch/OpenSearch。

Dify 原知识库可以通过 `scripts/import_dify.py` 直接读取已有 document/chunk，生成 `data/knowledge.json`；当前自部署环境默认使用浏览器 Cookie 调用 Console API，详细说明见 `docs/dify-import.md`。

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置模型

```bash
cp .env.example .env
```

至少填写：

```env
OPENAI_API_KEY=your-key
```

默认配置使用阿里云 DashScope OpenAI-compatible endpoint：

```env
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
CHAT_MODEL=qwen3.7-plus
FAST_MODEL=qwen3.6-flash
VISION_MODEL=qwen3-vl-plus
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSIONS=1024
```

如果你有百炼 Workspace 专属 endpoint，建议直接把 `OPENAI_BASE_URL` 换成该地址。

### 3. 导入 Dify 知识库并构建索引

当前 Dify 环境使用 Cookie 登录态，`.env` 中填写：

```env
DIFY_BASE_URL=https://glassesai.0744trip.com/
DIFY_AUTH_MODE=cookie
DIFY_COOKIE=从已登录浏览器请求中复制的完整Cookie值
DIFY_DATASET_ID=
```

先列出可访问知识库并获取真实 UUID：

```bash
uv run python scripts/import_dify.py --list-datasets
```

把目标 UUID 填入 `DIFY_DATASET_ID` 后，只导出 Dify chunks 到 `data/knowledge.json`：

```bash
uv run python scripts/import_dify.py
```

导入后直接重建 Qdrant：

```bash
uv run python scripts/import_dify.py --reindex
```

如果只想手工维护 `data/knowledge.json`，也可以单独执行：

```bash
uv run python scripts/index_knowledge.py
```

只索引 `data/knowledge.json`，不加入景点静态解说：

```bash
uv run python scripts/index_knowledge.py --no-scenic
```

自定义知识数据格式：

```json
[
  {
    "id": "ticket-001",
    "title": "门票说明",
    "text": "这里放知识正文",
    "source": "manual",
    "metadata": {
      "category": "ticket"
    }
  }
]
```

### 4. 启动

```bash
uv run uvicorn rag_chatflow.api:app --reload --port 8000
```

检查：

```text
GET /health
```

## API

### 文本问答

```bash
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "demo-1",
    "query": "五指峰有什么特点？",
    "debug": true
  }'
```

### 图片 URL

```bash
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "demo-image",
    "image_urls": ["https://example.com/photo.jpg"],
    "scenic_name": "五指峰",
    "language": 1,
    "debug": true
  }'
```

`language` 与原 Dify Workflow 一致：1 中文，2 英文，3 日文，4 韩文，5 俄文，6 阿拉伯语，7 法语，8 西班牙语，9 葡萄牙语，10 德语，11 越南语，12 希伯来语，13 泰语，14 马来语。

### 本地上传图片

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/upload \
  -F "files=@photo.jpg" \
  -F "scenic_name=五指峰" \
  -F "language=1" \
  -F "debug=true"
```

## 目录

```text
src/rag_chatflow/
├── api.py          # FastAPI
├── workflow.py     # LangGraph 主流程
├── llm.py          # OpenAI-compatible 文本/视觉模型
├── vectorstore.py  # Qdrant 与混合排序
├── history.py      # JSON 历史问题存储
├── dify_import.py  # Dify Console/Knowledge API 导入器
├── resources.py    # 静态资源读取
├── prompts.py      # Prompt
├── models.py       # Pydantic / Graph state
└── service.py      # API 与 Workflow 之间的服务层

resources/
├── scenic_images.json
├── scenic_introductions.json
└── scenery_translations.json

data/
└── knowledge.json

scripts/
├── import_dify.py
└── index_knowledge.py
```

## 下一步

初版之后优先做这几件事：

1. 用真实 Dify Cookie 跑一次知识库导入并对齐 document/chunk 数量。
2. 用同一批测试问题同时跑 Dify 和本项目，比较路由、检索召回、回答正确率和延迟。
3. 根据数据决定是否改成 Qdrant 原生 sparse+dense hybrid retrieval 和 rerank。
4. 增加 SSE 流式响应。
5. 多实例部署时把 JSON history 换成 Redis。
