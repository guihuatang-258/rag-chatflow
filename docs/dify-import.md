# 从 Dify 导入知识库

当前迁移策略是保留 Dify 已经切好的 chunk，不重新切分原始 PDF/Word。这样可以先只比较 Workflow、Embedding 和向量库迁移带来的差异。

## 1. Cookie / Console API 模式（当前环境默认）

你的 Dify 地址是：

```text
https://glassesai.0744trip.com/
```

当前环境使用浏览器登录 Cookie 验证，因此默认走 Dify Console API：

```text
/console/api/datasets/{dataset_id}/documents
/console/api/datasets/{dataset_id}/documents/{document_id}/segments
```

先复制环境变量：

```bash
cp .env.example .env
```

配置：

```env
DIFY_BASE_URL=https://glassesai.0744trip.com/
DIFY_AUTH_MODE=cookie
DIFY_COOKIE=你的完整Cookie请求头内容
DIFY_DATASET_ID=真实知识库UUID
```

`DIFY_COOKIE` 建议从浏览器开发者工具获取：登录 Dify → Network → 打开任意 `/console/api/...` 请求 → Request Headers → 复制 `Cookie` 的值。可以只复制值，也可以连 `Cookie:` 前缀一起复制，导入器会自动处理。

典型 Cookie 类似：

```text
locale=zh-Hans; csrf_token=...; access_token=...; refresh_token=...
```

Dify Console API 要求 `X-CSRF-Token` 与 Cookie 中的 `csrf_token` 一致。导入器会自动从 `DIFY_COOKIE` 中提取名字以 `csrf_token` 结尾的 Cookie（同时兼容 `__Host-csrf_token`），并自动发送：

```text
X-CSRF-Token: <cookie 中的 csrf_token>
```

因此**不需要单独配置 CSRF Token**。如果 Cookie 中没有 `csrf_token`，导入器会在发请求前直接报错。

Cookie 属于登录凭证，只放在本机 `.env`，不要粘贴到代码、Issue、PR 或提交进 Git。Cookie 或其中的 CSRF Token 失效后，重新从同一条已登录 `/console/api/...` 请求复制完整 Cookie 即可。

## 2. 查找真实知识库 ID

Dify Workflow DSL 里保存的 dataset ID 可能经过加密，不能直接拿来调用 Console API。可以用当前 Cookie 列出可访问知识库：

```bash
uv run python scripts/import_dify.py --list-datasets
```

输出示例：

```text
7f8d...-...	武陵源知识库 documents=12
```

把目标知识库的 UUID 填入：

```env
DIFY_DATASET_ID=7f8d...-...
```

也可以运行时临时指定：

```bash
uv run python scripts/import_dify.py --dataset-id YOUR_DATASET_ID
```

## 3. 导出 Dify chunks 到 JSON

```bash
uv run python scripts/import_dify.py
```

导入器会：

1. 分页读取知识库中的所有文档；
2. 分页读取每个文档已有的 chunk/segment；
3. 默认跳过 disabled 或 archived 文档，以及 disabled chunk；
4. 保留 document/segment ID、position、keywords、doc_form 等元数据；
5. 写入 `data/knowledge.json`。

Dify QA 类型的 chunk 如果包含单独的 `answer` 字段，会把答案附加到索引文本中，同时在 metadata 中保留原始 answer。

## 4. 导入后直接重建 Qdrant

```bash
uv run python scripts/import_dify.py --reindex
```

Qdrant 的文本 RAG 索引现在只来源于 `data/knowledge.json`。`resources/scenic_introductions.json` 不再重复写入 Qdrant，它只保留给图片识别链路在景点匹配命中后生成景点介绍。

如果已经有本地 `data/knowledge.json`，只想重新构建 Qdrant，也可以运行：

```bash
uv run python scripts/index_knowledge.py
```

## 5. API Key 模式（备用）

如果以后启用了 Dify Knowledge Service API Key，可以切回 `/v1`：

```env
DIFY_AUTH_MODE=api_key
DIFY_DATASET_API_KEY=dataset-xxxx
DIFY_DATASET_ID=...
```

也可以用 `DIFY_AUTH_MODE=auto`：配置了 Cookie 时优先使用 Cookie，否则使用 API Key。

## 6. 导入禁用内容

默认行为尽量贴近线上检索：不导入 disabled/archived 内容。如果确实需要完整备份：

```bash
uv run python scripts/import_dify.py --include-disabled
```

## 输出格式

`data/knowledge.json` 中每个 Dify chunk 会转成一个独立记录：

```json
[
  {
    "id": "dify:dataset-id:document-id:segment-id",
    "title": "门票说明.txt",
    "text": "chunk 原文",
    "source": "dify:门票说明.txt",
    "metadata": {
      "dataset_id": "dataset-id",
      "document_id": "document-id",
      "segment_id": "segment-id",
      "document_name": "门票说明.txt",
      "position": 1,
      "keywords": ["门票", "优惠"],
      "doc_form": "text_model",
      "document_metadata": [],
      "dify_segment_status": "completed"
    }
  }
]
```

## 常见错误

### `401: CSRF token is missing or invalid`

这通常不是“Cookie 变量没传”，而是以下几种情况之一：

- Cookie 中没有 `csrf_token`；
- `csrf_token` 和 `access_token` 不是从同一登录会话复制的；
- Cookie/CSRF Token 已过期；
- 复制 Cookie 时漏掉了部分内容。

导入器会自动把 Cookie 中的 `csrf_token` 放到 `X-CSRF-Token`，因此出现该错误时，直接重新登录 Dify，并从**同一条 `/console/api/...` 请求**复制完整 Cookie。

### 其他 401 / 403

Cookie 模式下一般表示浏览器登录态已经失效，重新登录 Dify 并复制最新 Cookie。

API Key 模式下则检查 `DIFY_DATASET_API_KEY` 是否为 Knowledge API Key。

### 404

Cookie/Console API 模式要求真实的知识库 UUID。建议先执行：

```bash
uv run python scripts/import_dify.py --list-datasets
```

不要直接使用 DSL 中可能加密过的 dataset ID。

### 无法连接

导入器默认请求超时 30 秒，可以在 `.env` 调整：

```env
DIFY_REQUEST_TIMEOUT_SECONDS=60
```
