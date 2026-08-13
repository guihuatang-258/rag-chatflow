# 从 Dify 导入知识库

当前迁移策略是保留 Dify 已经切好的 chunk，不重新切分原始 PDF/Word。这样可以先只比较 Workflow、Embedding 和向量库迁移带来的差异。

## 1. 配置

复制环境变量：

```bash
cp .env.example .env
```

Dify 相关配置：

```env
DIFY_BASE_URL=https://glassesai.0744trip.com/
DIFY_DATASET_API_KEY=your-dataset-api-key
DIFY_DATASET_ID=your-dataset-id
```

`DIFY_BASE_URL` 可以填写 Dify 根地址，也可以直接填写以 `/v1` 结尾的 API 地址。导入器会自动规范成 `/v1`。

`DIFY_DATASET_API_KEY` 必须使用知识库 Knowledge API 的 API Key，不是 Chatflow/App API Key。API Key 不要提交进 Git。

## 2. 只导出 Dify chunks 到 JSON

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

## 3. 导入后直接重建 Qdrant

```bash
uv run python scripts/import_dify.py --reindex
```

这会在成功生成 `data/knowledge.json` 后直接重建本地 Qdrant collection。

默认仍会把 `resources/scenic_introductions.json` 中的景点解说一起放进 Qdrant。只想索引 Dify 知识库时：

```bash
uv run python scripts/import_dify.py --reindex --no-scenic
```

## 4. 临时覆盖知识库 ID

不想改 `.env` 时，可以只临时覆盖 dataset ID：

```bash
uv run python scripts/import_dify.py --dataset-id YOUR_DATASET_ID
```

API Key 仍建议只放 `.env`，不要作为命令行参数传入，避免进入 shell history。

## 5. 导入禁用内容

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

### 401 / 403

通常是 `DIFY_DATASET_API_KEY` 不正确，或者使用了 App API Key 而不是 Knowledge API Key。

### 404

先确认 `DIFY_DATASET_ID` 是否是当前 Dify 实例中的真实知识库 ID，以及反向代理是否暴露 `/v1` API。

### 无法连接

导入器默认请求超时 30 秒，可以在 `.env` 调整：

```env
DIFY_REQUEST_TIMEOUT_SECONDS=60
```
