from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .models import KnowledgeDocument


def normalize_dify_base_url(base_url: str) -> str:
    """Normalize a Dify host/root URL to its API `/v1` base."""
    value = base_url.strip().rstrip("/")
    if not value:
        raise ValueError("DIFY_BASE_URL 不能为空")
    if not value.endswith("/v1"):
        value = f"{value}/v1"
    return value


class DifyKnowledgeClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        dataset_id: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("DIFY_DATASET_API_KEY 未配置")
        if not dataset_id.strip():
            raise ValueError("DIFY_DATASET_ID 未配置")
        self.base_url = normalize_dify_base_url(base_url)
        self.api_key = api_key.strip()
        self.dataset_id = dataset_id.strip()
        self.timeout_seconds = timeout_seconds

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urlencode(params)}"
        request = Request(
            url,
            method="GET",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "User-Agent": "rag-chatflow/0.1",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Dify API 请求失败: HTTP {exc.code} {path}; {detail[:500]}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"无法连接 Dify API: {exc.reason}") from exc

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Dify API 返回了非 JSON 响应: {path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Dify API 返回格式异常: {path}")
        return payload

    def iter_documents(self) -> Iterable[dict[str, Any]]:
        page = 1
        limit = 100
        while True:
            payload = self._get_json(
                f"/datasets/{self.dataset_id}/documents",
                {"page": page, "limit": limit},
            )
            items = payload.get("data") or []
            if not isinstance(items, list):
                raise RuntimeError("Dify 文档列表 data 字段不是数组")
            yield from (item for item in items if isinstance(item, dict))

            if payload.get("has_more") is False:
                break
            if payload.get("has_more") is True:
                page += 1
                continue

            total_pages = _as_int(payload.get("total_pages"))
            if total_pages is not None:
                if page >= total_pages:
                    break
                page += 1
                continue

            total = _as_int(payload.get("total"))
            if total is not None and page * limit >= total:
                break
            if len(items) < limit:
                break
            page += 1

    def iter_segments(self, document_id: str) -> Iterable[dict[str, Any]]:
        page = 1
        limit = 100
        while True:
            payload = self._get_json(
                f"/datasets/{self.dataset_id}/documents/{document_id}/segments",
                {"page": page, "limit": limit},
            )
            items = payload.get("data") or []
            if not isinstance(items, list):
                raise RuntimeError(f"Dify chunks data 字段不是数组: document={document_id}")
            yield from (item for item in items if isinstance(item, dict))

            if payload.get("has_more") is False:
                break
            if payload.get("has_more") is True:
                page += 1
                continue

            total_pages = _as_int(payload.get("total_pages"))
            if total_pages is not None:
                if page >= total_pages:
                    break
                page += 1
                continue

            total = _as_int(payload.get("total"))
            if total is not None and page * limit >= total:
                break
            if len(items) < limit:
                break
            page += 1


@dataclass(slots=True)
class ImportStats:
    documents_seen: int = 0
    documents_imported: int = 0
    documents_skipped: int = 0
    segments_seen: int = 0
    segments_imported: int = 0
    segments_skipped: int = 0


def import_dataset(
    client: DifyKnowledgeClient,
    *,
    include_disabled: bool = False,
) -> tuple[list[KnowledgeDocument], ImportStats]:
    imported: list[KnowledgeDocument] = []
    stats = ImportStats()

    for document in client.iter_documents():
        stats.documents_seen += 1
        if not include_disabled and _document_disabled(document):
            stats.documents_skipped += 1
            continue

        document_id = str(document.get("id") or "").strip()
        if not document_id:
            stats.documents_skipped += 1
            continue

        document_imported = False
        for segment in client.iter_segments(document_id):
            stats.segments_seen += 1
            if not include_disabled and segment.get("enabled") is False:
                stats.segments_skipped += 1
                continue

            converted = segment_to_knowledge_document(
                client.dataset_id,
                document,
                segment,
            )
            if converted is None:
                stats.segments_skipped += 1
                continue
            imported.append(converted)
            stats.segments_imported += 1
            document_imported = True

        if document_imported:
            stats.documents_imported += 1
        else:
            stats.documents_skipped += 1

    return imported, stats


def segment_to_knowledge_document(
    dataset_id: str,
    document: dict[str, Any],
    segment: dict[str, Any],
) -> KnowledgeDocument | None:
    document_id = str(document.get("id") or "").strip()
    segment_id = str(segment.get("id") or "").strip()
    content = str(segment.get("content") or "").strip()
    answer = str(segment.get("answer") or "").strip()

    if not document_id or not segment_id or not content:
        return None

    # In Dify QA-style chunks, the answer is stored separately. Include it in the indexed
    # text so the downstream RAG prompt receives the same useful information.
    text = content
    if answer:
        text = f"{content}\n\n答案：{answer}"

    document_name = str(document.get("name") or document_id)
    metadata: dict[str, Any] = {
        "dataset_id": dataset_id,
        "document_id": document_id,
        "segment_id": segment_id,
        "document_name": document_name,
        "position": segment.get("position"),
        "keywords": segment.get("keywords") or [],
        "doc_form": document.get("doc_form"),
        "document_metadata": document.get("doc_metadata") or [],
        "dify_segment_status": segment.get("status"),
    }
    if answer:
        metadata["answer"] = answer

    return KnowledgeDocument(
        id=f"dify:{dataset_id}:{document_id}:{segment_id}",
        title=document_name,
        text=text,
        source=f"dify:{document_name}",
        metadata=metadata,
    )


def write_knowledge_json(path: Path, documents: list[KnowledgeDocument]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [document.model_dump(mode="json") for document in documents]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _document_disabled(document: dict[str, Any]) -> bool:
    if document.get("enabled") is False:
        return True
    if document.get("archived") is True:
        return True
    return False


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
