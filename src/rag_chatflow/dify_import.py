from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .models import KnowledgeDocument

DifyAuthMode = Literal["auto", "cookie", "api_key"]


def normalize_dify_root_url(base_url: str) -> str:
    """Normalize a Dify URL to its deployment root, without API suffixes."""
    value = base_url.strip().rstrip("/")
    if not value:
        raise ValueError("DIFY_BASE_URL 不能为空")
    for suffix in ("/console/api", "/v1"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value.rstrip("/")


def normalize_cookie(cookie: str) -> str:
    value = cookie.strip()
    if value.lower().startswith("cookie:"):
        value = value.split(":", 1)[1].strip()
    return value


def extract_cookie_value(cookie: str, name_suffix: str) -> str:
    """Return the first cookie whose name ends with ``name_suffix``.

    Dify may use the plain ``csrf_token`` cookie name or the secure
    ``__Host-csrf_token`` form. Matching by suffix mirrors Dify's own
    console client.
    """
    normalized = normalize_cookie(cookie)
    for part in normalized.split(";"):
        name, separator, value = part.strip().partition("=")
        if separator and name.strip().endswith(name_suffix):
            return value.strip()
    return ""


class DifyKnowledgeClient:
    def __init__(
        self,
        base_url: str,
        dataset_id: str = "",
        *,
        cookie: str = "",
        api_key: str = "",
        auth_mode: DifyAuthMode = "auto",
        timeout_seconds: float = 30.0,
    ) -> None:
        self.root_url = normalize_dify_root_url(base_url)
        self.cookie = normalize_cookie(cookie)
        self.api_key = api_key.strip()
        self.csrf_token = extract_cookie_value(self.cookie, "csrf_token")
        self.auth_mode = self._resolve_auth_mode(auth_mode)
        self.dataset_id = dataset_id.strip()
        self.timeout_seconds = timeout_seconds
        self.base_url = (
            f"{self.root_url}/console/api"
            if self.auth_mode == "cookie"
            else f"{self.root_url}/v1"
        )

    def _resolve_auth_mode(self, auth_mode: DifyAuthMode) -> Literal["cookie", "api_key"]:
        if auth_mode not in {"auto", "cookie", "api_key"}:
            raise ValueError("DIFY_AUTH_MODE 只能是 auto、cookie 或 api_key")
        if auth_mode == "cookie":
            self._ensure_cookie_auth()
            return "cookie"
        if auth_mode == "api_key":
            if not self.api_key:
                raise ValueError("DIFY_DATASET_API_KEY 未配置")
            return "api_key"
        if self.cookie:
            self._ensure_cookie_auth()
            return "cookie"
        if self.api_key:
            return "api_key"
        raise ValueError("DIFY_COOKIE 和 DIFY_DATASET_API_KEY 至少配置一个")

    def _ensure_cookie_auth(self) -> None:
        if not self.cookie:
            raise ValueError("DIFY_COOKIE 未配置")
        if not self.csrf_token:
            raise ValueError(
                "DIFY_COOKIE 中未找到 csrf_token；请从浏览器已登录 Dify 的同一条 "
                "/console/api 请求中复制完整 Cookie"
            )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "rag-chatflow/0.1",
        }
        if self.auth_mode == "cookie":
            headers["Cookie"] = self.cookie
            headers["X-CSRF-Token"] = self.csrf_token
            headers["Referer"] = f"{self.root_url}/datasets"
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urlencode(params, doseq=True)}"
        request = Request(url, method="GET", headers=self._headers())
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            auth_hint = ""
            if self.auth_mode == "cookie" and exc.code in {401, 403}:
                if "csrf token" in detail.lower():
                    auth_hint = (
                        "；CSRF 校验失败：导入器已从 DIFY_COOKIE 自动读取 csrf_token，"
                        "请确认 Cookie 来自同一条已登录 /console/api 请求且仍在有效期内"
                    )
                else:
                    auth_hint = (
                        "；Cookie 登录态可能已过期，请从浏览器已登录 Dify 的请求中重新复制完整 Cookie"
                    )
            raise RuntimeError(
                f"Dify API 请求失败: HTTP {exc.code} {path}; {detail[:500]}{auth_hint}"
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

    def iter_datasets(self) -> Iterable[dict[str, Any]]:
        page = 1
        limit = 100
        while True:
            payload = self._get_json("/datasets", {"page": page, "limit": limit})
            items = _payload_items(payload, "Dify 知识库列表")
            yield from items
            if not _has_next_page(payload, page=page, limit=limit, item_count=len(items)):
                break
            page += 1

    def iter_documents(self) -> Iterable[dict[str, Any]]:
        self._ensure_dataset_id()
        page = 1
        limit = 100
        while True:
            payload = self._get_json(
                f"/datasets/{self.dataset_id}/documents",
                {"page": page, "limit": limit},
            )
            items = _payload_items(payload, "Dify 文档列表")
            yield from items
            if not _has_next_page(payload, page=page, limit=limit, item_count=len(items)):
                break
            page += 1

    def iter_segments(self, document_id: str) -> Iterable[dict[str, Any]]:
        self._ensure_dataset_id()
        page = 1
        limit = 100
        while True:
            payload = self._get_json(
                f"/datasets/{self.dataset_id}/documents/{document_id}/segments",
                {"page": page, "limit": limit},
            )
            items = _payload_items(payload, f"Dify chunks: document={document_id}")
            yield from items
            if not _has_next_page(payload, page=page, limit=limit, item_count=len(items)):
                break
            page += 1

    def _ensure_dataset_id(self) -> None:
        if not self.dataset_id:
            raise RuntimeError("DIFY_DATASET_ID 未配置，请先用 --list-datasets 查看真实知识库 ID")


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


def _payload_items(payload: dict[str, Any], label: str) -> list[dict[str, Any]]:
    items = payload.get("data") or []
    if not isinstance(items, list):
        raise RuntimeError(f"{label} data 字段不是数组")
    return [item for item in items if isinstance(item, dict)]


def _has_next_page(payload: dict[str, Any], *, page: int, limit: int, item_count: int) -> bool:
    has_more = payload.get("has_more")
    if isinstance(has_more, bool):
        return has_more

    total_pages = _as_int(payload.get("total_pages"))
    if total_pages is not None:
        return page < total_pages

    total = _as_int(payload.get("total"))
    if total is not None:
        return page * limit < total

    return item_count >= limit


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
