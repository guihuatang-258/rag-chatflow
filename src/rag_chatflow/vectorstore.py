from __future__ import annotations

import asyncio
import hashlib
import json
import re

from openai import AsyncOpenAI, OpenAI
from qdrant_client import QdrantClient, models

from .config import Settings
from .models import KnowledgeDocument, RetrievedDocument


class QdrantRetriever:
    """Qdrant dense retrieval + lightweight lexical fusion.

    The original Dify workflow used vector 0.6 + keyword 0.4. This initial version keeps
    Qdrant as the vector database and applies the keyword component as a local rerank,
    avoiding an extra Elasticsearch/OpenSearch service.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.qdrant_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(settings.qdrant_path))
        self.embedding_client = AsyncOpenAI(
            api_key=settings.openai_api_key or "not-configured",
            base_url=settings.openai_base_url,
        )

    async def search(self, query: str) -> list[RetrievedDocument]:
        if not query.strip() or not self.collection_exists():
            return []
        self.settings.ensure_api_key()
        embedding = await self._embed_query(query)
        candidate_limit = max(self.settings.retrieval_top_k * 3, 20)
        result = await asyncio.to_thread(
            self.client.query_points,
            collection_name=self.settings.qdrant_collection,
            query=embedding,
            limit=candidate_limit,
            with_payload=True,
        )
        docs: list[RetrievedDocument] = []
        for point in result.points:
            payload = dict(point.payload or {})
            dense = max(0.0, min(1.0, float(point.score or 0.0)))
            text = str(payload.get("text", ""))
            keyword = _keyword_score(query, text)
            score = (
                self.settings.vector_weight * dense
                + self.settings.keyword_weight * keyword
            )
            if score < self.settings.retrieval_score_threshold:
                continue
            docs.append(
                RetrievedDocument(
                    id=str(payload.get("id") or point.id),
                    title=str(payload.get("title", "")),
                    text=text,
                    source=str(payload.get("source", "")),
                    metadata=dict(payload.get("metadata") or {}),
                    dense_score=dense,
                    keyword_score=keyword,
                    score=score,
                )
            )
        docs.sort(key=lambda item: item.score, reverse=True)
        return docs[: self.settings.retrieval_top_k]

    def collection_exists(self) -> bool:
        try:
            return bool(self.client.collection_exists(self.settings.qdrant_collection))
        except Exception:
            return False

    async def _embed_query(self, query: str) -> list[float]:
        response = await self.embedding_client.embeddings.create(
            model=self.settings.embedding_model,
            input=query,
            dimensions=self.settings.embedding_dimensions,
            encoding_format="float",
        )
        return list(response.data[0].embedding)


def load_index_documents(settings: Settings) -> list[KnowledgeDocument]:
    """Load only the canonical knowledge JSON used by text retrieval.

    Scenic introductions remain static resources for the image-recognition workflow and
    are intentionally not duplicated into Qdrant.
    """
    docs: list[KnowledgeDocument] = []
    if settings.knowledge_path.exists():
        raw = json.loads(settings.knowledge_path.read_text(encoding="utf-8"))
        docs.extend(KnowledgeDocument.model_validate(item) for item in raw)

    deduped = {doc.id: doc for doc in docs}
    return list(deduped.values())


def rebuild_index(settings: Settings, docs: list[KnowledgeDocument]) -> int:
    settings.ensure_api_key()
    settings.qdrant_path.parent.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(settings.qdrant_path))
    try:
        if client.collection_exists(settings.qdrant_collection):
            _ensure_compatible_collection(client, settings)
            # Qdrant Local keeps the collection's SQLite connection open. On Windows,
            # delete_collection() therefore cannot remove storage.sqlite, and its
            # ignored rmtree error causes old points to reappear in the new collection.
            # Deleting all points through Qdrant avoids that platform-specific leak.
            client.delete(
                collection_name=settings.qdrant_collection,
                points_selector=models.FilterSelector(filter=models.Filter()),
                wait=True,
            )
        else:
            client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=models.VectorParams(
                    size=settings.embedding_dimensions,
                    distance=models.Distance.COSINE,
                ),
            )

        embedder = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        batch_size = 10
        for start in range(0, len(docs), batch_size):
            batch = docs[start : start + batch_size]
            response = embedder.embeddings.create(
                model=settings.embedding_model,
                input=[doc.text for doc in batch],
                dimensions=settings.embedding_dimensions,
                encoding_format="float",
            )
            points = []
            for doc, item in zip(batch, response.data, strict=True):
                point_id = hashlib.md5(doc.id.encode("utf-8"), usedforsecurity=False).hexdigest()
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=list(item.embedding),
                        payload=doc.model_dump(),
                    )
                )
            client.upsert(collection_name=settings.qdrant_collection, points=points, wait=True)
    finally:
        client.close()
    return len(docs)


def _ensure_compatible_collection(client: QdrantClient, settings: Settings) -> None:
    vectors = client.get_collection(settings.qdrant_collection).config.params.vectors
    compatible = (
        isinstance(vectors, models.VectorParams)
        and vectors.size == settings.embedding_dimensions
        and vectors.distance == models.Distance.COSINE
    )
    if not compatible:
        raise RuntimeError(
            f"Existing Qdrant collection {settings.qdrant_collection!r} has an incompatible "
            "vector configuration; remove it while the application is stopped, then reindex"
        )


def _keyword_score(query: str, text: str) -> float:
    q_tokens = _tokens(query)
    if not q_tokens:
        return 0.0
    t_tokens = _tokens(text)
    if not t_tokens:
        return 0.0
    matched = q_tokens & t_tokens
    return len(matched) / len(q_tokens)


def _tokens(text: str) -> set[str]:
    lowered = text.lower()
    words = set(re.findall(r"[a-z0-9]+", lowered))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", lowered))
    chars = set(chinese)
    bigrams = {chinese[i : i + 2] for i in range(max(0, len(chinese) - 1))}
    return words | chars | bigrams
