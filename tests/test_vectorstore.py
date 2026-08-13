import json
from types import SimpleNamespace

from qdrant_client import QdrantClient, models

from rag_chatflow import vectorstore
from rag_chatflow.config import Settings
from rag_chatflow.models import KnowledgeDocument
from rag_chatflow.vectorstore import load_index_documents, rebuild_index


def test_load_index_documents_does_not_index_scenic_introductions(tmp_path) -> None:
    knowledge_path = tmp_path / "knowledge.json"
    resource_dir = tmp_path / "resources"
    resource_dir.mkdir()

    knowledge_path.write_text(
        json.dumps(
            [
                {
                    "id": "dify:dataset:doc:segment",
                    "title": "知识库内容",
                    "text": "五指峰是黄石寨景区的代表性景点。",
                    "source": "dify:test",
                    "metadata": {},
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (resource_dir / "scenic_introductions.json").write_text(
        json.dumps({"五指峰": "这段静态介绍只用于图片链。"}, ensure_ascii=False),
        encoding="utf-8",
    )

    settings = Settings(knowledge_path=knowledge_path, resource_dir=resource_dir)
    docs = load_index_documents(settings)

    assert [doc.id for doc in docs] == ["dify:dataset:doc:segment"]
    assert all(not doc.id.startswith("scenic:") for doc in docs)


def test_rebuild_index_removes_points_missing_from_new_documents(tmp_path, monkeypatch) -> None:
    settings = Settings(
        openai_api_key="test-key",
        embedding_dimensions=3,
        qdrant_path=tmp_path / "qdrant",
        qdrant_collection="test_knowledge",
    )
    client = QdrantClient(path=str(settings.qdrant_path))
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
    )
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=[
            models.PointStruct(
                id="8fc89c7f-33dc-4f1d-b95e-90a62d721d11",
                vector=[1.0, 0.0, 0.0],
                payload={
                    "id": "scenic:五指峰",
                    "title": "五指峰",
                    "text": "旧静态介绍",
                    "source": "resources/scenic_introductions.json",
                    "metadata": {"type": "scenic_introduction"},
                },
            )
        ],
        wait=True,
    )
    client.close()

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            self.embeddings = self

        def create(self, *, input, **kwargs):
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.0, 1.0, 0.0]) for _ in input])

    monkeypatch.setattr(vectorstore, "OpenAI", FakeOpenAI)
    docs = [
        KnowledgeDocument(
            id="dify:dataset:doc:segment",
            title="知识库内容",
            text="五指峰是黄石寨景区的代表性景点。",
            source="dify:test",
        )
    ]

    assert rebuild_index(settings, docs) == 1

    client = QdrantClient(path=str(settings.qdrant_path))
    points, next_offset = client.scroll(
        collection_name=settings.qdrant_collection,
        limit=10,
        with_payload=True,
        with_vectors=False,
    )
    client.close()

    assert next_offset is None
    assert [point.payload["id"] for point in points] == ["dify:dataset:doc:segment"]
