import json

from rag_chatflow.config import Settings
from rag_chatflow.vectorstore import load_index_documents


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
