from rag_chatflow.dify_import import (
    DifyKnowledgeClient,
    import_dataset,
    normalize_dify_base_url,
    segment_to_knowledge_document,
)


def test_normalize_dify_base_url() -> None:
    assert normalize_dify_base_url("https://example.com/") == "https://example.com/v1"
    assert normalize_dify_base_url("https://example.com/v1/") == "https://example.com/v1"


def test_segment_to_knowledge_document_preserves_dify_metadata() -> None:
    document = {
        "id": "doc-1",
        "name": "门票说明.txt",
        "doc_form": "qa_model",
        "doc_metadata": [{"id": "m1", "name": "category", "value": "ticket"}],
    }
    segment = {
        "id": "seg-1",
        "position": 2,
        "content": "老人有优惠吗？",
        "answer": "符合条件的老人可按景区政策享受优惠。",
        "keywords": ["老人", "优惠"],
        "enabled": True,
        "status": "completed",
    }

    result = segment_to_knowledge_document("dataset-1", document, segment)

    assert result is not None
    assert result.id == "dify:dataset-1:doc-1:seg-1"
    assert "答案：" in result.text
    assert result.metadata["keywords"] == ["老人", "优惠"]
    assert result.metadata["document_id"] == "doc-1"


class FakeClient(DifyKnowledgeClient):
    def __init__(self) -> None:
        self.dataset_id = "dataset-1"

    def iter_documents(self):
        yield {"id": "doc-1", "name": "guide.txt", "enabled": True, "archived": False}
        yield {"id": "doc-2", "name": "disabled.txt", "enabled": False, "archived": False}

    def iter_segments(self, document_id: str):
        assert document_id == "doc-1"
        yield {"id": "seg-1", "content": "黄石寨介绍", "enabled": True}
        yield {"id": "seg-2", "content": "不应导入", "enabled": False}


def test_import_dataset_skips_disabled_by_default() -> None:
    docs, stats = import_dataset(FakeClient())

    assert len(docs) == 1
    assert docs[0].text == "黄石寨介绍"
    assert stats.documents_seen == 2
    assert stats.documents_imported == 1
    assert stats.documents_skipped == 1
    assert stats.segments_seen == 2
    assert stats.segments_imported == 1
    assert stats.segments_skipped == 1
