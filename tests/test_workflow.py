from pathlib import Path

import pytest

from rag_chatflow.models import ImageClassification, Intent, QueryAnalysis, RetrievedDocument
from rag_chatflow.resources import ResourceStore
from rag_chatflow.workflow import WorkflowServices, build_workflow


class FakeModels:
    def __init__(self, analysis: QueryAnalysis | None = None, image_class: str = "山", matched=None):
        self.analysis = analysis or QueryAnalysis(
            language="中文", translated_text="五指峰是什么", intent=Intent.SCENIC
        )
        self.image_class = image_class
        self.matched = matched if matched is not None else ["五指峰"]

    async def analyze_query(self, query, history):
        return self.analysis

    async def answer_with_context(self, query, language, docs):
        return f"RAG:{docs[0].title}"

    async def general_answer(self, query, language):
        return f"GENERAL:{query}"

    async def translate(self, text, language):
        return text

    async def classify_image(self, image_url):
        return ImageClassification(image_class=self.image_class)

    async def match_scenic(self, image_url, image_class, scenic_name, refs):
        return self.matched

    async def recognize_image(self, image_url, language):
        return "识别到山峰"

    async def polish_image_result(self, raw_result, language, scenic_context=""):
        return f"IMAGE:{'matched' if scenic_context else 'fallback'}"


class FakeRetriever:
    def __init__(self, docs=None):
        self.docs = docs or []

    async def search(self, query):
        return self.docs


def resources():
    root = Path(__file__).parents[1] / "resources"
    return ResourceStore.load(root)


@pytest.mark.asyncio
async def test_scenic_text_uses_rag():
    doc = RetrievedDocument(id="1", title="五指峰", text="介绍", score=0.8)
    graph = build_workflow(
        WorkflowServices(FakeModels(), FakeRetriever([doc]), resources())
    )
    result = await graph.ainvoke({"query": "五指峰是什么", "image_urls": [], "history": []})
    assert result["route"] == "text_rag"
    assert result["answer"] == "RAG:五指峰"


@pytest.mark.asyncio
async def test_sensitive_text_uses_fixed_response():
    models = FakeModels(
        QueryAnalysis(language="中文", translated_text="敏感问题", intent=Intent.SENSITIVE)
    )
    graph = build_workflow(WorkflowServices(models, FakeRetriever(), resources()))
    result = await graph.ainvoke({"query": "test", "image_urls": [], "history": []})
    assert result["route"] == "text_sensitive"
    assert "没法回答" in result["answer"]


@pytest.mark.asyncio
async def test_image_reference_match_path():
    graph = build_workflow(WorkflowServices(FakeModels(), FakeRetriever(), resources()))
    result = await graph.ainvoke(
        {
            "query": "",
            "image_urls": ["data:image/jpeg;base64,abc"],
            "scenic_name": "五指峰",
            "requested_language": 1,
            "history": [],
        }
    )
    assert result["route"] == "image_reference_answer"
    assert result["matched_scenic_names"] == ["五指峰"]


@pytest.mark.asyncio
async def test_image_without_location_falls_back_to_direct_recognition():
    graph = build_workflow(WorkflowServices(FakeModels(), FakeRetriever(), resources()))
    result = await graph.ainvoke(
        {
            "query": "",
            "image_urls": ["data:image/jpeg;base64,abc"],
            "scenic_name": None,
            "requested_language": 1,
            "history": [],
        }
    )
    assert result["route"] == "image_fallback_answer"
    assert result["answer"] == "IMAGE:fallback"
