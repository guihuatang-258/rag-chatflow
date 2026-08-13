from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator


class Intent(StrEnum):
    SCENIC = "scenic"
    CHAT = "chat"
    SENSITIVE = "sensitive"
    IDENTITY = "identity"


class QueryAnalysis(BaseModel):
    language: str = "中文"
    translated_text: str
    intent: Intent


class ImageClassification(BaseModel):
    image_class: Literal["山", "古建筑", "动物", "植物", "其他"]


class KnowledgeDocument(BaseModel):
    id: str
    title: str = ""
    text: str
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedDocument(KnowledgeDocument):
    dense_score: float = 0.0
    keyword_score: float = 0.0
    score: float = 0.0


class ChatRequest(BaseModel):
    thread_id: str | None = None
    query: str = ""
    image_urls: list[str] = Field(default_factory=list, max_length=5)
    scenic_name: str | None = None
    language: int = Field(default=1, ge=1, le=14)
    debug: bool = False

    @model_validator(mode="after")
    def validate_input(self) -> "ChatRequest":
        if not self.query.strip() and not self.image_urls:
            raise ValueError("query 和 image_urls 至少需要一个")
        return self


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    route: str
    language: str | None = None
    intent: Intent | None = None
    matched_scenic_names: list[str] = Field(default_factory=list)
    retrieved_count: int = 0
    debug: dict[str, Any] | None = None


class ChatState(TypedDict, total=False):
    query: str
    image_urls: list[str]
    scenic_name: str | None
    requested_language: int
    history: list[str]

    language: str
    translated_text: str
    intent: Intent
    retrieved_docs: list[RetrievedDocument]

    image_class: str
    image_language: str
    matched_scenic_names: list[str]
    image_raw_result: str

    answer: str
    route: str
    error: str
