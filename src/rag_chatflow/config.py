from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    openai_api_key: str = ""

    chat_model: str = "qwen3.7-plus"
    fast_model: str = "qwen3.6-flash"
    vision_model: str = "qwen3-vl-plus"
    embedding_model: str = "text-embedding-v4"
    embedding_dimensions: int = 1024
    enable_web_search: bool = True

    dify_base_url: str = "https://glassesai.0744trip.com/"
    dify_dataset_api_key: str = ""
    dify_dataset_id: str = ""
    dify_request_timeout_seconds: float = 30.0

    qdrant_path: Path = Path(".data/qdrant")
    qdrant_collection: str = "wulingyuan_knowledge"
    history_path: Path = Path(".data/conversations.json")
    resource_dir: Path = Path("resources")
    knowledge_path: Path = Path("data/knowledge.json")

    retrieval_top_k: int = 10
    retrieval_score_threshold: float = 0.05
    vector_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.4, ge=0.0, le=1.0)

    def ensure_api_key(self) -> None:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY 未配置，请先复制 .env.example 为 .env 并填写 API Key")

    def ensure_dify_config(self, dataset_id: str | None = None) -> None:
        if not self.dify_dataset_api_key:
            raise RuntimeError(
                "DIFY_DATASET_API_KEY 未配置，请在 Dify 知识库的 API 页面创建知识库 API Key"
            )
        if not (dataset_id or self.dify_dataset_id).strip():
            raise RuntimeError("DIFY_DATASET_ID 未配置，请填写要迁移的知识库 ID")
