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
    dify_auth_mode: str = "cookie"
    dify_cookie: str = ""
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

    def ensure_dify_config(
        self,
        dataset_id: str | None = None,
        *,
        require_dataset: bool = True,
        auth_mode: str | None = None,
    ) -> None:
        mode = (auth_mode or self.dify_auth_mode or "auto").strip().lower()
        if mode not in {"auto", "cookie", "api_key"}:
            raise RuntimeError("DIFY_AUTH_MODE 只能是 auto、cookie 或 api_key")
        if mode == "cookie" and not self.dify_cookie.strip():
            raise RuntimeError("DIFY_COOKIE 未配置，请从已登录 Dify 的浏览器请求中复制 Cookie")
        if mode == "api_key" and not self.dify_dataset_api_key.strip():
            raise RuntimeError("DIFY_DATASET_API_KEY 未配置")
        if mode == "auto" and not (self.dify_cookie.strip() or self.dify_dataset_api_key.strip()):
            raise RuntimeError("DIFY_COOKIE 和 DIFY_DATASET_API_KEY 至少配置一个")
        if require_dataset and not (dataset_id or self.dify_dataset_id).strip():
            raise RuntimeError("DIFY_DATASET_ID 未配置，可先运行 import_dify.py --list-datasets")
