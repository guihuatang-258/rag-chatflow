from __future__ import annotations

import base64
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .config import Settings
from .history import JsonHistoryStore
from .llm import OpenAICompatibleGateway
from .models import ChatRequest, ChatResponse
from .resources import ResourceStore
from .service import ChatFlowService
from .vectorstore import QdrantRetriever
from .workflow import WorkflowServices, build_workflow


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    resources = ResourceStore.load(settings.resource_dir)
    model_gateway = OpenAICompatibleGateway(settings, resources)
    retriever = QdrantRetriever(settings)
    graph = build_workflow(
        WorkflowServices(models=model_gateway, retriever=retriever, resources=resources)
    )
    service = ChatFlowService(graph, JsonHistoryStore(settings.history_path))

    app = FastAPI(title="rag-chatflow", version="0.1.0")
    app.state.settings = settings
    app.state.chatflow = service
    app.state.retriever = retriever

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "vector_collection_ready": retriever.collection_exists(),
            "collection": settings.qdrant_collection,
        }

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        return await service.chat(request)

    @app.post("/v1/chat/upload", response_model=ChatResponse)
    async def chat_upload(
        files: Annotated[list[UploadFile], File()],
        query: Annotated[str, Form()] = "",
        thread_id: Annotated[str | None, Form()] = None,
        scenic_name: Annotated[str | None, Form()] = None,
        language: Annotated[int, Form()] = 1,
        debug: Annotated[bool, Form()] = False,
    ) -> ChatResponse:
        if not files:
            raise HTTPException(status_code=400, detail="至少上传一张图片")
        if len(files) > 5:
            raise HTTPException(status_code=400, detail="最多上传5张图片")
        image_urls = []
        for upload in files:
            if not (upload.content_type or "").startswith("image/"):
                raise HTTPException(status_code=400, detail=f"{upload.filename} 不是图片")
            raw = await upload.read()
            encoded = base64.b64encode(raw).decode("ascii")
            image_urls.append(f"data:{upload.content_type};base64,{encoded}")
        request = ChatRequest(
            thread_id=thread_id,
            query=query,
            image_urls=image_urls,
            scenic_name=scenic_name,
            language=language,
            debug=debug,
        )
        return await service.chat(request)

    return app


app = create_app()
