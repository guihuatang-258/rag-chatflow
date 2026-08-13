from __future__ import annotations

import uuid

from .history import HistoryStore
from .models import ChatRequest, ChatResponse, ChatState


class ChatFlowService:
    def __init__(self, graph, history_store: HistoryStore):
        self.graph = graph
        self.history_store = history_store

    async def chat(self, request: ChatRequest) -> ChatResponse:
        thread_id = request.thread_id or str(uuid.uuid4())
        history = self.history_store.recent_queries(thread_id, limit=5)
        state: ChatState = {
            "query": request.query.strip(),
            "image_urls": request.image_urls,
            "scenic_name": request.scenic_name,
            "requested_language": request.language,
            "history": history,
        }
        result = await self.graph.ainvoke(state)

        if not request.image_urls and request.query.strip():
            self.history_store.append_query(thread_id, request.query.strip())

        debug = None
        if request.debug:
            debug = {
                "translated_text": result.get("translated_text"),
                "image_class": result.get("image_class"),
                "error": result.get("error"),
                "retrieved": [
                    {
                        "id": doc.id,
                        "title": doc.title,
                        "score": round(doc.score, 4),
                        "dense_score": round(doc.dense_score, 4),
                        "keyword_score": round(doc.keyword_score, 4),
                    }
                    for doc in result.get("retrieved_docs", [])
                ],
            }

        return ChatResponse(
            thread_id=thread_id,
            answer=result.get("answer", "服务器繁忙，请稍后重试。"),
            route=result.get("route", "unknown"),
            language=result.get("language") or result.get("image_language"),
            intent=result.get("intent"),
            matched_scenic_names=result.get("matched_scenic_names", []),
            retrieved_count=len(result.get("retrieved_docs", [])),
            debug=debug,
        )
