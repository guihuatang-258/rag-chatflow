from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from langgraph.graph import END, START, StateGraph

from .llm import BUSY_ZH, IDENTITY_ZH, SENSITIVE_ZH, LANGUAGE_BY_CODE, ModelGateway
from .models import ChatState, Intent, RetrievedDocument
from .resources import ResourceStore


class Retriever(Protocol):
    async def search(self, query: str) -> list[RetrievedDocument]: ...


@dataclass(frozen=True)
class WorkflowServices:
    models: ModelGateway
    retriever: Retriever
    resources: ResourceStore


def build_workflow(services: WorkflowServices):
    async def analyze_query(state: ChatState) -> dict:
        try:
            result = await services.models.analyze_query(state.get("query", ""), state.get("history", []))
            return {
                "language": result.language,
                "translated_text": result.translated_text,
                "intent": result.intent,
                "route": "text_analyzed",
            }
        except Exception as exc:
            return {"answer": SENSITIVE_ZH, "route": "text_error", "error": str(exc)}

    def route_after_analysis(state: ChatState) -> str:
        if state.get("answer"):
            return "done"
        intent = state.get("intent", Intent.CHAT)
        return {
            Intent.SCENIC: "retrieve",
            Intent.SENSITIVE: "sensitive",
            Intent.IDENTITY: "identity",
            Intent.CHAT: "general",
        }[intent]

    async def retrieve(state: ChatState) -> dict:
        try:
            docs = await services.retriever.search(state.get("translated_text", ""))
            return {"retrieved_docs": docs, "route": "retrieved"}
        except Exception as exc:
            return {"retrieved_docs": [], "route": "retrieve_error", "error": str(exc)}

    def route_after_retrieve(state: ChatState) -> str:
        return "rag_answer" if state.get("retrieved_docs") else "general"

    async def rag_answer(state: ChatState) -> dict:
        try:
            answer = await services.models.answer_with_context(
                state.get("translated_text", state.get("query", "")),
                state.get("language", "中文"),
                state.get("retrieved_docs", []),
            )
            return {"answer": answer, "route": "text_rag"}
        except Exception as exc:
            return {"answer": BUSY_ZH, "route": "rag_error", "error": str(exc)}

    async def general_answer(state: ChatState) -> dict:
        try:
            answer = await services.models.general_answer(
                state.get("translated_text", state.get("query", "")),
                state.get("language", "中文"),
            )
            return {"answer": answer, "route": "text_general"}
        except Exception as exc:
            return {"answer": BUSY_ZH, "route": "general_error", "error": str(exc)}

    async def identity_answer(state: ChatState) -> dict:
        language = state.get("language", "中文")
        try:
            answer = await services.models.translate(IDENTITY_ZH, language)
            return {"answer": answer, "route": "text_identity"}
        except Exception as exc:
            return {"answer": IDENTITY_ZH, "route": "identity_error", "error": str(exc)}

    async def sensitive_answer(state: ChatState) -> dict:
        language = state.get("language", "中文")
        try:
            answer = await services.models.translate(SENSITIVE_ZH, language)
            return {"answer": answer, "route": "text_sensitive"}
        except Exception as exc:
            return {"answer": SENSITIVE_ZH, "route": "sensitive_error", "error": str(exc)}

    async def classify_image(state: ChatState) -> dict:
        language = LANGUAGE_BY_CODE.get(state.get("requested_language", 1), "中文（普通话）")
        try:
            result = await services.models.classify_image(state["image_urls"][0])
            return {
                "image_class": result.image_class,
                "image_language": language,
                "route": "image_classified",
            }
        except Exception as exc:
            return {"answer": BUSY_ZH, "image_language": language, "route": "image_classify_error", "error": str(exc)}

    def route_after_image_classification(state: ChatState) -> str:
        if state.get("answer"):
            return "done"
        matchable = state.get("image_class") in {"山", "古建筑"}
        has_location = bool((state.get("scenic_name") or "").strip())
        return "image_match" if matchable and has_location else "image_recognize"

    async def image_match(state: ChatState) -> dict:
        scenic_name = (state.get("scenic_name") or "").strip()
        refs = services.resources.reference_images(state.get("image_class", ""), scenic_name)
        if not refs:
            return {"matched_scenic_names": [], "route": "image_no_reference"}
        try:
            matched = await services.models.match_scenic(
                state["image_urls"][0], state.get("image_class", ""), scenic_name, refs
            )
            return {"matched_scenic_names": matched, "route": "image_matched"}
        except Exception as exc:
            return {"answer": BUSY_ZH, "route": "image_match_error", "error": str(exc)}

    def route_after_image_match(state: ChatState) -> str:
        if state.get("answer"):
            return "done"
        return "image_scenic_answer" if state.get("matched_scenic_names") else "image_recognize"

    async def image_scenic_answer(state: ChatState) -> dict:
        context = services.resources.introductions_for(state.get("matched_scenic_names", []))
        if not context:
            return {"route": "image_missing_intro"}
        try:
            answer = await services.models.polish_image_result(
                raw_result=context,
                language=state.get("image_language", "中文（普通话）"),
                scenic_context=context,
            )
            return {"answer": answer, "route": "image_reference_answer"}
        except Exception as exc:
            return {"answer": BUSY_ZH, "route": "image_polish_error", "error": str(exc)}

    async def image_recognize(state: ChatState) -> dict:
        try:
            raw = await services.models.recognize_image(
                state["image_urls"][0], state.get("image_language", "中文（普通话）")
            )
            return {"image_raw_result": raw, "route": "image_recognized"}
        except Exception as exc:
            return {"answer": BUSY_ZH, "route": "image_recognize_error", "error": str(exc)}

    def route_after_image_recognize(state: ChatState) -> str:
        return "done" if state.get("answer") else "image_polish"

    async def image_polish(state: ChatState) -> dict:
        try:
            answer = await services.models.polish_image_result(
                state.get("image_raw_result", ""), state.get("image_language", "中文（普通话）")
            )
            return {"answer": answer, "route": "image_fallback_answer"}
        except Exception as exc:
            return {"answer": BUSY_ZH, "route": "image_polish_error", "error": str(exc)}

    def route_input(state: ChatState) -> Literal["analyze_query", "classify_image"]:
        return "classify_image" if state.get("image_urls") else "analyze_query"

    graph = StateGraph(ChatState)
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rag_answer", rag_answer)
    graph.add_node("general", general_answer)
    graph.add_node("identity", identity_answer)
    graph.add_node("sensitive", sensitive_answer)
    graph.add_node("classify_image", classify_image)
    graph.add_node("image_match", image_match)
    graph.add_node("image_scenic_answer", image_scenic_answer)
    graph.add_node("image_recognize", image_recognize)
    graph.add_node("image_polish", image_polish)

    graph.add_conditional_edges(START, route_input)
    graph.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "retrieve": "retrieve",
            "sensitive": "sensitive",
            "identity": "identity",
            "general": "general",
            "done": END,
        },
    )
    graph.add_conditional_edges(
        "retrieve", route_after_retrieve, {"rag_answer": "rag_answer", "general": "general"}
    )
    graph.add_edge("rag_answer", END)
    graph.add_edge("general", END)
    graph.add_edge("identity", END)
    graph.add_edge("sensitive", END)

    graph.add_conditional_edges(
        "classify_image",
        route_after_image_classification,
        {"image_match": "image_match", "image_recognize": "image_recognize", "done": END},
    )
    graph.add_conditional_edges(
        "image_match",
        route_after_image_match,
        {
            "image_scenic_answer": "image_scenic_answer",
            "image_recognize": "image_recognize",
            "done": END,
        },
    )
    graph.add_conditional_edges(
        "image_scenic_answer",
        lambda state: "image_recognize" if not state.get("answer") else "done",
        {"image_recognize": "image_recognize", "done": END},
    )
    graph.add_conditional_edges(
        "image_recognize",
        route_after_image_recognize,
        {"image_polish": "image_polish", "done": END},
    )
    graph.add_edge("image_polish", END)

    return graph.compile()
