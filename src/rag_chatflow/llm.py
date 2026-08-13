from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, Protocol

from openai import AsyncOpenAI, BadRequestError

from .config import Settings
from .models import ImageClassification, Intent, QueryAnalysis, RetrievedDocument
from .prompts import (
    ANALYZE_QUERY_SYSTEM,
    GENERAL_ANSWER_SYSTEM,
    IMAGE_CLASSIFY_PROMPT,
    IMAGE_MATCH_PROMPT,
    IMAGE_POLISH_SYSTEM,
    IMAGE_RECOGNIZE_PROMPT,
    RAG_ANSWER_SYSTEM,
)
from .resources import ResourceStore

LANGUAGE_BY_CODE = {
    1: "中文（普通话）",
    2: "英文",
    3: "日文",
    4: "韩文",
    5: "俄文",
    6: "阿拉伯语",
    7: "法语",
    8: "西班牙语",
    9: "葡萄牙语",
    10: "德语",
    11: "越南语",
    12: "希伯来语",
    13: "泰语",
    14: "马来语",
}

IDENTITY_ZH = "您好，我是张家界黄石寨景区的智能导览。我可以回答景区门票、游览路线、景点介绍等相关问题。"
SENSITIVE_ZH = "抱歉，这个问题我没法回答。"
BUSY_ZH = "服务器繁忙，请稍后重试。"


class ModelGateway(Protocol):
    async def analyze_query(self, query: str, history: list[str]) -> QueryAnalysis: ...

    async def answer_with_context(
        self, query: str, language: str, docs: list[RetrievedDocument]
    ) -> str: ...

    async def general_answer(self, query: str, language: str) -> str: ...

    async def translate(self, text: str, language: str) -> str: ...

    async def classify_image(self, image_url: str) -> ImageClassification: ...

    async def match_scenic(
        self,
        image_url: str,
        image_class: str,
        scenic_name: str,
        refs: dict[str, Any],
    ) -> list[str]: ...

    async def recognize_image(self, image_url: str, language: str) -> str: ...

    async def polish_image_result(
        self, raw_result: str, language: str, scenic_context: str = ""
    ) -> str: ...


class OpenAICompatibleGateway:
    def __init__(self, settings: Settings, resources: ResourceStore):
        self.settings = settings
        self.resources = resources
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key or "not-configured",
            base_url=settings.openai_base_url,
        )

    async def analyze_query(self, query: str, history: list[str]) -> QueryAnalysis:
        self.settings.ensure_api_key()
        history_text = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(history)) or "无"
        user = (
            f"官方景点名称参考：\n{self.resources.translation_prompt()}\n\n"
            f"用户当前问题：\n{query}\n\n历史问题：\n{history_text}"
        )
        content = await self._json_chat(
            model=self.settings.fast_model,
            system=ANALYZE_QUERY_SYSTEM,
            user=user,
            temperature=0.3,
        )
        data = _extract_json(content)
        raw_intent = str(data.get("intent", "chat")).lower().strip()
        intent_aliases = {
            "武陵源景区和民俗文化": Intent.SCENIC,
            "闲聊": Intent.CHAT,
            "敏感问题": Intent.SENSITIVE,
            "身份问题": Intent.IDENTITY,
            "大模型厂商问题": Intent.IDENTITY,
        }
        if raw_intent in intent_aliases:
            intent = intent_aliases[raw_intent]
        else:
            intent = Intent(raw_intent)
        return QueryAnalysis(
            language=str(data.get("language") or "中文"),
            translated_text=str(data.get("translated_text") or query).strip(),
            intent=intent,
        )

    async def answer_with_context(
        self, query: str, language: str, docs: list[RetrievedDocument]
    ) -> str:
        context = "\n\n".join(
            f"【{doc.title or doc.id}】\n{doc.text}" for doc in docs
        )
        system = (
            f"{RAG_ANSWER_SYSTEM}\n\n指定回复语言：{language}\n\n"
            f"官方景点名称：\n{self.resources.translation_prompt()}"
        )
        user = f"用户问题：{query}\n\n检索内容：\n{context}"
        return await self._text_chat(
            self.settings.chat_model, system, user, 0.3, enable_search=True
        )

    async def general_answer(self, query: str, language: str) -> str:
        system = (
            f"{GENERAL_ANSWER_SYSTEM}\n\n指定回复语言：{language}\n\n"
            f"官方景点名称：\n{self.resources.translation_prompt()}"
        )
        return await self._text_chat(
            self.settings.chat_model, system, query, 0.7, enable_search=True
        )

    async def translate(self, text: str, language: str) -> str:
        if "中文" in language:
            return text
        system = f"你是张家界黄石寨景区的翻译。把用户文本准确翻译成{language}，只输出译文。"
        return await self._text_chat(self.settings.fast_model, system, text, 0.2)

    async def classify_image(self, image_url: str) -> ImageClassification:
        content = await self._vision_json([image_url], IMAGE_CLASSIFY_PROMPT)
        data = _extract_json(content)
        image_class = str(data.get("class", "其他")).strip()
        if image_class not in {"山", "古建筑", "动物", "植物", "其他"}:
            image_class = "其他"
        return ImageClassification(image_class=image_class)

    async def match_scenic(
        self,
        image_url: str,
        image_class: str,
        scenic_name: str,
        refs: dict[str, Any],
    ) -> list[str]:
        ref_items = refs.get("urls", [])
        reference_urls: list[str] = []
        composite_names: list[str] = []
        if refs.get("type") == "single":
            reference_urls = [str(url) for url in ref_items]
        elif refs.get("type") == "composite":
            for item in ref_items:
                if isinstance(item, Sequence) and len(item) >= 2:
                    reference_urls.append(str(item[0]))
                    composite_names.append(str(item[1]))

        if not reference_urls:
            return []

        content_parts: list[dict[str, Any]] = [
            {"type": "text", "text": "图片id：1"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        for index, url in enumerate(reference_urls, start=2):
            content_parts.extend(
                [
                    {"type": "text", "text": f"图片id：{index}"},
                    {"type": "image_url", "image_url": {"url": url}},
                ]
            )
        content_parts.append(
            {
                "type": "text",
                "text": f"图片类别：{image_class}\n景点候选：{scenic_name}\n{IMAGE_MATCH_PROMPT}",
            }
        )
        content = await self._vision_content(content_parts, json_mode=True)
        data = _extract_json(content)
        raw_ids = data.get("id", []) or []
        ids = []
        for value in raw_ids:
            try:
                ids.append(int(value))
            except (TypeError, ValueError):
                continue
        if not ids:
            return []
        if refs.get("type") == "single":
            return [scenic_name]
        result = []
        for image_id in ids:
            offset = image_id - 2
            if 0 <= offset < len(composite_names):
                result.append(composite_names[offset])
        return list(dict.fromkeys(result))

    async def recognize_image(self, image_url: str, language: str) -> str:
        prompt = f"{IMAGE_RECOGNIZE_PROMPT}\n指定回复语言：{language}"
        return await self._vision_text([image_url], prompt)

    async def polish_image_result(
        self, raw_result: str, language: str, scenic_context: str = ""
    ) -> str:
        system = f"{IMAGE_POLISH_SYSTEM}\n\n指定回复语言：{language}"
        if scenic_context:
            system += f"\n\n官方景点名称：\n{self.resources.translation_prompt()}"
        user = f"图像识别结果：\n{scenic_context or raw_result}"
        return await self._text_chat(self.settings.fast_model, system, user, 0.4)

    async def _text_chat(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float,
        enable_search: bool = False,
    ) -> str:
        self.settings.ensure_api_key()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        extra_body = self._provider_extra(enable_search=enable_search)
        if extra_body:
            kwargs["extra_body"] = extra_body
        response = await self.client.chat.completions.create(**kwargs)
        return _message_text(response.choices[0].message.content)

    async def _json_chat(
        self, model: str, system: str, user: str, temperature: float
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        extra_body = self._provider_extra()
        if extra_body:
            kwargs["extra_body"] = extra_body
        return await self._create_with_optional_json_mode(kwargs)

    async def _vision_json(self, image_urls: list[str], prompt: str) -> str:
        parts: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": url}} for url in image_urls
        ]
        parts.append({"type": "text", "text": prompt})
        return await self._vision_content(parts, json_mode=True)

    async def _vision_text(self, image_urls: list[str], prompt: str) -> str:
        parts: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": url}} for url in image_urls
        ]
        parts.append({"type": "text", "text": prompt})
        return await self._vision_content(parts, json_mode=False)

    async def _vision_content(self, parts: list[dict[str, Any]], json_mode: bool) -> str:
        self.settings.ensure_api_key()
        kwargs: dict[str, Any] = {
            "model": self.settings.vision_model,
            "messages": [{"role": "user", "content": parts}],
        }
        extra_body = self._provider_extra()
        if extra_body:
            kwargs["extra_body"] = extra_body
        if json_mode:
            return await self._create_with_optional_json_mode(kwargs)
        response = await self.client.chat.completions.create(**kwargs)
        return _message_text(response.choices[0].message.content)

    async def _create_with_optional_json_mode(self, kwargs: dict[str, Any]) -> str:
        self.settings.ensure_api_key()
        try:
            response = await self.client.chat.completions.create(
                **kwargs, response_format={"type": "json_object"}
            )
        except BadRequestError:
            response = await self.client.chat.completions.create(**kwargs)
        return _message_text(response.choices[0].message.content)

    def _provider_extra(self, enable_search: bool = False) -> dict[str, Any]:
        base_url = self.settings.openai_base_url.lower()
        is_dashscope = "dashscope" in base_url or ".maas.aliyuncs.com" in base_url
        if not is_dashscope:
            return {}
        extra: dict[str, Any] = {"enable_thinking": False}
        if enable_search and self.settings.enable_web_search:
            extra["enable_search"] = True
        return extra


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return str(content).strip()


def _extract_json(text: str) -> dict[str, Any]:
    text = text.replace("\u00a0", " ").strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    else:
        left = text.find("{")
        right = text.rfind("}")
        if left >= 0 and right > left:
            text = text[left : right + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("模型结构化输出不是 JSON object")
    return data
