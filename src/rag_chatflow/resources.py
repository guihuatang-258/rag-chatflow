from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResourceStore:
    translations: dict[str, str]
    introductions: dict[str, str]
    scenic_images: dict[str, dict[str, dict[str, Any]]]

    @classmethod
    def load(cls, root: Path) -> "ResourceStore":
        return cls(
            translations=_read_json(root / "scenery_translations.json"),
            introductions=_read_json(root / "scenic_introductions.json"),
            scenic_images=_read_json(root / "scenic_images.json"),
        )

    def translation_prompt(self) -> str:
        return "\n".join(f"{cn} - {en}" for cn, en in self.translations.items())

    def reference_images(self, image_class: str, scenic_name: str) -> dict[str, Any] | None:
        return self.scenic_images.get(image_class, {}).get(scenic_name)

    def introductions_for(self, scenic_names: list[str]) -> str:
        blocks = []
        for name in scenic_names:
            text = self.introductions.get(name)
            if text:
                blocks.append(f"{name}：{text}")
        return "\n\n".join(blocks)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
