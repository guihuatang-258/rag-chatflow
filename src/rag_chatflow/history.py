from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Protocol


class HistoryStore(Protocol):
    def recent_queries(self, thread_id: str, limit: int = 5) -> list[str]: ...

    def append_query(self, thread_id: str, query: str, keep: int = 20) -> None: ...


class JsonHistoryStore:
    """Small single-process JSON history store.

    It mirrors Dify's queryHistory conversation variable without introducing Redis/PostgreSQL.
    Replace this class behind HistoryStore when multi-process deployment needs shared state.
    """

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()

    def recent_queries(self, thread_id: str, limit: int = 5) -> list[str]:
        with self._lock:
            data = self._read()
            return list(data.get(thread_id, []))[-limit:]

    def append_query(self, thread_id: str, query: str, keep: int = 20) -> None:
        query = query.strip()
        if not query:
            return
        with self._lock:
            data = self._read()
            history = list(data.get(thread_id, []))
            history.append(query)
            data[thread_id] = history[-keep:]
            self._atomic_write(data)

    def _read(self) -> dict[str, list[str]]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _atomic_write(self, data: dict[str, list[str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)
