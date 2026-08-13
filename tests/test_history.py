from rag_chatflow.history import JsonHistoryStore


def test_history_keeps_recent_queries(tmp_path):
    store = JsonHistoryStore(tmp_path / "history.json")
    for index in range(7):
        store.append_query("thread-1", f"q{index}")
    assert store.recent_queries("thread-1", limit=5) == ["q2", "q3", "q4", "q5", "q6"]
