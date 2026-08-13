from __future__ import annotations

from rag_chatflow.config import Settings
from rag_chatflow.vectorstore import load_index_documents, rebuild_index


def main() -> None:
    settings = Settings()
    docs = load_index_documents(settings)
    count = rebuild_index(settings, docs)
    print(f"Indexed {count} documents into {settings.qdrant_collection} at {settings.qdrant_path}")


if __name__ == "__main__":
    main()
