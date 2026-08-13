from __future__ import annotations

import argparse

from rag_chatflow.config import Settings
from rag_chatflow.vectorstore import load_index_documents, rebuild_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the local Qdrant knowledge index")
    parser.add_argument(
        "--no-scenic",
        action="store_true",
        help="Do not include resources/scenic_introductions.json in the index",
    )
    args = parser.parse_args()

    settings = Settings()
    docs = load_index_documents(settings, include_scenic=not args.no_scenic)
    count = rebuild_index(settings, docs)
    print(f"Indexed {count} documents into {settings.qdrant_collection} at {settings.qdrant_path}")


if __name__ == "__main__":
    main()
