from __future__ import annotations

import argparse

from rag_chatflow.config import Settings
from rag_chatflow.dify_import import DifyKnowledgeClient, import_dataset, write_knowledge_json
from rag_chatflow.vectorstore import load_index_documents, rebuild_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a Dify knowledge base's existing chunks into data/knowledge.json"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Rebuild the local Qdrant collection immediately after importing",
    )
    parser.add_argument(
        "--no-scenic",
        action="store_true",
        help="With --reindex, do not include scenic introductions in Qdrant",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Also import disabled or archived Dify documents/chunks",
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="Override DIFY_DATASET_ID for this run",
    )
    args = parser.parse_args()

    settings = Settings()
    dataset_id = (args.dataset_id or settings.dify_dataset_id).strip()
    settings.ensure_dify_config(dataset_id=dataset_id)

    client = DifyKnowledgeClient(
        base_url=settings.dify_base_url,
        api_key=settings.dify_dataset_api_key,
        dataset_id=dataset_id,
        timeout_seconds=settings.dify_request_timeout_seconds,
    )
    documents, stats = import_dataset(
        client,
        include_disabled=args.include_disabled,
    )
    write_knowledge_json(settings.knowledge_path, documents)

    print(
        "Dify import complete: "
        f"documents={stats.documents_imported}/{stats.documents_seen}, "
        f"segments={stats.segments_imported}/{stats.segments_seen}, "
        f"output={settings.knowledge_path}"
    )

    if args.reindex:
        index_documents = load_index_documents(
            settings,
            include_scenic=not args.no_scenic,
        )
        count = rebuild_index(settings, index_documents)
        print(
            f"Indexed {count} documents into "
            f"{settings.qdrant_collection} at {settings.qdrant_path}"
        )


if __name__ == "__main__":
    main()
