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
        "--list-datasets",
        action="store_true",
        help="List accessible Dify knowledge bases and exit",
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
    parser.add_argument(
        "--auth-mode",
        choices=("auto", "cookie", "api_key"),
        default=None,
        help="Override DIFY_AUTH_MODE for this run",
    )
    args = parser.parse_args()

    settings = Settings()
    dataset_id = (args.dataset_id or settings.dify_dataset_id).strip()
    auth_mode = (args.auth_mode or settings.dify_auth_mode or "auto").strip().lower()
    settings.ensure_dify_config(
        dataset_id=dataset_id,
        require_dataset=not args.list_datasets,
        auth_mode=auth_mode,
    )

    client = DifyKnowledgeClient(
        base_url=settings.dify_base_url,
        dataset_id=dataset_id,
        cookie=settings.dify_cookie,
        api_key=settings.dify_dataset_api_key,
        csrf_token=settings.dify_csrf_token,
        auth_mode=auth_mode,
        timeout_seconds=settings.dify_request_timeout_seconds,
    )

    if args.list_datasets:
        rows = list(client.iter_datasets())
        if not rows:
            print("No accessible Dify datasets found.")
            return
        for dataset in rows:
            dataset_id_value = dataset.get("id", "")
            name = dataset.get("name", "")
            document_count = dataset.get("document_count")
            suffix = f" documents={document_count}" if document_count is not None else ""
            print(f"{dataset_id_value}\t{name}{suffix}")
        return

    documents, stats = import_dataset(
        client,
        include_disabled=args.include_disabled,
    )
    write_knowledge_json(settings.knowledge_path, documents)

    print(
        "Dify import complete: "
        f"auth={client.auth_mode}, "
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
