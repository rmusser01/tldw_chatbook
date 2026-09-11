# backfill.py
# Description: CLI entry point for bulk RAG backfill (task-247).
"""
Bulk-index pre-existing media/notes/conversations into the RAG vector store.

Usage:
    python -m tldw_chatbook.RAG_Search.backfill [--types media,note,conversation]
                                                [--page-size N] [--batch-size N]

Requires the `embeddings_rag` optional dependencies. Incremental: items whose
last_modified is unchanged since the last successful index are skipped, so
the command can be re-run safely at any time.
"""

import argparse
import asyncio
import json
import sys

from .ingestion_indexing import (
    ITEM_TYPE_CONVERSATION,
    ITEM_TYPE_MEDIA,
    ITEM_TYPE_NOTE,
    backfill_semantic_index,
    semantic_indexing_available,
)

_VALID_TYPES = (ITEM_TYPE_MEDIA, ITEM_TYPE_NOTE, ITEM_TYPE_CONVERSATION)


def main(argv=None) -> int:
    """Run the backfill against the standard user databases and print a summary."""
    parser = argparse.ArgumentParser(
        prog="python -m tldw_chatbook.RAG_Search.backfill",
        description="Index pre-existing media/notes/conversations into the RAG vector store.",
    )
    parser.add_argument(
        "--types",
        default=",".join(_VALID_TYPES),
        help=f"Comma-separated item types to backfill (default: all of {','.join(_VALID_TYPES)})",
    )
    parser.add_argument(
        "--page-size", type=int, default=100, help="Source-DB pagination size"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Documents per indexing batch"
    )
    review_action = parser.add_mutually_exclusive_group()
    review_action.add_argument(
        "--review-recovery",
        action="store_true",
        help="Show current local RAG recovery review without indexing",
    )
    review_action.add_argument(
        "--approve-recovery",
        metavar="FINGERPRINT",
        help="Approve only the unchanged displayed RAG recovery review; no indexing",
    )
    review_action.add_argument(
        "--reconcile-recovery",
        action="store_true",
        help="Explicitly rebuild and verify the current local projection after owner review",
    )
    args = parser.parse_args(argv)

    if args.review_recovery or args.approve_recovery:
        from .activation import approve_recovery_review, preview_recovery_review
        from .simplified.active_config import resolve_active_rag_config

        try:
            config = resolve_active_rag_config()
            review = (
                approve_recovery_review(config, args.approve_recovery)
                if args.approve_recovery
                else preview_recovery_review(config)
            )
        except (OSError, ValueError, RuntimeError, PermissionError):
            print(
                "RAG recovery review unavailable or changed; request a fresh review.",
                file=sys.stderr,
            )
            return 1
        print(json.dumps(review.to_dict(), indent=2))
        return 0

    item_types = tuple(t.strip() for t in args.types.split(",") if t.strip())
    invalid = [t for t in item_types if t not in _VALID_TYPES]
    if invalid:
        print(
            f"Unknown item type(s): {', '.join(invalid)}. Valid: {', '.join(_VALID_TYPES)}",
            file=sys.stderr,
        )
        return 2

    if not semantic_indexing_available():
        print(
            "Semantic indexing is unavailable: install the embeddings extras "
            "(pip install tldw_chatbook[embeddings_rag]) and ensure "
            "[AppRAGSearchConfig.rag.indexing].enabled is not false.",
            file=sys.stderr,
        )
        return 1

    from ..config import get_chachanotes_db_path, get_media_db_path
    from ..DB.ChaChaNotes_DB import CharactersRAGDB
    from ..DB.Client_Media_DB_v2 import MediaDatabase

    media_db = None
    chachanotes_db = None
    if ITEM_TYPE_MEDIA in item_types:
        media_db = MediaDatabase(get_media_db_path(), client_id="rag_backfill_cli")
    if ITEM_TYPE_NOTE in item_types or ITEM_TYPE_CONVERSATION in item_types:
        chachanotes_db = CharactersRAGDB(get_chachanotes_db_path(), "rag_backfill_cli")

    def _progress(update):
        print(
            f"[{update['item_type']}] indexed={update['indexed']} "
            f"skipped={update['skipped']} failed={update['failed']}",
            file=sys.stderr,
        )

    summary = asyncio.run(
        backfill_semantic_index(
            media_db=media_db,
            chachanotes_db=chachanotes_db,
            item_types=item_types,
            page_size=args.page_size,
            batch_size=args.batch_size,
            progress_callback=_progress,
            reconcile_for_recovery=args.reconcile_recovery,
        )
    )

    from dataclasses import asdict, is_dataclass

    print(
        json.dumps(
            summary,
            indent=2,
            default=lambda value: asdict(value) if is_dataclass(value) else str(value),
        )
    )
    return 0 if summary["status"] == "ok" and summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
