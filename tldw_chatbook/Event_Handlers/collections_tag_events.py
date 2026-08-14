# tldw_chatbook/Event_Handlers/collections_tag_events.py
"""
Event handlers for the Collections/Tag management functionality.
Handles keyword operations like rename, merge, delete, and statistics.
"""

import asyncio
from typing import TYPE_CHECKING, Callable, List, Dict, Any
from textual.message import Message
from loguru import logger

if TYPE_CHECKING:
    from ..app import TldwCli


async def _media_db_off_loop(app: "TldwCli", func: Callable, /, *args: Any) -> Any:
    """Run one sync media-DB call off the event loop.

    task-15471: this file's handlers called ``app.run_in_thread``, which
    does not exist -- neither Textual 8.x's ``App`` nor ``TldwCli`` defines
    it -- so every rename/merge/delete raised ``AttributeError`` into its
    own error toast. Replaced with ``asyncio.to_thread`` (the DB uses
    thread-local connections, so a pool thread opens its own), guarded the
    same way as the Console browser-search threading: a per-connection
    ``:memory:`` DB is only visible to the thread that migrated it and must
    stay on the loop thread.
    """
    if bool(getattr(app.media_db, "is_memory_db", False)):
        return func(*args)
    return await asyncio.to_thread(func, *args)


class KeywordRenameEvent(Message):
    """Event for keyword rename operation."""

    def __init__(self, keyword_id: int, new_name: str) -> None:
        super().__init__()
        self.keyword_id = keyword_id
        self.new_name = new_name


class KeywordMergeEvent(Message):
    """Event for keyword merge operation."""

    def __init__(
        self,
        source_keyword_ids: List[int],
        target_keyword: str,
        create_if_not_exists: bool = True,
    ) -> None:
        super().__init__()
        self.source_keyword_ids = source_keyword_ids
        self.target_keyword = target_keyword
        self.create_if_not_exists = create_if_not_exists


class KeywordDeleteEvent(Message):
    """Event for keyword delete operation."""

    def __init__(self, keyword_ids: List[int]) -> None:
        super().__init__()
        self.keyword_ids = keyword_ids


async def handle_keyword_rename(app: "TldwCli", event: KeywordRenameEvent) -> None:
    """
    Handle keyword rename operation.

    Args:
        app: The application instance
        event: The rename event containing keyword_id and new_name
    """
    logger.info(f"Renaming keyword ID {event.keyword_id} to '{event.new_name}'")

    try:
        if not app.media_db:
            raise RuntimeError("Media DB service not available")

        # Perform the rename operation
        success = await _media_db_off_loop(
            app, app.media_db.rename_keyword, event.keyword_id, event.new_name
        )

        if success:
            app.notify(
                f"Keyword renamed successfully to '{event.new_name}'",
                severity="information",
            )

            # Refresh the Collections/Tags window if it's active
            try:
                from ..Widgets.collections_tag_window import CollectionsTagWindow

                collections_window = app.query_one(CollectionsTagWindow)
                collections_window.load_keywords()
                collections_window.clear_selection()
            except Exception:
                pass  # Window might not be active
        else:
            app.notify("Failed to rename keyword", severity="error")

    except ValueError as e:
        # Handle validation errors (empty name, duplicate, etc.)
        app.notify(str(e), severity="warning")
    except Exception as e:
        logger.opt(exception=True).error(f"Error renaming keyword: {e}")
        app.notify(f"Error renaming keyword: {str(e)[:100]}", severity="error")


async def handle_keyword_merge(app: "TldwCli", event: KeywordMergeEvent) -> None:
    """
    Handle keyword merge operation.

    Args:
        app: The application instance
        event: The merge event containing source_keyword_ids and target_keyword
    """
    count = len(event.source_keyword_ids)
    logger.info(f"Merging {count} keywords into '{event.target_keyword}'")

    try:
        if not app.media_db:
            raise RuntimeError("Media DB service not available")

        # Perform the merge operation
        success = await _media_db_off_loop(
            app,
            app.media_db.merge_keywords,
            event.source_keyword_ids,
            event.target_keyword,
            event.create_if_not_exists,
        )

        if success:
            app.notify(
                f"Successfully merged {count} keyword{'s' if count > 1 else ''} into '{event.target_keyword}'",
                severity="information",
            )

            # Refresh the Collections/Tags window if it's active
            try:
                from ..Widgets.collections_tag_window import CollectionsTagWindow

                collections_window = app.query_one(CollectionsTagWindow)
                collections_window.load_keywords()
                collections_window.clear_selection()
            except Exception:
                pass  # Window might not be active

        else:
            app.notify("Failed to merge keywords", severity="error")

    except ValueError as e:
        # Handle validation errors
        app.notify(str(e), severity="warning")
    except Exception as e:
        logger.opt(exception=True).error(f"Error merging keywords: {e}")
        app.notify(f"Error merging keywords: {str(e)[:100]}", severity="error")


async def handle_keyword_delete(app: "TldwCli", event: KeywordDeleteEvent) -> None:
    """
    Handle keyword delete operation.

    Args:
        app: The application instance
        event: The delete event containing keyword_ids
    """
    count = len(event.keyword_ids)
    logger.info(f"Deleting {count} keyword(s)")

    try:
        if not app.media_db:
            raise RuntimeError("Media DB service not available")

        # Resolve keyword names ONCE, off the loop -- this lookup used to run
        # twice per keyword (once for the notification, once again before the
        # delete), synchronously on the event loop (task-15471). The single
        # batch serves both the notification and the delete below.
        def _fetch_keyword_names() -> Dict[int, str]:
            names: Dict[int, str] = {}
            for keyword_id in event.keyword_ids:
                try:
                    cursor = app.media_db.execute_query(
                        "SELECT keyword FROM Keywords WHERE id = ? AND deleted = 0",
                        (keyword_id,),
                    )
                    result = cursor.fetchone()
                    if result:
                        names[keyword_id] = result["keyword"]
                except Exception:
                    pass
            return names

        keyword_names = await _media_db_off_loop(app, _fetch_keyword_names)
        keywords_info = list(keyword_names.values())

        # Perform the delete operations
        success_count = 0
        for keyword_id in event.keyword_ids:
            keyword = keyword_names.get(keyword_id)
            if keyword is None:
                continue
            try:
                success = await _media_db_off_loop(
                    app, app.media_db.soft_delete_keyword, keyword
                )
                if success:
                    success_count += 1
            except Exception as e:
                logger.error(f"Error deleting keyword ID {keyword_id}: {e}")

        if success_count > 0:
            keyword_names = ", ".join(keywords_info[:3])
            if len(keywords_info) > 3:
                keyword_names += f" and {len(keywords_info) - 3} more"

            app.notify(
                f"Successfully deleted {success_count} keyword{'s' if success_count > 1 else ''}: {keyword_names}",
                severity="information",
            )

            # Refresh the Collections/Tags window if it's active
            try:
                from ..Widgets.collections_tag_window import CollectionsTagWindow

                collections_window = app.query_one(CollectionsTagWindow)
                collections_window.load_keywords()
                collections_window.clear_selection()
            except Exception:
                pass  # Window might not be active

        if success_count < count:
            app.notify(
                f"Some keywords could not be deleted ({count - success_count} failed)",
                severity="warning",
            )

    except Exception as e:
        logger.opt(exception=True).error(f"Error deleting keywords: {e}")
        app.notify(f"Error deleting keywords: {str(e)[:100]}", severity="error")


async def load_keyword_statistics(app: "TldwCli") -> List[Dict[str, Any]]:
    """
    Load keyword usage statistics from the database.

    Args:
        app: The application instance

    Returns:
        List of keyword dictionaries with usage statistics
    """
    try:
        if not app.media_db:
            logger.error("Media DB not available")
            return []

        # Get keyword statistics
        stats = await _media_db_off_loop(app, app.media_db.get_keyword_usage_stats)
        logger.debug(f"Loaded statistics for {len(stats)} keywords")
        return stats

    except Exception as e:
        logger.opt(exception=True).error(f"Error loading keyword statistics: {e}")
        return []
