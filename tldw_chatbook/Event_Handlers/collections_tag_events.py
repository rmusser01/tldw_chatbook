# tldw_chatbook/Event_Handlers/collections_tag_events.py
"""
Event handlers for the Collections/Tag management functionality.
Handles keyword operations like rename, merge, delete, and statistics.
"""

from typing import TYPE_CHECKING, List, Dict, Any
from textual.message import Message
from loguru import logger

if TYPE_CHECKING:
    from ..app import TldwCli


class KeywordRenameEvent(Message):
    """Event for keyword rename operation."""
    
    def __init__(self, keyword_id: int, new_name: str) -> None:
        super().__init__()
        self.keyword_id = keyword_id
        self.new_name = new_name


class KeywordMergeEvent(Message):
    """Event for keyword merge operation."""
    
    def __init__(self, source_keyword_ids: List[int], target_keyword: str, create_if_not_exists: bool = True) -> None:
        super().__init__()
        self.source_keyword_ids = source_keyword_ids
        self.target_keyword = target_keyword
        self.create_if_not_exists = create_if_not_exists


class KeywordDeleteEvent(Message):
    """Event for keyword delete operation."""
    
    def __init__(self, keyword_ids: List[int]) -> None:
        super().__init__()
        self.keyword_ids = keyword_ids


async def handle_keyword_rename(app: 'TldwCli', event: KeywordRenameEvent) -> None:
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
        success = await app.run_in_thread(
            app.media_db.rename_keyword,
            event.keyword_id,
            event.new_name
        )
        
        if success:
            app.notify(f"Keyword renamed successfully to '{event.new_name}'", severity="information")
            
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
        logger.error(f"Error renaming keyword: {e}", exc_info=True)
        app.notify(f"Error renaming keyword: {str(e)[:100]}", severity="error")


async def handle_keyword_merge(app: 'TldwCli', event: KeywordMergeEvent) -> None:
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
        success = await app.run_in_thread(
            app.media_db.merge_keywords,
            event.source_keyword_ids,
            event.target_keyword,
            event.create_if_not_exists
        )
        
        if success:
            app.notify(
                f"Successfully merged {count} keyword{'s' if count > 1 else ''} into '{event.target_keyword}'",
                severity="information"
            )
            
            # Refresh the Collections/Tags window if it's active
            try:
                from ..Widgets.collections_tag_window import CollectionsTagWindow
                collections_window = app.query_one(CollectionsTagWindow)
                collections_window.load_keywords()
                collections_window.clear_selection()
            except Exception:
                pass  # Window might not be active
                
            # Also refresh any active media views to reflect the keyword changes
            try:
                from ..Event_Handlers.media_events import perform_media_search_and_display
                # If we're in a media view, refresh it
                if hasattr(app, 'current_media_type_filter_slug') and app.current_media_type_filter_slug:
                    await perform_media_search_and_display(app, app.current_media_type_filter_slug, "")
            except Exception:
                pass
        else:
            app.notify("Failed to merge keywords", severity="error")
            
    except ValueError as e:
        # Handle validation errors
        app.notify(str(e), severity="warning")
    except Exception as e:
        logger.error(f"Error merging keywords: {e}", exc_info=True)
        app.notify(f"Error merging keywords: {str(e)[:100]}", severity="error")


async def handle_keyword_delete(app: 'TldwCli', event: KeywordDeleteEvent) -> None:
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
            
        # Get keywords info for notification
        keywords_info = []
        for keyword_id in event.keyword_ids:
            try:
                # Get keyword name before deletion
                cursor = app.media_db.execute_query(
                    "SELECT keyword FROM Keywords WHERE id = ? AND deleted = 0",
                    (keyword_id,)
                )
                result = cursor.fetchone()
                if result:
                    keywords_info.append(result['keyword'])
            except Exception:
                pass
                
        # Perform the delete operations
        success_count = 0
        for keyword_id in event.keyword_ids:
            try:
                # Get keyword name for soft_delete_keyword method
                cursor = app.media_db.execute_query(
                    "SELECT keyword FROM Keywords WHERE id = ? AND deleted = 0",
                    (keyword_id,)
                )
                result = cursor.fetchone()
                if result:
                    success = await app.run_in_thread(
                        app.media_db.soft_delete_keyword,
                        result['keyword']
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
                severity="information"
            )
            
            # Refresh the Collections/Tags window if it's active
            try:
                from ..Widgets.collections_tag_window import CollectionsTagWindow
                collections_window = app.query_one(CollectionsTagWindow)
                collections_window.load_keywords()
                collections_window.clear_selection()
            except Exception:
                pass  # Window might not be active
                
            # Also refresh any active media views
            try:
                from ..Event_Handlers.media_events import perform_media_search_and_display
                if hasattr(app, 'current_media_type_filter_slug') and app.current_media_type_filter_slug:
                    await perform_media_search_and_display(app, app.current_media_type_filter_slug, "")
            except Exception:
                pass
                
        if success_count < count:
            app.notify(
                f"Some keywords could not be deleted ({count - success_count} failed)",
                severity="warning"
            )
            
    except Exception as e:
        logger.error(f"Error deleting keywords: {e}", exc_info=True)
        app.notify(f"Error deleting keywords: {str(e)[:100]}", severity="error")


async def load_keyword_statistics(app: 'TldwCli') -> List[Dict[str, Any]]:
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
        stats = await app.run_in_thread(app.media_db.get_keyword_usage_stats)
        logger.debug(f"Loaded statistics for {len(stats)} keywords")
        return stats
        
    except Exception as e:
        logger.error(f"Error loading keyword statistics: {e}", exc_info=True)
        return []