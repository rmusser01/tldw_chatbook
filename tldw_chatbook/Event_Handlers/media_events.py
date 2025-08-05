# tldw_chatbook/Event_Handlers/media_events.py
#
#
# Imports
import math
from datetime import datetime
from typing import TYPE_CHECKING, Dict, Any
from loguru import logger
#
# 3rd-party Libraries
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import ListView, Input, TextArea, Label, ListItem, Button, Markdown, Static  # Added ListItem
from textual.css.query import QueryError
from textual.message import Message
from textual.screen import ModalScreen
from rich.text import Text  # For formatting details
#
# Local Imports
from ..DB.Client_Media_DB_v2 import fetch_keywords_for_media
#
if TYPE_CHECKING:
    from ..app import TldwCli
########################################################################################################################
#
# Statics:
RESULTS_PER_PAGE = 20
#
# Event Classes:

class MediaTypeSelectedEvent(Message):
    """Event fired when a media type is selected in the navigation panel."""
    
    def __init__(self, type_slug: str, display_name: str) -> None:
        super().__init__()
        self.type_slug = type_slug
        self.display_name = display_name


class MediaMetadataUpdateEvent(Message):
    """Event for updating media metadata."""
    
    def __init__(self, media_id: int, title: str, media_type: str, author: str, 
                 url: str, keywords: list, type_slug: str) -> None:
        super().__init__()
        self.media_id = media_id
        self.title = title
        self.media_type = media_type
        self.author = author
        self.url = url
        self.keywords = keywords
        self.type_slug = type_slug


class MediaDeleteConfirmationEvent(Message):
    """Event to trigger deletion confirmation dialog."""
    
    def __init__(self, media_id: int, media_title: str, type_slug: str) -> None:
        super().__init__()
        self.media_id = media_id
        self.media_title = media_title
        self.type_slug = type_slug


class MediaUndeleteEvent(Message):
    """Event to trigger media undeletion."""
    
    def __init__(self, media_id: int, type_slug: str) -> None:
        super().__init__()
        self.media_id = media_id
        self.type_slug = type_slug


class MediaListCollapseEvent(Message):
    """Event fired when media list should be collapsed/expanded."""
    pass


class SidebarCollapseEvent(Message):
    """Event fired when sidebar should be collapsed/expanded."""
    pass


class MediaAnalysisRequestEvent(Message):
    """Event fired when requesting LLM analysis of media content."""
    
    def __init__(self, media_id: int, provider: str, model: str, 
                 system_prompt: str, user_prompt: str, type_slug: str,
                 temperature: float = 0.7, top_p: float = 0.95, 
                 min_p: float = 0.05, max_tokens: int = 4096) -> None:
        super().__init__()
        self.media_id = media_id
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.type_slug = type_slug
        self.temperature = temperature
        self.top_p = top_p
        self.min_p = min_p
        self.max_tokens = max_tokens


class MediaAnalysisSaveEvent(Message):
    """Event fired when saving new analysis content."""
    
    def __init__(self, media_id: int, analysis_content: str, type_slug: str) -> None:
        super().__init__()
        self.media_id = media_id
        self.analysis_content = analysis_content
        self.type_slug = type_slug


class MediaAnalysisSaveAsNoteEvent(Message):
    """Event fired when saving analysis as a new note."""
    
    def __init__(self, media_id: int, media_title: str, analysis_content: str) -> None:
        super().__init__()
        self.media_id = media_id
        self.media_title = media_title
        self.analysis_content = analysis_content


class MediaAnalysisOverwriteEvent(Message):
    """Event fired when overwriting existing analysis content."""
    
    def __init__(self, media_id: int, analysis_content: str, type_slug: str) -> None:
        super().__init__()
        self.media_id = media_id
        self.analysis_content = analysis_content
        self.type_slug = type_slug


class MediaAnalysisDeleteEvent(Message):
    """Event fired when deleting an analysis version."""
    
    def __init__(self, media_id: int, version_uuid: str, type_slug: str) -> None:
        super().__init__()
        self.media_id = media_id
        self.version_uuid = version_uuid
        self.type_slug = type_slug

#
# Functions:

async def handle_media_nav_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles media navigation button presses in the Media tab."""
    logger = app.loguru_logger
    button_id = event.button.id
    try:
        type_slug = button_id.replace("media-nav-", "")
        view_to_activate = f"media-view-{type_slug}"
        logger.debug(f"Media nav button '{button_id}' pressed. Activating view '{view_to_activate}', type filter: '{type_slug}'.")

        nav_button = app.query_one(f"#{button_id}", Button)
        app.current_media_type_filter_display_name = str(nav_button.label)

        media_window = app.query_one(f"#{app.TAB_MEDIA}-window").query_one("MediaWindow")
        media_window.media_active_view = view_to_activate

        app.current_media_type_filter_slug = type_slug
        app.media_current_page = 1
        
        # The MediaWindow's watcher will handle the search via its watch_media_active_view method
        # No need to call perform_media_search_and_display here as it would duplicate the work
        logger.info(f"MediaWindow activated for type: {type_slug}")

    except Exception as e:
        logger.error(f"Error in handle_media_nav_button_pressed for '{button_id}': {e}", exc_info=True)
        app.notify(f"Error switching media view: {str(e)[:100]}", severity="error")


async def handle_media_search_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles search button press within a specific media type view."""
    logger = app.loguru_logger
    button_id = event.button.id
    try:
        type_slug = button_id.replace("media-search-button-", "")
        search_input_id = f"media-search-input-{type_slug}"
        search_input_widget = app.query_one(f"#{search_input_id}", Input)
        search_term = search_input_widget.value.strip()
        
        # Get keyword filter
        keyword_filter_id = f"media-keyword-filter-{type_slug}"
        try:
            keyword_input_widget = app.query_one(f"#{keyword_filter_id}", Input)
            keyword_filter = keyword_input_widget.value.strip()
        except QueryError:
            keyword_filter = ""
            
        logger.info(f"Media search triggered for type '{type_slug}' with term: '{search_term}' and keywords: '{keyword_filter}'")
        app.media_current_page = 1 # Reset to page 1 for new search
        await perform_media_search_and_display(app, type_slug, search_term, keyword_filter)
    except QueryError as e:
        logger.error(f"UI component not found for media search button '{button_id}': {e}", exc_info=True)
        app.notify("Search UI error.", severity="error")
    except Exception as e:
        logger.error(f"Error in handle_media_search_button_pressed for '{button_id}': {e}", exc_info=True)
        app.notify(f"Error performing media search: {str(e)[:100]}", severity="error")


async def handle_media_search_input_changed(app: 'TldwCli', input_id: str, value: str) -> None:
    """Handles input changes in media search bars with debouncing."""
    logger = app.loguru_logger
    type_slug = input_id.replace("media-search-input-", "")

    if type_slug in app._media_search_timers and app._media_search_timers[type_slug]:
        app._media_search_timers[type_slug].stop()

    async def debounced_search():
        # Get search term from the search input
        search_term = ""
        search_input_id = f"media-search-input-{type_slug}"
        try:
            search_input_widget = app.query_one(f"#{search_input_id}", Input)
            search_term = search_input_widget.value.strip()
        except QueryError:
            pass
        
        # Get keyword filter
        keyword_filter = ""
        keyword_filter_id = f"media-keyword-filter-{type_slug}"
        try:
            keyword_input_widget = app.query_one(f"#{keyword_filter_id}", Input)
            keyword_filter = keyword_input_widget.value.strip()
        except QueryError:
            pass
            
        logger.info(f"Debounced media search executing for type '{type_slug}', term: '{search_term}', keywords: '{keyword_filter}'")
        app.media_current_page = 1
        await perform_media_search_and_display(app, type_slug, search_term, keyword_filter)

    app._media_search_timers[type_slug] = app.set_timer(0.6, debounced_search)


async def handle_media_list_item_selected(app: 'TldwCli', event: ListView.Selected) -> None:
    """
    Handles a media item being selected in any ListView within the Media tab.
    Populates the corresponding Markdown details pane with full, formatted content.
    """
    logger = app.loguru_logger
    list_view_id = event.list_view.id
    type_slug = ""

    if list_view_id and list_view_id.startswith("media-list-view-"):
        type_slug = list_view_id.replace("media-list-view-", "")
    else:
        logger.error(f"Could not determine type_slug from ListView ID: {list_view_id}")
        app.notify("Internal error: Could not identify media type for selection.", severity="error")
        return

    details_widget_id = f"media-details-widget-{type_slug}"

    try:
        from ..Widgets.media_details_widget import MediaDetailsWidget
        details_widget = app.query_one(f"#{details_widget_id}", MediaDetailsWidget)
    except QueryError:
        logger.error(f"MediaDetailsWidget '#{details_widget_id}' not found for type_slug '{type_slug}'.")
        app.notify(f"Details display area missing for {type_slug}.", severity="error")
        return

    # Show loading state
    details_widget.media_data = None

    if not hasattr(event.item, 'media_data') or not event.item.media_data:
        app.current_loaded_media_item = None
        details_widget.media_data = None
        return

    lightweight_media_data = event.item.media_data
    media_id_raw = lightweight_media_data.get('id')

    if media_id_raw is None:
        app.current_loaded_media_item = None
        details_widget.media_data = None
        return
    try:
        media_id = int(media_id_raw)
    except (ValueError, TypeError):
        app.current_loaded_media_item = None
        details_widget.media_data = None
        return

    if not app.media_db:
        app.current_loaded_media_item = None
        details_widget.media_data = None
        return

    logger.info(f"Fetching full details for media item ID: {media_id} for view {type_slug}")
    full_media_data = app.media_db.get_media_by_id(media_id, include_trash=True)

    if full_media_data is None:
        app.current_loaded_media_item = None
        details_widget.media_data = None
        return

    app.current_loaded_media_item = full_media_data
    
    # Update the widget with the full media data
    details_widget.update_media_data(full_media_data)
    logger.info(f"Displayed details for media ID {media_id} in '#{details_widget_id}'.")


async def handle_media_load_selected_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """
    Handles loading a selected media item's full details and displaying them
    as rendered Markdown. This function now essentially re-uses the selection logic.
    """
    logger = app.loguru_logger
    button_id = event.button.id
    type_slug = ""

    if button_id and button_id.startswith("media-load-selected-button-"):
        type_slug = button_id.replace("media-load-selected-button-", "")

    if not type_slug:
        logger.error(f"Could not determine type_slug from button ID: {button_id}")
        app.notify("Error determining media type for loading.", severity="error")
        return

    list_view_id = f"media-list-view-{type_slug}"
    details_widget_id = f"media-details-widget-{type_slug}"

    try:
        list_view = app.query_one(f"#{list_view_id}", ListView)
        from ..Widgets.media_details_widget import MediaDetailsWidget
        details_widget = app.query_one(f"#{details_widget_id}", MediaDetailsWidget)
    except QueryError as e:
        logger.error(f"UI component not found for media load selected '{button_id}': {e}", exc_info=True)
        app.notify("Load details UI error.", severity="error")
        return

    if not list_view.highlighted_child or not hasattr(list_view.highlighted_child, 'media_data'):
        app.notify("No media item selected from the list.", severity="warning")
        details_widget.media_data = None
        app.current_loaded_media_item = None
        return

    # Simulate a ListView.Selected event for the highlighted item
    # This will trigger the main handle_media_list_item_selected function
    # which contains the consolidated logic.
    # We create a dummy event object.
    class DummyListViewSelectedEvent:
        def __init__(self, list_view_widget, item_widget):
            self.list_view = list_view_widget
            self.item = item_widget
            self.control = list_view_widget # For compatibility with Textual's event structure

    dummy_event = DummyListViewSelectedEvent(list_view, list_view.highlighted_child)
    await handle_media_list_item_selected(app, dummy_event)
    logger.info(f"Triggered detail loading via button for type '{type_slug}' by simulating selection.")


async def handle_media_page_change_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles Next/Previous page button presses."""
    logger = app.loguru_logger
    button_id = event.button.id

    if "next" in button_id:
        app.media_current_page += 1
    elif "prev" in button_id and app.media_current_page > 1:
        app.media_current_page -= 1
    else:
        return

    type_slug = app.current_media_type_filter_slug # Relies on app state being correct
    search_term = ""
    keyword_filter = ""
    try:
        # Construct search input ID based on current type_slug
        search_input_widget = app.query_one(f"#media-search-input-{type_slug}", Input)
        search_term = search_input_widget.value
        
        # Get keyword filter
        keyword_filter_id = f"media-keyword-filter-{type_slug}"
        keyword_input_widget = app.query_one(f"#{keyword_filter_id}", Input)
        keyword_filter = keyword_input_widget.value
    except QueryError:
        logger.warning(f"Search input not found for type_slug '{type_slug}' during pagination.")
    except Exception as e:
        logger.error(f"Error getting search term during pagination for '{type_slug}': {e}")


    logger.info(f"Changing to page {app.media_current_page} for type '{type_slug}'")
    await perform_media_search_and_display(app, type_slug, search_term, keyword_filter)


async def perform_media_search_and_display(app: 'TldwCli', type_slug: str, search_term: str = "", keyword_filter: str = "") -> None:
    """Performs search in media DB and populates the ListView with rich, informative items."""
    logger = app.loguru_logger
    
    # Skip search for special windows that don't have standard media views
    if type_slug in ["collections-tags", "multi-item-review"]:
        logger.info(f"Skipping search for special window: {type_slug}")
        return
    
    list_view_id = f"media-list-view-{type_slug}"

    try:
        list_view = app.query_one(f"#{list_view_id}", ListView)
        await list_view.clear()

        try:
            from ..Widgets.media_details_widget import MediaDetailsWidget
            details_widget = app.query_one(f"#media-details-widget-{type_slug}", MediaDetailsWidget)
            details_widget.media_data = None  # Clear the widget
        except QueryError:
            logger.warning(f"MediaDetailsWidget '#media-details-widget-{type_slug}' not found for clearing.")
            pass # Continue if details display isn't critical for search itself

        if not app.media_db:
            raise RuntimeError("Media DB service not available.")

        media_types_filter = None
        # For "analysis-review", we search all types but display differently.
        # For other specific types, we filter by that type.
        if type_slug != "all-media" and type_slug != "analysis-review":
            db_media_type = type_slug.replace('-', '_') # Convert slug to DB type format if necessary
            media_types_filter = [db_media_type]

        # If type_slug is "analysis-review", media_types_filter remains None (search all types)

        query_arg = search_term if search_term else None
        fields_arg = ['title', 'content', 'author', 'url', 'type', 'analysis_content'] # Ensure analysis_content is searched
        
        # Parse keywords from filter
        keywords_list = None
        if keyword_filter:
            keywords_list = [k.strip() for k in keyword_filter.split(',') if k.strip()]

        # Check if we should show deleted items
        show_deleted = False
        try:
            # Try to get the MediaWindow and check its show_deleted_items state
            from ..UI.MediaWindow import MediaWindow
            media_window = app.query_one(MediaWindow)
            show_deleted = media_window.show_deleted_items
        except QueryError:
            logger.debug("MediaWindow not found, using default show_deleted=False")

        results, total_matches = app.media_db.search_media_db(
            search_query=query_arg,
            media_types=media_types_filter,
            search_fields=fields_arg,
            must_have_keywords=keywords_list,
            sort_by="last_modified_desc",
            page=app.media_current_page, # Use app.media_current_page directly
            results_per_page=RESULTS_PER_PAGE,
            include_trash=False,
            include_deleted=show_deleted
        )

        if not results:
            await list_view.append(ListItem(Label("No media items found.")))
        else:
            for item in results:
                title = item.get('title', 'Untitled')
                ingestion_date_raw = item.get('ingestion_date', '')
                
                # Handle both datetime objects and strings
                if isinstance(ingestion_date_raw, datetime):
                    ingestion_date = ingestion_date_raw.strftime('%Y-%m-%d')
                elif isinstance(ingestion_date_raw, str) and ingestion_date_raw:
                    ingestion_date = ingestion_date_raw.split('T')[0]
                else:
                    ingestion_date = 'N/A'

                # Determine snippet based on view type
                if type_slug == "analysis-review":
                    content_preview = item.get('analysis_content', '').strip()
                    snippet_label = "Analysis: "
                else:
                    content_preview = item.get('content', '').strip()
                    snippet_label = "Content: "

                snippet = (snippet_label + content_preview[:70] + '...') if content_preview else "No preview available."

                # Check if item is deleted
                is_deleted = item.get('deleted', 0) == 1
                title_display = f"[DELETED] {title}" if is_deleted else title
                
                # Create a richer ListItem with a Vertical layout
                rich_list_item = ListItem(
                    Vertical(
                        Label(f"{title_display}", classes="media-item-title media-item-deleted" if is_deleted else "media-item-title"),
                        Static(snippet, classes="media-item-snippet media-item-deleted" if is_deleted else "media-item-snippet"),
                        Static(f"Type: {item.get('type')}  |  Ingested: {ingestion_date}", classes="media-item-meta media-item-deleted" if is_deleted else "media-item-meta")
                    ),
                    classes="deleted-media-item" if is_deleted else ""
                )
                rich_list_item.media_data = item
                await list_view.append(rich_list_item)

        # Update pagination controls
        try:
            total_pages = math.ceil(total_matches / RESULTS_PER_PAGE) if total_matches > 0 else 1
            page_label = app.query_one(f"#media-page-label-{type_slug}", Label)
            prev_button = app.query_one(f"#media-prev-page-button-{type_slug}", Button)
            next_button = app.query_one(f"#media-next-page-button-{type_slug}", Button)

            page_label.update(f"Page {app.media_current_page} / {total_pages}")
            prev_button.disabled = (app.media_current_page <= 1)
            next_button.disabled = (app.media_current_page >= total_pages)
        except QueryError:
            pass

    except (QueryError, RuntimeError, Exception) as e:
        logger.error(f"Error during media search for type '{type_slug}': {str(e)}", exc_info=True)
        # Attempt to show error in the list view if possible
        try:
            list_view = app.query_one(f"#{list_view_id}", ListView)
            await list_view.clear()
            await list_view.append(ListItem(Label(f"Error loading: {str(e)[:50]}")))
        except (QueryError, AttributeError):
            # QueryError if list_view is not properly mounted/accessible
            # AttributeError if list_view is None or invalid
            pass


# REMOVE perform_analysis_review_search_and_display as its logic is now part of perform_media_search_and_display
# REMOVE handle_analysis_review_item_selected as its logic is now part of handle_media_list_item_selected

# format_media_details (Rich Text version) can be kept if used elsewhere, or removed if format_media_details_as_markdown is always preferred.
# For this task, we assume format_media_details_as_markdown is the target.

def format_media_details_as_markdown(app: 'TldwCli', media_data: Dict[str, Any]) -> str:
    """Formats media item details into a clean, scannable Markdown string."""
    if not media_data:
        return "### No Media Item Loaded"

    keywords_str = "N/A"
    media_id = media_data.get('id')
    if app.media_db and media_id:
        try:
            keywords = fetch_keywords_for_media(app.media_db, media_id)
            keywords_str = ", ".join(keywords) if keywords else "N/A"
        except Exception as e:
            keywords_str = f"*Error fetching keywords: {e}*"

    title = media_data.get('title', 'Untitled')

    # Create a well-structured metadata section
    metadata_section = "### Metadata\n\n"
    metadata_section += f"**ID:** `{media_data.get('id', 'N/A')}`\n"
    metadata_section += f"**UUID:** `{media_data.get('uuid', 'N/A')}`\n"
    metadata_section += f"**Type:** `{media_data.get('type', 'N/A')}`\n"
    metadata_section += f"**Author:** `{media_data.get('author', 'N/A')}`\n"
    metadata_section += f"**URL:** `{media_data.get('url', 'N/A')}`\n"
    metadata_section += f"**Keywords:** {keywords_str}\n"
    
    # Format timestamps in separate section
    timestamps_section = "\n### Timestamps\n\n"
    timestamps_section += f"**Ingested:** `{media_data.get('ingestion_date', 'N/A')}`\n"
    timestamps_section += f"**Modified:** `{media_data.get('last_modified', 'N/A')}`\n"

    content = media_data.get('content', 'N/A') or 'N/A'
    analysis_content = media_data.get('analysis_content', '') # Get analysis content

    # Assemble the final markdown string with better structure
    final_markdown = (
        f"# {title}\n\n"
        f"{metadata_section}"
        f"{timestamps_section}\n"
        "---\n\n"
        "### Content\n\n"
        f"{content}\n"
    )

    # Add Analysis section if content exists for it
    if analysis_content:
        final_markdown += (
            "\n---\n\n"
            "### Analysis\n\n"
            f"{analysis_content}\n"
        )

    return final_markdown

async def handle_media_metadata_update(app: 'TldwCli', event: MediaMetadataUpdateEvent) -> None:
    """
    Handles media metadata update requests from the MediaDetailsWidget.
    Updates the media item's metadata in the database and refreshes the display.
    """
    logger = app.loguru_logger
    
    try:
        if not app.media_db:
            raise RuntimeError("Media DB service not available")
        
        # Call the update method with the new metadata
        success, message = app.media_db.update_media_metadata(
            media_id=event.media_id,
            title=event.title,
            media_type=event.media_type,
            author=event.author,
            url=event.url,
            keywords=event.keywords
        )
        
        if success:
            # Show success notification
            app.notify(message, severity="information")
            
            # Refresh the media item display by fetching updated data
            updated_media = app.media_db.get_media_by_id(event.media_id)
            if updated_media:
                # Update the app's current loaded media item
                app.current_loaded_media_item = updated_media
                
                # Update the widget with new data and exit edit mode
                try:
                    from ..Widgets.media_details_widget import MediaDetailsWidget
                    details_widget = app.query_one(f"#media-details-widget-{event.type_slug}", MediaDetailsWidget)
                    details_widget.update_media_data(updated_media)
                    details_widget.edit_mode = False  # Exit edit mode
                except QueryError:
                    logger.warning(f"Could not find MediaDetailsWidget for type_slug: {event.type_slug}")
                
                # Also refresh the list view to reflect any title changes
                # Get current search term and keyword filter
                search_term = ""
                keyword_filter = ""
                try:
                    search_input = app.query_one(f"#media-search-input-{event.type_slug}", Input)
                    search_term = search_input.value
                    keyword_input = app.query_one(f"#media-keyword-filter-{event.type_slug}", Input)
                    keyword_filter = keyword_input.value
                except QueryError:
                    pass
                await perform_media_search_and_display(app, event.type_slug, search_term, keyword_filter)
            else:
                logger.error(f"Could not fetch updated media data for ID {event.media_id}")
                app.notify("Error refreshing media data", severity="error")
        else:
            app.notify(f"Failed to update media: {message}", severity="error")
            
    except Exception as e:
        logger.error(f"Error updating media metadata: {e}", exc_info=True)
        app.notify(f"Error updating metadata: {str(e)[:100]}", severity="error")


async def handle_media_delete_confirmation(app: 'TldwCli', event: MediaDeleteConfirmationEvent) -> None:
    """
    Handles media delete confirmation by showing a modal dialog.
    
    Note: This function is kept for backwards compatibility but the actual
    confirmation dialog is now handled directly in MediaWindow_v2 to avoid
    the NoActiveWorker error when using push_screen_wait.
    """
    logger.warning("handle_media_delete_confirmation called directly - should be handled in MediaWindow_v2")
    # The MediaWindow_v2 now handles the confirmation dialog directly


async def handle_media_undelete(app: 'TldwCli', event: MediaUndeleteEvent) -> None:
    """
    Handles media undelete requests.
    """
    logger = app.loguru_logger
    
    try:
        if not app.media_db:
            raise RuntimeError("Media DB service not available")
        
        # Get media info for notification
        media_item = app.media_db.get_media_by_id(event.media_id)
        if not media_item:
            app.notify("Media item not found", severity="error")
            return
            
        # Perform undelete
        success = app.media_db.undelete_media(event.media_id)
        
        if success:
            app.notify(f"'{media_item.get('title', 'Untitled')}' has been restored", severity="information")
            # MediaWindow_v2 will handle the refresh when it receives the event
        else:
            app.notify(f"Failed to restore '{media_item.get('title', 'Untitled')}'", severity="error")
            
    except Exception as e:
        logger.error(f"Error undeleting media: {e}", exc_info=True)
        app.notify(f"Error restoring media: {str(e)[:100]}", severity="error")


async def handle_media_generate_analysis(app: App, event: Button.Pressed) -> None:
    """Handle media generate analysis button press."""
    try:
        # Get the MediaWindow and find the MediaViewerPanel
        media_window = app.query_one("#media-window")
        if hasattr(media_window, '_actual_window') and media_window._actual_window:
            # It's a PlaceholderWindow, get the actual window
            media_window = media_window._actual_window
        
        # Find the MediaViewerPanel
        from tldw_chatbook.Widgets.Media.media_viewer_panel import MediaViewerPanel
        viewer_panel = media_window.query_one(MediaViewerPanel)
        
        # Trigger the analysis
        viewer_panel.handle_generate_analysis()
        
    except Exception as e:
        logger.error(f"Error generating media analysis: {e}", exc_info=True)
        app.notify(f"Error generating analysis: {str(e)[:100]}", severity="error")


async def handle_media_save_analysis(app: App, event: Button.Pressed) -> None:
    """Handle media save analysis button press."""
    try:
        media_window = app.query_one("#media-window")
        if hasattr(media_window, '_actual_window') and media_window._actual_window:
            media_window = media_window._actual_window
            
        from tldw_chatbook.Widgets.Media.media_viewer_panel import MediaViewerPanel
        viewer_panel = media_window.query_one(MediaViewerPanel)
        viewer_panel.handle_save_analysis()
        
    except Exception as e:
        logger.error(f"Error saving media analysis: {e}", exc_info=True)
        app.notify(f"Error saving analysis: {str(e)[:100]}", severity="error")


async def handle_media_edit_analysis(app: App, event: Button.Pressed) -> None:
    """Handle media edit analysis button press."""
    try:
        media_window = app.query_one("#media-window")
        if hasattr(media_window, '_actual_window') and media_window._actual_window:
            media_window = media_window._actual_window
            
        from tldw_chatbook.Widgets.Media.media_viewer_panel import MediaViewerPanel
        viewer_panel = media_window.query_one(MediaViewerPanel)
        viewer_panel.handle_edit_analysis()
        
    except Exception as e:
        logger.error(f"Error editing media analysis: {e}", exc_info=True)
        app.notify(f"Error editing analysis: {str(e)[:100]}", severity="error")


async def handle_media_overwrite_analysis(app: App, event: Button.Pressed) -> None:
    """Handle media overwrite analysis button press."""
    try:
        media_window = app.query_one("#media-window")
        if hasattr(media_window, '_actual_window') and media_window._actual_window:
            media_window = media_window._actual_window
            
        from tldw_chatbook.Widgets.Media.media_viewer_panel import MediaViewerPanel
        viewer_panel = media_window.query_one(MediaViewerPanel)
        viewer_panel.handle_overwrite_analysis()
        
    except Exception as e:
        logger.error(f"Error overwriting media analysis: {e}", exc_info=True)
        app.notify(f"Error overwriting analysis: {str(e)[:100]}", severity="error")


async def handle_media_sidebar_toggle(app: App, event: Button.Pressed) -> None:
    """Handle media sidebar toggle button press."""
    try:
        # Toggle the media sidebar collapsed state
        if hasattr(app, 'media_sidebar_collapsed'):
            app.media_sidebar_collapsed = not app.media_sidebar_collapsed
        else:
            app.notify("Media sidebar state not available", severity="warning")
            
    except Exception as e:
        logger.error(f"Error toggling media sidebar: {e}", exc_info=True)


async def handle_media_list_collapse(app: App, event: Button.Pressed) -> None:
    """Handle media list collapse button press."""
    try:
        # Toggle the media list collapsed state
        media_window = app.query_one("#media-window")
        if hasattr(media_window, '_actual_window') and media_window._actual_window:
            media_window = media_window._actual_window
            
        # Toggle the collapsible state
        from tldw_chatbook.UI.MediaWindow_v2 import MediaWindow
        if isinstance(media_window, MediaWindow):
            if hasattr(media_window, 'toggle_media_list'):
                media_window.toggle_media_list()
            else:
                app.notify("Media list toggle not available", severity="warning")
                
    except Exception as e:
        logger.error(f"Error collapsing media list: {e}", exc_info=True)


async def handle_media_refresh_button_pressed(app: App, event: Button.Pressed) -> None:
    """Handle media refresh button press."""
    try:
        media_window = app.query_one("#media-window")
        if hasattr(media_window, '_actual_window') and media_window._actual_window:
            media_window = media_window._actual_window
            
        # Trigger refresh on the media window
        if hasattr(media_window, 'refresh_current_view'):
            await media_window.refresh_current_view()
        else:
            app.notify("Media refresh not available", severity="warning")
            
    except Exception as e:
        logger.error(f"Error refreshing media view: {e}", exc_info=True)
        app.notify(f"Error refreshing media: {str(e)[:100]}", severity="error")


# --- Button Handler Map ---
# This map will be dynamically generated in app.py's _build_handler_map based on _media_types_for_ui.
# However, ensure the handler names referenced there match the functions here.
# For example, handle_media_nav_button_pressed, handle_media_search_button_pressed,
# handle_media_load_selected_button_pressed, handle_media_page_change_button_pressed.

MEDIA_BUTTON_HANDLERS = {
    # Nav buttons (example, will be generated dynamically)
    # "media-nav-all-media": handle_media_nav_button_pressed,
    # "media-nav-video": handle_media_nav_button_pressed,
    # ... etc for all slugs ...
    # "media-nav-analysis-review": handle_media_nav_button_pressed,

    # Search buttons (example, will be generated dynamically)
    # "media-search-button-all-media": handle_media_search_button_pressed,
    # ... etc for all slugs ...

    # Load selected buttons (example, will be generated dynamically)
    # "media-load-selected-button-all-media": handle_media_load_selected_button_pressed,
    # ... etc for all slugs ...

    # Pagination buttons (example, will be generated dynamically)
    # "media-prev-page-button-all-media": handle_media_page_change_button_pressed,
    # "media-next-page-button-all-media": handle_media_page_change_button_pressed,
    # ... etc for all slugs ...
    
    # Media control buttons
    "media-refresh": handle_media_refresh_button_pressed,
    
    # Media panel buttons
    "generate-analysis-btn": handle_media_generate_analysis,
    "save-analysis-btn": handle_media_save_analysis,
    "edit-analysis-btn": handle_media_edit_analysis,
    "overwrite-analysis-btn": handle_media_overwrite_analysis,
    
    # Analysis navigation buttons
    "prev-analysis-btn": lambda app, event: None,  # Handled by widget itself
    "next-analysis-btn": lambda app, event: None,  # Handled by widget itself
    "delete-analysis-btn": lambda app, event: None,  # Handled by widget itself
    
    # Media sidebar toggle
    "media-sidebar-toggle": handle_media_sidebar_toggle,
    "collapse-media-list": handle_media_list_collapse,
}
# Note: The actual assignment of handlers to specific button IDs now happens more dynamically
# in app.py's _build_handler_map, based on `self._media_types_for_ui`.
# The MEDIA_BUTTON_HANDLERS dict in this file might become less important if all handlers are
# assigned directly by name in app.py. However, keeping it for reference or if some buttons
# don't fit the dynamic pattern. For this refactor, the dynamic assignment in app.py for
# media type specific buttons is key.

#
# End of media_events.py
########################################################################################################################
