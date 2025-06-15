# tldw_chatbook/Event_Handlers/media_events.py
#
#
# Imports
import math
from typing import TYPE_CHECKING, Dict, Any
#
# 3rd-party Libraries
from textual.containers import Vertical
from textual.widgets import ListView, Input, TextArea, Label, ListItem, Button, Markdown, Static  # Added ListItem
from textual.css.query import QueryError
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
        await perform_media_search_and_display(app, type_slug, search_term="")

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
        logger.info(f"Media search triggered for type '{type_slug}' with term: '{search_term}'")
        app.media_current_page = 1 # Reset to page 1 for new search
        await perform_media_search_and_display(app, type_slug, search_term)
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
        logger.info(f"Debounced media search executing for type '{type_slug}', term: '{value.strip()}'")
        app.media_current_page = 1
        await perform_media_search_and_display(app, type_slug, value.strip())

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

    details_display_widget_id = f"media-details-display-{type_slug}"

    try:
        details_display = app.query_one(f"#{details_display_widget_id}", Markdown)
    except QueryError:
        logger.error(f"Markdown details display widget '#{details_display_widget_id}' not found for type_slug '{type_slug}'.")
        app.notify(f"Details display area missing for {type_slug}.", severity="error")
        return

    await details_display.update("### Loading details...")

    if not hasattr(event.item, 'media_data') or not event.item.media_data:
        await details_display.update("### Error: Selected item has no displayable data.")
        app.current_loaded_media_item = None
        return

    lightweight_media_data = event.item.media_data
    media_id_raw = lightweight_media_data.get('id')

    if media_id_raw is None:
        await details_display.update("### Error: Selected item has no ID.")
        app.current_loaded_media_item = None
        return
    try:
        media_id = int(media_id_raw)
    except (ValueError, TypeError):
        await details_display.update(f"### Error: Invalid media ID format '{media_id_raw}'.")
        app.current_loaded_media_item = None
        return

    if not app.media_db:
        await details_display.update("### Error: Database connection is not available.")
        app.current_loaded_media_item = None
        return

    logger.info(f"Fetching full details for media item ID: {media_id} for view {type_slug}")
    full_media_data = app.media_db.get_media_by_id(media_id, include_trash=True)

    if full_media_data is None:
        await details_display.update(
            f"### Error\n\nCould not find media item with ID `{media_id}`. It may have been deleted."
        )
        app.current_loaded_media_item = None
        return

    app.current_loaded_media_item = full_media_data

    # Special formatting for "analysis-review"
    if type_slug == "analysis-review":
        title = full_media_data.get('title', 'Untitled')
        url = full_media_data.get('url', 'No URL')
        analysis_content = full_media_data.get('analysis_content', '')
        if not analysis_content:
            analysis_content = "No analysis available for this item."
        markdown_details_string = f"## {title}\n\n**URL:** {url}\n\n### Analysis\n{analysis_content}"
    else:
        markdown_details_string = format_media_details_as_markdown(app, full_media_data)

    await details_display.update(markdown_details_string)
    details_display.scroll_home(animate=False)
    logger.info(f"Displayed details for media ID {media_id} in '#{details_display_widget_id}'.")


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
    details_display_widget_id = f"media-details-display-{type_slug}"

    try:
        list_view = app.query_one(f"#{list_view_id}", ListView)
        details_display = app.query_one(f"#{details_display_widget_id}", Markdown) # Expect Markdown
    except QueryError as e:
        logger.error(f"UI component not found for media load selected '{button_id}': {e}", exc_info=True)
        app.notify("Load details UI error.", severity="error")
        return

    if not list_view.highlighted_child or not hasattr(list_view.highlighted_child, 'media_data'):
        app.notify("No media item selected from the list.", severity="warning")
        await details_display.update("### No item selected from the list.")
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
    try:
        # Construct search input ID based on current type_slug
        search_input_widget = app.query_one(f"#media-search-input-{type_slug}", Input)
        search_term = search_input_widget.value
    except QueryError:
        logger.warning(f"Search input not found for type_slug '{type_slug}' during pagination.")
    except Exception as e:
        logger.error(f"Error getting search term during pagination for '{type_slug}': {e}")


    logger.info(f"Changing to page {app.media_current_page} for type '{type_slug}'")
    await perform_media_search_and_display(app, type_slug, search_term)


async def perform_media_search_and_display(app: 'TldwCli', type_slug: str, search_term: str = "") -> None:
    """Performs search in media DB and populates the ListView with rich, informative items."""
    logger = app.loguru_logger
    list_view_id = f"media-list-view-{type_slug}"
    details_display_id = f"media-details-display-{type_slug}"

    try:
        list_view = app.query_one(f"#{list_view_id}", ListView)
        await list_view.clear()

        try:
            details_display = app.query_one(f"#{details_display_id}", Markdown) # Expect Markdown
            await details_display.update("### Select an item to see details") # Use await and update
        except QueryError:
            logger.warning(f"Details display widget '#{details_display_id}' not found for clearing.")
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

        results, total_matches = app.media_db.search_media_db(
            search_query=query_arg,
            media_types=media_types_filter,
            search_fields=fields_arg,
            sort_by="last_modified_desc",
            page=app.media_current_page, # Use app.media_current_page directly
            results_per_page=RESULTS_PER_PAGE,
            include_trash=False,
            include_deleted=False
        )

        if not results:
            await list_view.append(ListItem(Label("No media items found.")))
        else:
            for item in results:
                title = item.get('title', 'Untitled')
                ingestion_date_str = item.get('ingestion_date', '')
                ingestion_date = ingestion_date_str.split('T')[0] if ingestion_date_str else 'N/A'

                # Determine snippet based on view type
                if type_slug == "analysis-review":
                    content_preview = item.get('analysis_content', '').strip()
                    snippet_label = "Analysis: "
                else:
                    content_preview = item.get('content', '').strip()
                    snippet_label = "Content: "

                snippet = (snippet_label + content_preview[:70] + '...') if content_preview else "No preview available."

                # Create a richer ListItem with a Vertical layout
                rich_list_item = ListItem(
                    Vertical(
                        Label(f"{title}", classes="media-item-title"),
                        Static(snippet, classes="media-item-snippet"),
                        Static(f"Type: {item.get('type')}  |  Ingested: {ingestion_date}", classes="media-item-meta")
                    )
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
        logger.error(f"Error during media search for type '{type_slug}': {e}", exc_info=True)
        # Attempt to show error in the list view if possible
        try:
            list_view = app.query_one(f"#{list_view_id}", ListView)
            await list_view.clear()
            await list_view.append(ListItem(Label(f"Error loading: {str(e)[:50]}")))
        except: #pylint: disable=bare-except
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

    # Create a compact, multi-line metadata block
    meta_header = (
        f"**ID:** `{media_data.get('id', 'N/A')}`  **UUID:** `{media_data.get('uuid', 'N/A')}`\n"
        f"**Type:** `{media_data.get('type', 'N/A')}`  **Author:** `{media_data.get('author', 'N/A')}`\n"
        f"**URL:** `{media_data.get('url', 'N/A')}`\n"
        f"**Keywords:** {keywords_str}"
    )

    # Format timestamps
    ingested = f"**Ingested:** `{media_data.get('ingestion_date', 'N/A')}`"
    modified = f"**Modified:** `{media_data.get('last_modified', 'N/A')}`"

    content = media_data.get('content', 'N/A') or 'N/A'
    analysis_content = media_data.get('analysis_content', '') # Get analysis content

    # Assemble the final markdown string
    final_markdown = (
        f"## {title}\n\n"
        f"{meta_header}\n\n"
        "---\n\n"
        f"{ingested}\n{modified}\n\n"
        "### Content\n\n"
        "```text\n" # Using text for now, can be ```markdown if content is markdown
        f"{content}\n"
        "```"
    )

    # Add Analysis section if content exists for it
    if analysis_content:
        final_markdown += (
            "\n\n### Analysis\n\n"
            "```text\n" # Or ```markdown if analysis is markdown
            f"{analysis_content}\n"
            "```"
        )

    return final_markdown

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
