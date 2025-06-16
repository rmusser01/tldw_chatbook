# chat_events_sidebar.py
#
# Imports
#
# Standard Library Imports
import math
from typing import TYPE_CHECKING, Dict, Any, Optional
import logging

# 3rd-party Libraries
from loguru import logger as loguru_logger, logger
from textual.widgets import ListItem, Input, ListView, TextArea, Button, Label
from textual.css.query import QueryError
from textual.app import App
#
# Local Imports
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Event_Handlers.media_events import RESULTS_PER_PAGE

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli  # Assuming your app class is TldwCli
#
###########################################################################################################################
#
# Globals
#
RESULTS_PER_PAGE = 20
#
###########################################################################################################################
#
# Functions:
#


def _clear_and_disable_media_display(app: 'TldwCli'):
    """Helper to clear media display TextAreas and disable copy buttons."""
    app.current_sidebar_media_item = None
    try:
        app.query_one("#chat-media-title-display", TextArea).clear()
        app.query_one("#chat-media-content-display", TextArea).clear()
        app.query_one("#chat-media-author-display", TextArea).clear()
        app.query_one("#chat-media-url-display", TextArea).clear()
        app.query_one("#chat-media-copy-title-button", Button).disabled = True
        app.query_one("#chat-media-copy-content-button", Button).disabled = True
        app.query_one("#chat-media-copy-author-button", Button).disabled = True
        app.query_one("#chat-media-copy-url-button", Button).disabled = True
    except QueryError as e:
         loguru_logger.warning(f"Could not find a media display/copy widget to clear/disable: {e}")


async def perform_media_sidebar_search(app: 'TldwCli', search_term: str = ""):
    """
    Performs a search for media items based on the given search term and populates the results.
    This function is used by tests and is a simplified version of perform_media_search.
    """
    loguru_logger.debug(f"Performing media sidebar search with term: '{search_term}'")
    try:
        results_list_view = app.query_one("#chat-media-search-results-listview", ListView)
    except QueryError as e:
        loguru_logger.error(f"Error querying media search UI elements: {e}")
        app.notify(f"Error accessing media search UI: {e}", severity="error")
        return

    await results_list_view.clear()
    _clear_and_disable_media_display(app)

    if not search_term.strip():
        loguru_logger.debug("Search term is empty, clearing results.")
        return

    if not app.media_db:
        loguru_logger.error("app.media_db is not available.")
        app.notify("Media database service not initialized.", severity="error")
        return

    try:
        search_fields = ['title', 'content', 'author', 'keywords', 'notes']
        media_types_filter = None

        media_items = app.media_db.search_media_db(
            search_query=search_term,
            search_fields=search_fields,
            media_types=media_types_filter,
            include_trash=False,
            include_deleted=False,
            page=1,
            results_per_page=50
        )

        if not media_items:
            # FIX: Await the async append method.
            await results_list_view.append(ListItem(Label("No media found.")))
        else:
            for item_dict in media_items:
                if isinstance(item_dict, dict):
                    title = item_dict.get('title', 'Untitled')
                    media_id = item_dict.get('media_id', 'Unknown ID')
                    display_label = f"{title} (ID: {media_id})"
                    list_item = ListItem(Label(display_label))
                    setattr(list_item, 'media_data', item_dict)
                    await results_list_view.append(list_item)
                else:
                    loguru_logger.warning(f"Skipping non-dictionary item from DB search results: {item_dict}")

    except Exception as e:
        loguru_logger.error(f"Exception during media search: {e}", exc_info=True)
        app.notify(f"Error during media search: {e}", severity="error")
        await results_list_view.clear()
        # FIX: Await the async append method.
        await results_list_view.append(ListItem(Label(f"Search error.")))


async def perform_media_search(app: 'TldwCli'):
    """
    Performs a search for media items based on sidebar inputs and populates the results.
    """
    loguru_logger.debug("Performing media search in chat sidebar.")
    try:
        results_list_view = app.query_one("#chat-media-search-results-listview", ListView)
        search_input = app.query_one("#chat-media-search-input", Input)
        keyword_input = app.query_one("#chat-media-keyword-filter-input", Input)
        page_label = app.query_one("#chat-media-page-label", Label)
        prev_button = app.query_one("#chat-media-prev-page-button", Button)
        next_button = app.query_one("#chat-media-next-page-button", Button)
    except QueryError as e:
        loguru_logger.error(f"Error querying media search UI elements: {e}")
        app.notify(f"Error accessing media search UI: {e}", severity="error")
        return

    await results_list_view.clear()
    _clear_and_disable_media_display(app)

    search_term = search_input.value.strip()
    keywords_str = keyword_input.value.strip()
    keywords_list = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
    loguru_logger.debug(f"Media Search - Term: '{search_term}', Keywords: {keywords_list}")

    # Store the original search term for LIKE queries
    original_search_term = search_term

    # For very short search terms (1-2 characters), skip exact phrase matching
    # as it's likely to be too restrictive for partial matches
    if search_term and len(search_term) <= 2:
        loguru_logger.debug(f"Short search term '{search_term}' detected, skipping exact phrase matching")
        # Keep the original search term for LIKE queries
        pass
    # For longer search terms, if they don't already have quotes and don't contain any special characters,
    # wrap them in quotes to enable exact phrase matching in FTS
    elif search_term and not search_term.startswith('"') and not search_term.endswith('"'):
        # Check if it's a simple phrase without special FTS syntax
        if not any(char in search_term for char in '*+-()'):
            # Wrap in quotes for exact phrase matching
            exact_search_term = f'"{search_term}"'
            loguru_logger.debug(f"Converting search term to exact phrase: '{exact_search_term}'")
            search_term = exact_search_term


    if not app.media_db:
        loguru_logger.error("app.media_db is not available.")
        app.notify("Media database service not initialized.", severity="error")
        return

    try:
        if not app.media_db:
            loguru_logger.error("app.media_db is not available.")
            app.notify("Media database service not initialized.", severity="error")
            return

        db_instance = app.media_db

        search_fields = ['title', 'content', 'author', 'keywords', 'notes']
        media_types_filter = None

        # If no search criteria provided, we'll do a general search without keyword filtering
        if not keywords_list and not search_term:
            loguru_logger.debug("No search term or keywords provided, performing general search")
            # We'll leave both search_query and must_have_keywords as None to get all results

        loguru_logger.debug(f"Media Search - Requesting page: {app.media_search_current_page}")
        # loguru_logger.debug(f"Searching media DB with term: '{search_term}', fields: {search_fields}, types: {media_types_filter}") # This is a bit redundant with the one above

        # Only apply keyword filtering if keywords were explicitly provided
        must_have_keywords_param = keywords_list if keywords_list else None

        loguru_logger.debug(f"Media Search - Parameters: search_query={search_term if search_term else 'None'}, original_term={original_search_term if original_search_term else 'None'}, must_have_keywords={must_have_keywords_param}")

        # Try with the exact phrase search first (quoted term)
        media_items, total_matches = db_instance.search_media_db(
            search_query=search_term if search_term else None,
            search_fields=search_fields,
            media_types=media_types_filter,
            date_range=None,  # No date range filtering
            must_have_keywords=must_have_keywords_param,
            must_not_have_keywords=None,
            sort_by="last_modified_desc",  # Default sort order
            media_ids_filter=None,  # No specific media IDs to filter
            page=app.media_search_current_page, # Use current page from app
            results_per_page=RESULTS_PER_PAGE,
            include_trash=False,
            include_deleted=False,
        )

        # If no results with exact phrase, try with the original term (without quotes)
        if total_matches == 0 and search_term != original_search_term:
            loguru_logger.debug(f"No results with exact phrase search, trying with original term: '{original_search_term}'")
            media_items, total_matches = db_instance.search_media_db(
                search_query=original_search_term,
                search_fields=search_fields,
                media_types=media_types_filter,
                date_range=None,  # No date range filtering
                must_have_keywords=must_have_keywords_param,
                must_not_have_keywords=None,
                sort_by="last_modified_desc",  # Default sort order
                media_ids_filter=None,  # No specific media IDs to filter
                page=app.media_search_current_page, # Use current page from app
                results_per_page=RESULTS_PER_PAGE,
                include_trash=False,
                include_deleted=False,
            )
        loguru_logger.debug(f"Media Search - DB returned total_matches: {total_matches}, items_for_page: {len(media_items)}")

        # Calculate total pages
        if total_matches > 0:
            app.media_search_total_pages = math.ceil(total_matches / RESULTS_PER_PAGE)
        else:
            app.media_search_total_pages = 1 # Ensure at least 1 page even if no results

        loguru_logger.debug(f"Media Search - Calculated total_pages: {app.media_search_total_pages}")
        # loguru_logger.info(f"Media search: current_page={app.media_search_current_page}, total_matches={total_matches}, total_pages={app.media_search_total_pages}") # loguru_logger.info is a bit verbose for this

        # Update page label
        page_label.update(f"Page {app.media_search_current_page}/{app.media_search_total_pages}")

        # Enable/disable pagination buttons
        prev_button.disabled = app.media_search_current_page == 1
        next_button.disabled = app.media_search_current_page >= app.media_search_total_pages
        loguru_logger.debug(f"Media Search - Prev button disabled: {prev_button.disabled}, Next button disabled: {next_button.disabled}")

        if not media_items:
            # FIX: Await the async append method.
            await results_list_view.append(ListItem(Label("No media found.")))
        else:
            # Get all media IDs to fetch keywords in batch
            media_ids = [item.get('id') for item in media_items if isinstance(item, dict) and item.get('id')]

            # Fetch keywords for all media items in one batch operation
            keywords_map = {}
            if media_ids:
                try:
                    keywords_map = db_instance.fetch_keywords_for_media_batch(media_ids)
                    loguru_logger.debug(f"Fetched keywords for {len(keywords_map)} media items")
                except Exception as e:
                    loguru_logger.error(f"Error fetching keywords batch: {e}")
                    # Continue without keywords if fetching fails

            for item_dict in media_items:
                if isinstance(item_dict, dict):
                    title = item_dict.get('title', 'Untitled')
                    author = item_dict.get('author', 'Unknown Author')
                    uuid_value = item_dict.get('uuid', 'Unknown')

                    # Get keywords for this media item
                    media_id = item_dict.get('id')
                    keywords = keywords_map.get(media_id, []) if media_id else []
                    keywords_str = ", ".join(keywords) if keywords else "None"

                    # Create a formatted display with all required fields
                    display_label = f"Title: {title}\nAuthor: {author}\nID: {uuid_value}\nKeywords: {keywords_str}"

                    list_item = ListItem(Label(display_label))
                    setattr(list_item, 'media_data', item_dict)
                    await results_list_view.append(list_item)
                else:
                    loguru_logger.warning(f"Skipping non-dictionary item from DB search results: {item_dict}")

    except Exception as e:
        loguru_logger.error(f"Exception during media search: {e}", exc_info=True)
        app.notify(f"Error during media search: {e}", severity="error")
        await results_list_view.clear()
        # FIX: Await the async append method.
        await results_list_view.append(ListItem(Label(f"Search error.")))


async def handle_chat_media_sidebar_input_changed(app: 'TldwCli'):
    """
    Handles changes in the media search input, debouncing the search.
    """
    loguru_logger.debug("Media search input changed, debouncing search.")

    async def debounced_search():
        app.media_search_current_page = 1  # Reset to page 1 for new search
        await perform_media_search(app)

    app._media_sidebar_search_timer = app.set_timer(0.5, debounced_search)

async def handle_media_item_selected(app: 'TldwCli', list_item: ListItem) -> None:
    """
    Loads the selected media item's details into the review display.
    """
    loguru_logger.debug("Media item selected in ListView.")
    _clear_and_disable_media_display(app)  # Clear previous state

    if not hasattr(list_item, 'media_data'):
        app.notify("Selected item has no data.", severity="warning")
        return

    media_data_light = getattr(list_item, 'media_data')
    media_id = media_data_light.get('id')

    if not media_id or not app.media_db:
        app.notify("Cannot load details: Invalid item ID or DB not available.", severity="error")
        return

    # Fetch full details from the database
    full_media_data = app.media_db.get_media_by_id(media_id)

    if not full_media_data:
        app.notify(f"Could not load details for media ID {media_id}.", severity="error")
        return

    app.current_sidebar_media_item = full_media_data
    loguru_logger.info(f"Loaded media ID {media_id} into sidebar for review.")

    try:
        title = full_media_data.get('title', '')
        content = full_media_data.get('content', '')
        author = full_media_data.get('author', '')
        url = full_media_data.get('url', '')

        app.query_one("#chat-media-title-display", TextArea).load_text(title)
        app.query_one("#chat-media-content-display", TextArea).load_text(content)
        app.query_one("#chat-media-author-display", TextArea).load_text(author)
        app.query_one("#chat-media-url-display", TextArea).load_text(url)

        # Enable copy buttons for fields that have content
        app.query_one("#chat-media-copy-title-button", Button).disabled = not bool(title)
        app.query_one("#chat-media-copy-content-button", Button).disabled = not bool(content)
        app.query_one("#chat-media-copy-author-button", Button).disabled = not bool(author)
        app.query_one("#chat-media-copy-url-button", Button).disabled = not bool(url)
    except QueryError as e:
        loguru_logger.error(f"Error populating media detail widgets: {e}")


async def handle_media_page_change(app: 'TldwCli', direction: int):
    """Handles next/previous page requests."""
    new_page = app.media_search_current_page + direction
    if 1 <= new_page <= app.media_search_total_pages:
        app.media_search_current_page = new_page
        await perform_media_search(app)
    else:
        loguru_logger.debug(f"Page change to {new_page} blocked (out of bounds).")


async def handle_chat_media_copy_title_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Copies the title of the currently loaded sidebar media to clipboard."""
    loguru_logger.debug("Copy Title button pressed.")
    if app.current_sidebar_media_item and app.current_sidebar_media_item.get('title'):
        title = str(app.current_sidebar_media_item['title'])
        app.copy_to_clipboard(title)
        app.notify("Title copied to clipboard.")
        loguru_logger.info(f"Copied title: '{title}'")
    else:
        app.notify("No media title to copy.", severity="warning")
        loguru_logger.warning("No media title available to copy.")


async def handle_chat_media_copy_content_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Copies the content of the currently loaded sidebar media to clipboard."""
    loguru_logger.debug("Copy Content button pressed.")
    if app.current_sidebar_media_item and app.current_sidebar_media_item.get('content'):
        content = str(app.current_sidebar_media_item['content'])
        app.copy_to_clipboard(content)
        app.notify("Content copied to clipboard.")
        loguru_logger.info("Copied content (length: {})", len(content))
    else:
        app.notify("No media content to copy.", severity="warning")
        logger.warning("No media content available to copy.")


async def handle_chat_media_copy_author_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Copies the author of the currently loaded sidebar media to clipboard."""
    logger.debug("Copy Author button pressed.")
    if app.current_sidebar_media_item and app.current_sidebar_media_item.get('author'):
        author = str(app.current_sidebar_media_item['author'])
        app.copy_to_clipboard(author)
        app.notify("Author copied to clipboard.")
        logger.info(f"Copied author: '{author}'")
    else:
        app.notify("No media author to copy.", severity="warning")
        logger.warning("No media author available to copy.")


async def handle_chat_media_copy_url_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Copies the URL of the currently loaded sidebar media to clipboard."""
    logger.debug("Copy URL button pressed.")
    if app.current_sidebar_media_item and app.current_sidebar_media_item.get('url'):
        url = str(app.current_sidebar_media_item['url'])
        app.copy_to_clipboard(url)
        app.notify("URL copied to clipboard.")
        logger.info(f"Copied URL: '{url}'")
    else:
        app.notify("No media URL to copy.", severity="warning")
        logger.warning("No media URL available to copy.")


async def handle_chat_media_search_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the search button press in the media search section of the chat sidebar."""
    logger.debug("Media search button pressed.")
    app.media_search_current_page = 1  # Reset to page 1 for new search
    await perform_media_search(app)



# --- Button Handler Map ---
CHAT_SIDEBAR_BUTTON_HANDLERS = {
    "chat-media-copy-title-button": handle_chat_media_copy_title_button_pressed,
    "chat-media-copy-content-button": handle_chat_media_copy_content_button_pressed,
    "chat-media-copy-author-button": handle_chat_media_copy_author_button_pressed,
    "chat-media-copy-url-button": handle_chat_media_copy_url_button_pressed,
    "chat-media-prev-page-button": lambda app, event: handle_media_page_change(app, -1),
    "chat-media-next-page-button": lambda app, event: handle_media_page_change(app, 1),
    "chat-media-search-button": handle_chat_media_search_button_pressed,
}


#
# End of chat_events_sidebar.py
########################################################################################################################
