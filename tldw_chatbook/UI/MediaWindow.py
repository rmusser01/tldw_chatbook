# tldw_chatbook/UI/MediaWindow.py
#
#
# Imports
from typing import TYPE_CHECKING, List, Optional
#
# Third-party Libraries
from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from textual.css.query import QueryError
from textual.reactive import reactive
from textual.widgets import Static, Button, Label, Input, ListView, TextArea, Markdown, Checkbox
#
# Local Imports
from ..Utils.text import slugify
from ..Event_Handlers import media_events
from ..Event_Handlers.media_events import MediaDeleteConfirmationEvent, MediaUndeleteEvent, MediaMetadataUpdateEvent
from ..Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE
from ..Widgets.media_details_widget import MediaDetailsWidget
if TYPE_CHECKING:
    from ..app import TldwCli
#
########################################################################################################################
#
# Functions:

MEDIA_SUB_TABS = [
    ("Video/Audio", "video-audio"),
    ("Documents", "documents"),
    ("PDFs", "pdfs"),
    ("Ebooks", "ebooks"),
    ("Websites", "websites"),
    ("MediaWiki", "mediawiki"),
    ("Analysis Review", "analysis-review"),
    ("Placeholder", "placeholder")
]

class MediaWindow(Container):
    """
    A fully self-contained component for the Media Tab, featuring a collapsible
    sidebar and a two-pane browser for media types.
    """
    
    # CSS to ensure views are hidden by default
    DEFAULT_CSS = """
    MediaWindow {
        layout: horizontal;
        width: 100%;
        height: 100%;
    }
    
    .media-view-area {
        display: none;
        layout: horizontal;
    }
    
    #media-nav-container {
        width: 30;
        height: 100%;
    }
    
    #media-nav-container.collapsed {
        width: 0;
        display: none;
    }
    
    #media-content-pane {
        width: 1fr;
        height: 100%;
    }
    """
    
    # --- STATE LIVES HERE NOW ---
    media_sidebar_collapsed: reactive[bool] = reactive(False)
    media_active_view: reactive[Optional[str]] = reactive(None)
    show_deleted_items: reactive[bool] = reactive(False)

    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.media_types_from_db: List[str] = getattr(self.app_instance, '_media_types_for_ui', [])
        self.log.debug(f"MediaWindow __init__: Received media types: {self.media_types_from_db}")

    # --- WATCHERS LIVE HERE NOW ---
    def watch_media_sidebar_collapsed(self, collapsed: bool) -> None:
        """Dynamically adjusts the media browser panes when the sidebar is collapsed or expanded."""
        try:
            # Add class to self (MediaWindow) for grid column adjustment
            self.set_class(collapsed, "sidebar-collapsed")
            
            # Target the container for collapsing
            nav_container = self.query_one("#media-nav-container")
            toggle_button = self.query_one("#media-sidebar-toggle-button")
            nav_container.set_class(collapsed, "collapsed")
            toggle_button.set_class(collapsed, "collapsed")
        except QueryError as e:
            self.log.warning(f"UI component not found during media sidebar collapse: {e}")

    def watch_media_active_view(self, old_view: Optional[str], new_view: Optional[str]) -> None:
        """Shows/hides the relevant content view when the active view slug changes."""
        self.log.debug(f"MediaWindow.watch_media_active_view: old='{old_view}', new='{new_view}'")
        try:
            # Hide all media views first to ensure only one is active
            for view_container in self.query(".media-view-area"):
                if view_container.id: # Make sure it has an ID
                    view_container.styles.display = "none"

            # Then show the new active view
            if new_view:
                view_to_show = self.query_one(f"#{new_view}")
                view_to_show.styles.display = "block" # Textual only supports block or none
                self.log.info(f"MediaWindow: Set display to 'block' for #{new_view}")

                type_slug = new_view.replace("media-view-", "")

                # Update app's reactive variables for current filter slug and display name
                self.app_instance.current_media_type_filter_slug = type_slug

                nav_button_id = f"media-nav-{type_slug}"
                try:
                    # Try to find the button in MediaWindow itself first
                    nav_button = self.query_one(f"#{nav_button_id}", Button)
                    self.app_instance.current_media_type_filter_display_name = str(nav_button.label)
                except QueryError:
                    # Fallback to querying the app instance if not found in MediaWindow
                    try:
                        nav_button_app_query = self.app_instance.query_one(f"#{nav_button_id}", Button)
                        self.app_instance.current_media_type_filter_display_name = str(nav_button_app_query.label)
                    except QueryError:
                        # Specific fallback for known slugs like 'analysis-review'
                        if type_slug == "analysis-review":
                            self.app_instance.current_media_type_filter_display_name = "Analysis Review"
                        else:
                            self.log.warning(f"Could not find nav button {nav_button_id} in MediaWindow or App to update display name for slug '{type_slug}'.")

                # Perform search for the newly activated view (skip special windows)
                # Ensure the app's media_current_page is reset for a new view activation if that's desired behavior
                # self.app_instance.media_current_page = 1 # Already done in handle_nav_button_press
                if type_slug not in ["collections-tags", "multi-item-review"]:
                    # Create a coroutine and run it with run_worker
                    async def perform_search():
                        await media_events.perform_media_search_and_display(self.app_instance, type_slug, "", "")
                    self.run_worker(perform_search(), exclusive=True)
            else:
                self.log.info("MediaWindow.watch_media_active_view: new_view is None, all .media-view-area views remain hidden.")

        except QueryError as e:
            self.log.error(f"MediaWindow.watch_media_active_view: QueryError - {e}", exc_info=True)
        except Exception as e:
            self.log.error(f"MediaWindow.watch_media_active_view: Unexpected error - {e}", exc_info=True)

    def watch_show_deleted_items(self, old_value: bool, new_value: bool) -> None:
        """React to show deleted items checkbox changes."""
        self.log.debug(f"MediaWindow.watch_show_deleted_items: old={old_value}, new={new_value}")
        # Refresh the current view's search when checkbox state changes
        if self.media_active_view:
            type_slug = self.media_active_view.replace("media-view-", "")
            # Skip special windows that don't have standard search functionality
            if type_slug not in ["collections-tags", "multi-item-review"]:
                # Get current search term and keyword filter
                search_term = ""
                keyword_filter = ""
                try:
                    search_input = self.query_one(f"#media-search-input-{type_slug}", Input)
                    search_term = search_input.value
                    keyword_input = self.query_one(f"#media-keyword-filter-{type_slug}", Input)
                    keyword_filter = keyword_input.value
                except QueryError:
                    pass
                async def perform_search():
                    await media_events.perform_media_search_and_display(self.app_instance, type_slug, search_term, keyword_filter)
                self.run_worker(perform_search(), exclusive=True)

    @on(Checkbox.Changed, ".show-deleted-checkbox")
    def handle_show_deleted_checkbox(self, event: Checkbox.Changed) -> None:
        """Handle checkbox changes for showing deleted items."""
        self.show_deleted_items = event.value
        self.log.info(f"Show deleted items checkbox changed to: {event.value}")

    @on(Button.Pressed, "#media-sidebar-toggle-button")
    def handle_sidebar_toggle(self) -> None:
        """Toggles the sidebar's collapsed state."""
        self.media_sidebar_collapsed = not self.media_sidebar_collapsed

    @on(Button.Pressed, ".media-nav-button")
    def handle_nav_button_press(self, event: Button.Pressed) -> None:
        """Handles a click on a media type navigation button."""
        if event.button.id:
            type_slug = event.button.id.replace("media-nav-", "")
            self.media_active_view = f"media-view-{type_slug}" # This triggers the watcher
            # The watcher now also handles setting app.current_media_type_filter_slug and display name
            self.app_instance.media_current_page = 1

    @on(ListView.Selected, ".media-items-list")
    async def handle_list_item_selection(self, event: ListView.Selected) -> None:
        """Calls the event handler to load details when an item is selected."""
        # We delegate to the existing function in media_events.py
        await media_events.handle_media_list_item_selected(self.app_instance, event)

    # --- This method is the key to fixing the crash ---
    def activate_initial_view(self) -> None:
        if not self.media_active_view and self.media_types_from_db:
            initial_slug = slugify("All Media") # Default to "All Media"
            # Ensure "All Media" is actually in media_types_from_db or handle gracefully
            if "All Media" not in self.media_types_from_db:
                 # If "All Media" isn't present, pick the first available type_slug
                if self.media_types_from_db:
                    initial_slug = slugify(self.media_types_from_db[0])
                else: # No media types at all
                    self.log.warning("MediaWindow: No media types available to set an initial view.")
                    return

            self.log.info(f"MediaWindow: Activating initial view for slug '{initial_slug}'.")
            self.media_active_view = f"media-view-{initial_slug}" # Triggers watcher


    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self.log.info(f"MediaWindow on_mount: UI composed with types: {self.media_types_from_db}")
        # Call activate_initial_view here AFTER the window and its children are mounted
        # and queryable, ensuring the watcher can find the views.
        # However, app.py's watch_current_tab is probably a better place if this MediaWindow
        # itself is mounted as part of a tab switch.
        # If MediaWindow is always present, on_mount here is fine for its internal setup.
        # For now, let's rely on app.py's watch_current_tab calling activate_initial_view.
        pass

    def compose(self) -> ComposeResult:
        self.log.debug(f"MediaWindow composing. Initial types from __init__: {self.media_types_from_db}")

        # Left Navigation Pane - wrap in Container for proper layout
        with Container(id="media-nav-container"):
            with VerticalScroll(classes="media-nav-pane", id="media-nav-pane"):
                yield Static("Media Types", classes="sidebar-title")
                if not self.media_types_from_db or (
                        len(self.media_types_from_db) == 1 and self.media_types_from_db[0] in ["Error Loading Types",
                                                                                               "DB Error", "Service Error",
                                                                                               "DB Error or No Media in DB",
                                                                                               "No media types loaded."]):
                    error_message = "No media types loaded."
                    if self.media_types_from_db and isinstance(self.media_types_from_db[0], str):
                        error_message = self.media_types_from_db[0]
                    yield Label(error_message)
                else:
                    for media_type_display_name in self.media_types_from_db:
                        type_slug = slugify(media_type_display_name)
                        yield Button(media_type_display_name, id=f"media-nav-{type_slug}", classes="media-nav-button")

                    # Add Analysis Review button explicitly
                    yield Button("Analysis Review", id="media-nav-analysis-review", classes="media-nav-button")
                    
                    # Add Collections/Tags and Multi-Item Review buttons
                    yield Button("Collections/Tags", id="media-nav-collections-tags", classes="media-nav-button")
                    yield Button("Multi-Item Review", id="media-nav-multi-item-review", classes="media-nav-button")

        # Main Content Pane
        with Container(classes="media-content-pane", id="media-content-pane"):
            yield Button(
                get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE),
                id="media-sidebar-toggle-button",
                classes="sidebar-toggle",
                tooltip="Toggle sidebar"
            )

            # Create views for "All Media" AND each specific media type using a loop
            # This includes "All Media" in the loop if it's in self.media_types_from_db
            all_view_types = self.media_types_from_db + ["analysis-review", "collections-tags", "multi-item-review"]
            # Ensure unique slugs, especially if "Analysis Review" might be in media_types_from_db
            processed_slugs = set()

            for media_type_display_name in all_view_types:
                type_slug = slugify(media_type_display_name)
                if type_slug in processed_slugs: # Avoid duplicate views if "Analysis Review" is already in media_types_from_db
                    continue
                processed_slugs.add(type_slug)

                view_id = f"media-view-{type_slug}"
                
                # Special handling for Collections/Tags and Multi-Item Review
                if type_slug == "collections-tags":
                    # Import the custom window class (we'll create this next)
                    from ..Widgets.collections_tag_window import CollectionsTagWindow
                    yield CollectionsTagWindow(
                        self.app_instance,
                        id=view_id,
                        classes="media-view-area"
                    )
                elif type_slug == "multi-item-review":
                    # Import the custom window class (we'll create this next)
                    from ..Widgets.multi_item_review_window import MultiItemReviewWindow
                    yield MultiItemReviewWindow(
                        self.app_instance,
                        id=view_id,
                        classes="media-view-area"
                    )
                else:
                    # Standard media view layout
                    # Each media view is a Horizontal container for left (list) and right (details) panes
                    with Horizontal(id=view_id, classes="media-view-area"):
                        # --- LEFT PANE (for list and controls) ---
                        with Container(classes="media-content-left-pane"):
                            yield Label(f"{media_type_display_name} Management", classes="pane-title")
                            yield Input(placeholder=f"Search in {media_type_display_name}...",
                                        id=f"media-search-input-{type_slug}",
                                        classes="sidebar-input media-search-input")
                            # Add keyword filter input
                            yield Label("Filter by keywords:", classes="keyword-filter-label")
                            yield Input(placeholder="Enter keywords separated by commas",
                                        id=f"media-keyword-filter-{type_slug}",
                                        classes="keyword-filter-input")
                            # Add checkbox for showing deleted items
                            yield Checkbox("Show deleted items", 
                                          id=f"show-deleted-checkbox-{type_slug}",
                                          classes="show-deleted-checkbox",
                                          value=False)
                            # This ListView is the .media-items-list
                            yield ListView(id=f"media-list-view-{type_slug}", classes="media-items-list")
                            # Pagination bar
                            with Horizontal(classes="media-pagination-bar"):
                                yield Button("Previous", id=f"media-prev-page-button-{type_slug}", disabled=True)
                                yield Label("Page 1 / 1", id=f"media-page-label-{type_slug}", classes="media-page-label")
                                yield Button("Next", id=f"media-next-page-button-{type_slug}", disabled=True)

                        # --- RIGHT PANE (using MediaDetailsWidget for editing capability) ---
                        # This VerticalScroll is the .media-content-right-pane
                        with VerticalScroll(classes="media-content-right-pane"):
                            details_widget = MediaDetailsWidget(
                                self.app_instance,
                                type_slug,
                                id=f"media-details-widget-{type_slug}",
                                classes="media-details-theme"
                            )
                            # Force the widget to expand to fill available space
                            details_widget.styles.height = "1fr"
                            yield details_widget

            # All views are already hidden by default during creation; watcher will manage visibility

    @on(MediaMetadataUpdateEvent)
    async def handle_media_metadata_update(self, event: MediaMetadataUpdateEvent) -> None:
        """Handle media metadata update event."""
        await media_events.handle_media_metadata_update(self.app_instance, event)

    @on(MediaDeleteConfirmationEvent)
    async def handle_media_delete_confirmation(self, event: MediaDeleteConfirmationEvent) -> None:
        """Handle media delete confirmation event."""
        await media_events.handle_media_delete_confirmation(self.app_instance, event)

    @on(MediaUndeleteEvent)
    async def handle_media_undelete(self, event: MediaUndeleteEvent) -> None:
        """Handle media undelete event."""
        await media_events.handle_media_undelete(self.app_instance, event)

#
# End of MediaWindow.py
#######################################################################################################################
