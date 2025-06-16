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
from textual.widgets import Static, Button, Label, Input, ListView, TextArea, Markdown
#
# Local Imports
from ..Utils.text import slugify
from ..Event_Handlers import media_events
from ..Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE
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
    # --- STATE LIVES HERE NOW ---
    media_sidebar_collapsed: reactive[bool] = reactive(False)
    media_active_view: reactive[Optional[str]] = reactive(None)

    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.media_types_from_db: List[str] = getattr(self.app_instance, '_media_types_for_ui', [])
        self.log.debug(f"MediaWindow __init__: Received media types: {self.media_types_from_db}")

    # --- WATCHERS LIVE HERE NOW ---
    def watch_media_sidebar_collapsed(self, collapsed: bool) -> None:
        """Dynamically adjusts the media browser panes when the sidebar is collapsed or expanded."""
        try:
            nav_pane = self.query_one("#media-nav-pane")
            toggle_button = self.query_one("#media-sidebar-toggle-button")
            nav_pane.set_class(collapsed, "collapsed")
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
                view_to_show.styles.display = "block" # Or "flex" if that's its natural display
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

                # Perform search for the newly activated view
                # Ensure the app's media_current_page is reset for a new view activation if that's desired behavior
                # self.app_instance.media_current_page = 1 # Already done in handle_nav_button_press
                self.app_instance.call_later(media_events.perform_media_search_and_display, self.app_instance, type_slug, "")
            else:
                self.log.info("MediaWindow.watch_media_active_view: new_view is None, all .media-view-area views remain hidden.")

        except QueryError as e:
            self.log.error(f"MediaWindow.watch_media_active_view: QueryError - {e}", exc_info=True)
        except Exception as e:
            self.log.error(f"MediaWindow.watch_media_active_view: Unexpected error - {e}", exc_info=True)


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

        # Left Navigation Pane
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

        # Main Content Pane
        with Container(classes="media-content-pane", id="media-content-pane"):
            yield Button(
                get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE),
                id="media-sidebar-toggle-button",
                classes="sidebar-toggle"
            )

            # Create views for "All Media" AND each specific media type using a loop
            # This includes "All Media" in the loop if it's in self.media_types_from_db
            all_view_types = self.media_types_from_db + ["analysis-review"]
            # Ensure unique slugs, especially if "Analysis Review" might be in media_types_from_db
            processed_slugs = set()

            for media_type_display_name in all_view_types:
                type_slug = slugify(media_type_display_name)
                if type_slug in processed_slugs: # Avoid duplicate views if "Analysis Review" is already in media_types_from_db
                    continue
                processed_slugs.add(type_slug)

                view_id = f"media-view-{type_slug}"
                # Each media view is a Horizontal container for left (list) and right (details) panes
                with Horizontal(id=view_id, classes="media-view-area"):
                    # --- LEFT PANE (for list and controls) ---
                    with Container(classes="media-content-left-pane"):
                        yield Label(f"{media_type_display_name} Management", classes="pane-title")
                        yield Input(placeholder=f"Search in {media_type_display_name}...",
                                    id=f"media-search-input-{type_slug}",
                                    classes="sidebar-input media-search-input")
                        # This ListView is the .media-items-list
                        yield ListView(id=f"media-list-view-{type_slug}", classes="media-items-list")
                        # Pagination bar
                        with Horizontal(classes="media-pagination-bar"):
                            yield Button("Previous", id=f"media-prev-page-button-{type_slug}", disabled=True)
                            yield Label("Page 1 / 1", id=f"media-page-label-{type_slug}", classes="media-page-label")
                            yield Button("Next", id=f"media-next-page-button-{type_slug}", disabled=True)

                    # --- RIGHT PANE (standardized to Markdown) ---
                    # This VerticalScroll is the .media-content-right-pane
                    with VerticalScroll(classes="media-content-right-pane"):
                        yield Markdown(
                            "Select an item from the list to see its details.",
                            id=f"media-details-display-{type_slug}",
                            classes="media-details-theme"
                        )

            # Hide all views by default; watcher will manage visibility
            # This loop is crucial for the initial state.
            for view_area in self.query(".media-view-area"):
                view_area.styles.display = "none"

#
# End of MediaWindow.py
#######################################################################################################################
