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
            toggle_button.set_class(collapsed, "collapsed") # This answers your pseudo_class question!
        except QueryError as e:
            self.log.warning(f"UI component not found during media sidebar collapse: {e}")

    def watch_media_active_view(self, old_view: Optional[str], new_view: Optional[str]) -> None:
        """Shows/hides the relevant content view when the active view slug changes."""
        if old_view:
            try:
                self.query_one(f"#{old_view}").styles.display = "none"
            except QueryError: pass
        if new_view:
            try:
                view_to_show = self.query_one(f"#{new_view}")
                view_to_show.styles.display = "block"
                # Trigger a search for the new view
                type_slug = new_view.replace("media-view-", "")
                self.app_instance.call_later(media_events.perform_media_search_and_display, self.app_instance, type_slug, "")
            except QueryError:
                self.log.error(f"Could not find new media view to display: #{new_view}")

    # --- EVENT HANDLERS LIVE HERE NOW ---
    @on(Button.Pressed, "#media-sidebar-toggle-button")
    def handle_sidebar_toggle(self) -> None:
        """Toggles the sidebar's collapsed state."""
        self.media_sidebar_collapsed = not self.media_sidebar_collapsed

    @on(Button.Pressed, ".media-nav-button")
    def handle_nav_button_press(self, event: Button.Pressed) -> None:
        """Handles a click on a media type navigation button."""
        if event.button.id:
            type_slug = event.button.id.replace("media-nav-", "")
            self.media_active_view = f"media-view-{type_slug}"
            self.app_instance.current_media_type_filter_slug = type_slug # Ensure app's state is also updated
            nav_button = self.app_instance.query_one(f"#{event.button.id}", Button)
            self.app_instance.current_media_type_filter_display_name = str(nav_button.label)
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
            self.media_active_view = f"media-view-{initial_slug}"
            self.app_instance.current_media_type_filter_slug = initial_slug # Set app state
            # Find the display name for the initial slug
            for name in self.media_types_from_db:
                if slugify(name) == initial_slug:
                    self.app_instance.current_media_type_filter_display_name = name
                    break


    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self.log.info(f"MediaWindow on_mount: UI composed with types: {self.media_types_from_db}")
        # The on_mount is now much simpler and no longer needs to fetch data.
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
                with Horizontal(id=view_id, classes="media-view-area"):
                    # --- LEFT PANE (for list and controls) ---
                    with VerticalScroll(classes="media-content-left-pane"):
                        yield Label(f"{media_type_display_name} Management", classes="pane-title")
                        yield Input(placeholder=f"Search in {media_type_display_name}...",
                                    id=f"media-search-input-{type_slug}",
                                    classes="sidebar-input media-search-input")
                        yield ListView(id=f"media-list-view-{type_slug}", classes="sidebar-listview media-items-list")
                        with Horizontal(classes="media-pagination-bar"): # Common class for pagination
                            yield Button("Previous", id=f"media-prev-page-button-{type_slug}", disabled=True)
                            yield Label("Page 1 / 1", id=f"media-page-label-{type_slug}", classes="media-page-label")
                            yield Button("Next", id=f"media-next-page-button-{type_slug}", disabled=True)

                    # --- RIGHT PANE (standardized to Markdown) ---
                    with VerticalScroll(classes="media-content-right-pane"):
                        yield Markdown(
                            "Select an item from the list to see its details.",
                            id=f"media-details-display-{type_slug}",
                            classes="media-details-theme" # Add a common class for styling
                        )

            # Hide all views by default; watcher will manage visibility
            for view_area in self.query(".media-view-area"):
                view_area.styles.display = "none"

#
# End of MediaWindow.py
#######################################################################################################################
