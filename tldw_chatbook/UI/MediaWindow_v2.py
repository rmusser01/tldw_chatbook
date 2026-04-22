"""
MediaWindow v2 - Orchestrator for media browsing components.

This is a refactored version that uses the new component-based architecture.
"""

import inspect
from typing import TYPE_CHECKING, List, Optional, Dict, Any
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.css.query import QueryError
from textual.reactive import reactive
from textual.widgets import Button, Markdown, Label
from loguru import logger

# Import media components
from ..Widgets.Media import (
    MediaNavigationPanel,
    MediaSearchPanel,
    MediaSearchEvent,
    MediaBrowseSubviewChangedEvent,
    MediaListPanel,
    MediaItemSelectedEvent,
    MediaViewerPanel
)
from ..Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE

# Import events
from ..Widgets.Media.media_navigation_panel import MediaTypeSelectedEvent
from ..Event_Handlers.media_events import (
    MediaMetadataUpdateEvent,
    MediaDeleteConfirmationEvent,
    MediaUndeleteEvent,
    MediaListCollapseEvent,
    SidebarCollapseEvent,
    MediaReadItLaterToggleEvent,
    MediaAnalysisRequestEvent,
    MediaAnalysisSaveEvent,
    MediaAnalysisSaveAsNoteEvent,
    MediaAnalysisOverwriteEvent,
    MediaAnalysisDeleteEvent
)

if TYPE_CHECKING:
    from ..app import TldwCli


class MediaWindow(Container):
    """
    Orchestrator for the Media Tab components.
    
    Manages communication between navigation, search, list, and viewer panels.
    """
    
    DEFAULT_CSS = """
    MediaWindow {
        layout: horizontal;
        height: 100%;
        width: 100%;
    }
    
    #media-nav-panel {
        width: 20%;
        min-width: 20;
        height: 100%;
    }
    
    #media-nav-panel.collapsed {
        display: none;
    }
    
    #media-main-content {
        layout: vertical;
        width: 1fr;
        height: 100%;
        overflow: hidden;
    }
    
    #media-search-panel {
        height: auto;
    }
    
    #media-content-container {
        layout: horizontal;
        height: 1fr;
        width: 100%;
        overflow: hidden;
    }
    
    #media-list-panel {
        width: 35%;
        height: 100%;
    }
        
    #media-list-panel.collapsed {
        display: none;
    }
    
    #media-viewer-panel {
        width: 1fr;
        height: 100%;
        overflow: hidden;
    }
    
    #media-viewer-panel.hidden {
        display: none;
    }
    
    #media-empty-state {
        width: 1fr;
        height: 100%;
        align: center middle;
        background: $surface;
        color: $text-muted;
    }
    
    #media-empty-state.hidden {
        display: none;
    }
    
    #empty-state-label {
        text-style: italic;
    }
    """
    
    # Reactive properties
    active_media_type: reactive[Optional[str]] = reactive(None)
    selected_media_id: reactive[Optional[str]] = reactive(None)
    media_active_view: reactive[Optional[str]] = reactive(None)
    sidebar_collapsed: reactive[bool] = reactive(False)
    list_collapsed: reactive[bool] = reactive(False)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the MediaWindow."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.runtime_state = getattr(app_instance, "media_runtime_state", None)
        self.media_types = self._get_media_types()
        
    def _get_media_types(self) -> List[str]:
        """Get media types from the app instance."""
        return getattr(self.app_instance, '_media_types_for_ui', [])

    def _scope_service(self):
        """Return the shared media-reading scope service."""
        return getattr(self.app_instance, "media_reading_scope_service", None)

    async def _maybe_await(self, value: Any) -> Any:
        """Support sync or async seam calls."""
        if inspect.isawaitable(value):
            return await value
        return value

    def _runtime_backend(self) -> str:
        """Return the currently active media backend."""
        runtime_state = getattr(self, "runtime_state", None)
        if runtime_state is None:
            runtime_state = getattr(self.app_instance, "media_runtime_state", None)
        if runtime_state is None:
            return "local"
        return str(getattr(runtime_state, "runtime_backend", "local") or "local")

    def _active_browse_subview(self) -> str:
        """Return the current browse subview."""
        runtime_state = getattr(self, "runtime_state", None)
        if runtime_state is None:
            return "all"
        return str(getattr(runtime_state, "active_browse_subview", "all") or "all")

    def _saved_view_available_for_context(self) -> bool:
        """Saved-view browsing is aggregate-only for server mode."""
        if self._runtime_backend() != "server":
            return True
        return (self.active_media_type or "all-media") == "all-media"

    def _sync_saved_view_controls(self) -> None:
        """Keep the search panel's browse-subview controls aligned with runtime state."""
        if not hasattr(self, "search_panel"):
            return
        self.search_panel.set_saved_view_enabled(self._saved_view_available_for_context())
        self.search_panel.set_browse_subview(self._active_browse_subview())

    def _reset_invalid_saved_view_for_context(self) -> bool:
        """Reset invalid saved-view state and notify the user once."""
        if self._active_browse_subview() != "read-it-later":
            return False
        if self._saved_view_available_for_context():
            return False

        if self.runtime_state is not None:
            self.runtime_state.active_browse_subview = "all"
        self._sync_saved_view_controls()
        self.app_instance.notify(
            "Read-it-later is only available in server mode from All Media.",
            severity="warning",
        )
        return True

    def _record_backend(self, record: Optional[Dict[str, Any]] = None) -> str:
        """Resolve backend from a normalized record or current runtime state."""
        if isinstance(record, dict) and record.get("backend"):
            return str(record["backend"])
        return self._runtime_backend()

    def _record_id(self, record: Optional[Dict[str, Any]] = None, fallback: Any = None) -> Optional[str]:
        """Resolve a normalized record ID from record payload or fallback values."""
        if isinstance(record, dict) and record.get("id") not in (None, ""):
            return str(record["id"])
        if fallback in (None, ""):
            return None
        return str(fallback)

    def _source_media_id(self, record: Optional[Dict[str, Any]] = None, fallback: Any = None) -> Any:
        """Resolve the backend-specific media/read-item identifier used by the seam."""
        if isinstance(record, dict):
            source_id = record.get("source_id")
            if source_id not in (None, ""):
                return source_id
            record_id = record.get("id")
            if isinstance(record_id, str) and ":" in record_id:
                return record_id.rsplit(":", 1)[-1]

        if isinstance(fallback, str) and ":" in fallback:
            return fallback.rsplit(":", 1)[-1]
        return fallback

    def _record_for_event(self, event: Any) -> Dict[str, Any]:
        """Resolve the richest available record for a UI event."""
        event_record = getattr(event, "media_data", None)
        if isinstance(event_record, dict):
            return dict(event_record)

        record_id = self._record_id(
            None,
            getattr(event, "record_id", None) or getattr(event, "media_id", None),
        )
        if record_id and self.runtime_state:
            cached = self.runtime_state.detail_by_record_id.get(record_id)
            if cached:
                return dict(cached)

        viewer_record = getattr(self.viewer_panel, "media_data", None)
        if isinstance(viewer_record, dict):
            return dict(viewer_record)

        return {}

    def _show_viewer(self) -> None:
        """Hide the empty state and display the viewer panel."""
        self.query_one("#media-empty-state").add_class("hidden")
        self.viewer_panel.remove_class("hidden")

    def _show_empty_state(self) -> None:
        """Show the empty state and hide the viewer panel."""
        self.query_one("#media-empty-state").remove_class("hidden")
        self.viewer_panel.add_class("hidden")

    def _merge_record_detail(self, record_id: str, updated: Optional[Dict[str, Any]], *, save_for_later: bool) -> Dict[str, Any]:
        """Merge mutation output into the cached detail record without assuming a full payload."""
        existing = {}
        if self.runtime_state is not None:
            existing = dict(self.runtime_state.detail_by_record_id.get(record_id) or {})
        if not existing and isinstance(getattr(self.viewer_panel, "media_data", None), dict):
            existing = dict(self.viewer_panel.media_data or {})

        merged = dict(existing)
        if isinstance(updated, dict):
            merged.update(updated)

        merged["id"] = record_id
        merged.setdefault("backend", self._record_backend(existing))
        merged.setdefault(
            "source_id",
            self._source_media_id(existing, fallback=updated.get("source_id") if isinstance(updated, dict) else None),
        )
        merged["supports_read_it_later"] = bool(merged.get("supports_read_it_later", True))
        merged["is_read_it_later"] = bool(
            merged.get("is_read_it_later", save_for_later)
            if isinstance(updated, dict) and "is_read_it_later" in updated
            else save_for_later
        )
        return merged

    def _clear_selection_for_record(self, record_id: Optional[str]) -> None:
        """Clear selection when the active record should be removed from the viewer."""
        if record_id in (None, ""):
            return

        selected_record_id = getattr(self.runtime_state, "selected_record_id", None) if self.runtime_state is not None else None
        if selected_record_id != record_id:
            return

        self.selected_media_id = None
        if self.runtime_state is not None:
            self.runtime_state.selected_record_id = None
        if hasattr(self.list_panel, "selected_id"):
            self.list_panel.selected_id = None
        self.viewer_panel.clear_display()
        self._show_empty_state()

    def _build_browse_filters(self, type_slug: str, keyword_filter: str, mode: str) -> Dict[str, Any]:
        """Build shared browse filters for media queries."""
        media_types_filter = None
        if type_slug not in ["all-media", "analysis-review"]:
            media_types_filter = [type_slug.replace("-", "_")]

        keywords_list = None
        if keyword_filter:
            keywords_list = [keyword.strip() for keyword in keyword_filter.split(",") if keyword.strip()]

        search_filters: Dict[str, Any] = {
            "sort_by": "last_modified_desc",
            "include_deleted": getattr(self.search_panel, "show_deleted", False),
        }
        if mode == "local":
            search_filters.update(
                {
                    "media_types": media_types_filter,
                    "must_have_keywords": keywords_list,
                    "fields": ["title", "content", "author", "url", "type", "analysis_content"],
                    "include_trash": False,
                }
            )
        return search_filters

    def _normalize_browse_payload(
        self,
        *,
        type_slug: str,
        mode: str,
        payload: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], int]:
        """Normalize browse payload results across search paths."""
        results = list(payload.get("items", []))
        total_matches = int(payload.get("total", len(results)) or 0)

        if mode == "server" and type_slug not in ["all-media", "analysis-review"]:
            expected_media_type = type_slug.replace("-", "_")
            results = [
                item for item in results
                if str(item.get("media_type") or "").strip().lower() == expected_media_type
            ]
            total_matches = len(results)

        return results, total_matches

    async def _execute_browse_query_async(
        self,
        *,
        type_slug: str,
        search_term: str,
        keyword_filter: str,
    ) -> tuple[List[Dict[str, Any]], int]:
        """Execute the active browse query through the current seam."""
        scope_service = self._scope_service()
        if scope_service is None:
            return [], 0

        mode = self._runtime_backend()
        search_filters = self._build_browse_filters(type_slug, keyword_filter, mode)
        offset = max(self.list_panel.current_page - 1, 0) * self.list_panel.items_per_page

        if self._active_browse_subview() == "read-it-later":
            payload = await scope_service.list_read_it_later(
                mode=mode,
                query=search_term if search_term else None,
                limit=self.list_panel.items_per_page,
                offset=offset,
                **search_filters,
            )
        else:
            payload = await scope_service.search_media(
                mode=mode,
                query=search_term if search_term else None,
                limit=self.list_panel.items_per_page,
                offset=offset,
                **search_filters,
            )

        if self.runtime_state is not None:
            self.runtime_state.search_term = search_term
            self.runtime_state.keyword_filter = keyword_filter

        return self._normalize_browse_payload(
            type_slug=type_slug,
            mode=mode,
            payload=payload,
        )

    async def _refresh_current_browse_results_async(self) -> List[Dict[str, Any]]:
        """Refresh current browse results through the active search path."""
        type_slug = self.active_media_type or "all-media"
        search_term = getattr(self.search_panel, "search_term", "")
        keyword_filter = getattr(self.search_panel, "keyword_filter", "")
        results, total_matches = await self._execute_browse_query_async(
            type_slug=type_slug,
            search_term=search_term,
            keyword_filter=keyword_filter,
        )

        total_pages = (total_matches + self.list_panel.items_per_page - 1) // self.list_panel.items_per_page
        self.update_search_results(results, 1 if not results else self.list_panel.current_page, max(total_pages, 1))
        return results

    async def handle_runtime_backend_changed(self, runtime_backend: str) -> None:
        """Reset media state when the active backend changes."""
        if self.runtime_state is not None:
            self.runtime_state.reset_for_backend(runtime_backend)
        self.active_media_type = None
        self.selected_media_id = None
        if hasattr(self.list_panel, "selected_id"):
            self.list_panel.selected_id = None
        if hasattr(self.viewer_panel, "media_data"):
            self.viewer_panel.media_data = None
        self.viewer_panel.clear_display()
        self._show_empty_state()
        self._sync_saved_view_controls()

    async def load_reading_progress(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load and cache reading progress for a normalized media record."""
        record_id = self._record_id(record)
        if record_id is None:
            return None

        if record.get("backing_media_id") in (None, ""):
            return None

        scope_service = self._scope_service()
        if scope_service is None:
            return None

        progress = await self._maybe_await(
            scope_service.get_reading_progress(
                mode=self._record_backend(record),
                record=record,
            )
        )
        if progress is not None and self.runtime_state is not None:
            self.runtime_state.reading_progress_by_record_id[record_id] = progress
        return progress

    async def _load_document_versions(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load analysis/document versions through the scope seam."""
        scope_service = self._scope_service()
        if scope_service is None:
            self.viewer_panel.load_analysis_versions([])
            return []

        try:
            versions = await self._maybe_await(
                scope_service.list_document_versions(
                    mode=self._record_backend(record),
                    media_id=self._source_media_id(record),
                    include_deleted=False,
                )
            )
        except ValueError as exc:
            logger.debug(f"Document versions unavailable for record {record.get('id')}: {exc}")
            versions = []
        except Exception as exc:
            logger.error(f"Failed to load document versions for record {record.get('id')}: {exc}")
            versions = []

        if not isinstance(versions, (list, tuple)):
            versions = []

        self.viewer_panel.load_analysis_versions(list(versions or []))
        return list(versions or [])
    
    def compose(self) -> ComposeResult:
        """Compose the MediaWindow UI."""
        # Navigation panel (docked left, full height)
        self.nav_panel = MediaNavigationPanel(
            self.app_instance,
            self.media_types,
            id="media-nav-panel"
        )
        yield self.nav_panel
        
        # Main content area (everything to the right of navigation)
        with Container(id="media-main-content"):
            # Search panel at top
            self.search_panel = MediaSearchPanel(
                self.app_instance,
                id="media-search-panel"
            )
            yield self.search_panel
            
            # Content container for list and viewer
            with Container(id="media-content-container"):
                # List panel
                self.list_panel = MediaListPanel(
                    self.app_instance,
                    id="media-list-panel"
                )
                yield self.list_panel
                
                # Viewer panel (takes remaining space)
                self.viewer_panel = MediaViewerPanel(
                    self.app_instance,
                    id="media-viewer-panel"
                )
                yield self.viewer_panel
                
                # Empty state placeholder (initially hidden or shown based on selection)
                yield Container(
                    Label("Select a media item to view details", id="empty-state-label"),
                    id="media-empty-state",
                    classes="hidden"
                )
    
    def on_mount(self) -> None:
        """Called when the MediaWindow is mounted."""
        logger.info("MediaWindow v2 mounted")
        
        # Don't activate initial view here - let activate_initial_view handle it
        
        # Check initial size for responsiveness
        self.call_after_refresh(self.check_responsive_layout)

    def on_resize(self, event) -> None:
        """Handle resize events for responsive layout."""
        self.check_responsive_layout()
        
    def check_responsive_layout(self) -> None:
        """Check window size and adjust layout accordingly."""
        if self.size.width < 100 and not self.sidebar_collapsed:
            self.sidebar_collapsed = True
            self.notify("Sidebar collapsed for small screen", severity="information")
        elif self.size.width >= 120 and self.sidebar_collapsed:
            # Optional: auto-expand on very wide screens? 
            # Let's keep it manual expansion to avoid annoyance
            pass
    
    def watch_sidebar_collapsed(self, collapsed: bool) -> None:
        """React to sidebar collapse changes."""
        self.nav_panel.collapsed = collapsed
        # Also add/remove the collapsed class on the nav panel element
        if collapsed:
            self.nav_panel.add_class("collapsed")
        else:
            self.nav_panel.remove_class("collapsed")
    
    def watch_list_collapsed(self, collapsed: bool) -> None:
        """React to list collapse changes."""
        # Toggle display of list panel
        if collapsed:
            self.list_panel.add_class("collapsed")
        else:
            self.list_panel.remove_class("collapsed")
    
    @on(MediaTypeSelectedEvent)
    def handle_media_type_selected(self, event: MediaTypeSelectedEvent) -> None:
        """Handle media type selection from navigation panel."""
        logger.info(f"Media type selected: {event.type_slug}")
        self.activate_media_type(event.type_slug, event.display_name)
    
    @on(MediaSearchEvent)
    def handle_media_search(self, event: MediaSearchEvent) -> None:
        """Handle search event from search panel."""
        logger.info(f"Search triggered: term='{event.search_term}', keywords='{event.keyword_filter}'")
        
        if self.active_media_type:
            # Perform search
            self._perform_search(
                self.active_media_type,
                event.search_term,
                event.keyword_filter
            )

    @on(MediaBrowseSubviewChangedEvent)
    def handle_browse_subview_changed(self, event: MediaBrowseSubviewChangedEvent) -> None:
        """Handle browse-subview selection independently from media-type navigation."""
        if self.runtime_state is not None:
            self.runtime_state.active_browse_subview = str(event.subview or "all")

        if self._reset_invalid_saved_view_for_context() and self.active_media_type:
            self._perform_search(
                self.active_media_type,
                self.search_panel.search_term,
                self.search_panel.keyword_filter,
            )
            return

        self._sync_saved_view_controls()
        if self.active_media_type:
            self._perform_search(
                self.active_media_type,
                self.search_panel.search_term,
                self.search_panel.keyword_filter,
            )
    
    @on(MediaItemSelectedEvent)
    async def handle_media_item_selected(self, event: MediaItemSelectedEvent) -> None:
        """Handle list selection using normalized record IDs and the shared seam."""
        record = self._record_for_event(event)
        record_id = self._record_id(
            record,
            getattr(event, "record_id", None) or getattr(event, "media_id", None),
        )
        if record_id is None:
            logger.warning("Media selection ignored because no record ID was provided")
            return

        record["id"] = record_id
        record.setdefault("backend", self._runtime_backend())

        logger.info(f"Media item selected: {record_id}")
        self.selected_media_id = record_id
        if self.runtime_state is not None:
            self.runtime_state.selected_record_id = record_id

        detail = dict(record)
        scope_service = self._scope_service()
        if scope_service is not None:
            try:
                scoped_detail = await self._maybe_await(
                    scope_service.get_media_detail(
                        mode=self._record_backend(record),
                        media_id=self._source_media_id(
                            record,
                            fallback=getattr(event, "media_id", None),
                        ),
                    )
                )
                if isinstance(scoped_detail, dict):
                    detail = scoped_detail
            except Exception as exc:
                logger.error(f"Failed to load media detail for {record_id}: {exc}")

        detail.setdefault("id", record_id)
        detail.setdefault("backend", record.get("backend", self._runtime_backend()))
        detail.setdefault(
            "source_id",
            self._source_media_id(detail, fallback=self._source_media_id(record, fallback=getattr(event, "media_id", None))),
        )

        if detail.get("reading_progress") is None:
            progress = await self.load_reading_progress(detail)
            if progress is not None:
                detail["reading_progress"] = progress
        elif self.runtime_state is not None:
            self.runtime_state.reading_progress_by_record_id[record_id] = detail["reading_progress"]

        if self.runtime_state is not None:
            self.runtime_state.detail_by_record_id[record_id] = detail
        self.viewer_panel.load_media(detail)
        await self._load_document_versions(detail)
        self._show_viewer()
    
    @on(MediaMetadataUpdateEvent)
    async def handle_metadata_update(self, event: MediaMetadataUpdateEvent) -> None:
        """Handle metadata updates through the scope seam."""
        event.type_slug = self.active_media_type or ""

        record = self._record_for_event(event)
        record_id = self._record_id(
            record,
            getattr(event, "record_id", None) or getattr(event, "media_id", None),
        )
        if record_id is None:
            self.app_instance.notify("Unable to determine media record for update", severity="error")
            return

        record["id"] = record_id
        try:
            await self._scope_service().update_media_metadata(
                mode=self._record_backend(record),
                media_id=self._source_media_id(record, fallback=getattr(event, "media_id", None)),
                title=event.title,
                media_type=event.media_type,
                author=event.author,
                url=event.url,
                keywords=event.keywords,
            )
            updated_record = dict(record)
            updated_record.update(
                {
                    "title": event.title,
                    "media_type": event.media_type,
                    "author": event.author,
                    "url": event.url,
                    "keywords": event.keywords,
                }
            )
            if self.runtime_state is not None:
                self.runtime_state.detail_by_record_id[record_id] = updated_record
            self.viewer_panel.load_media(updated_record)
            await self._load_document_versions(updated_record)
        except Exception as exc:
            logger.error(f"Error updating metadata for {record_id}: {exc}", exc_info=True)
            self.app_instance.notify(f"Error: {str(exc)[:100]}", severity="error")
            return

        if self.active_media_type:
            search_term = self.search_panel.search_term
            keyword_filter = self.search_panel.keyword_filter
            self._perform_search(
                self.active_media_type,
                search_term,
                keyword_filter
            )
    
    @on(MediaDeleteConfirmationEvent)
    def handle_delete_confirmation(self, event: MediaDeleteConfirmationEvent) -> None:
        """Handle delete confirmation from viewer panel."""
        # Run the async confirmation in a worker
        self.run_worker(self._handle_delete_confirmation_async(event))
    
    async def _handle_delete_confirmation_async(self, event: MediaDeleteConfirmationEvent) -> None:
        """Handle delete confirmation asynchronously in a worker."""
        # Show confirmation dialog using our consistent DeleteConfirmationDialog
        from ..Widgets.delete_confirmation_dialog import create_delete_confirmation
        dialog = create_delete_confirmation(
            item_type="Media",
            item_name=event.media_title,
            additional_warning="This will permanently remove the media and all associated data.",
            permanent=True
        )
        
        confirmed = await self.app.push_screen_wait(dialog)
        if confirmed:
            record = self._record_for_event(event)
            record_id = self._record_id(
                record,
                getattr(event, "record_id", None) or getattr(event, "media_id", None),
            )
            try:
                success = await self._scope_service().delete_media(
                    mode=self._record_backend(record),
                    media_id=self._source_media_id(record, fallback=getattr(event, "media_id", None)),
                )
            except Exception as exc:
                logger.error(f"Error deleting media {record_id}: {exc}", exc_info=True)
                self.app_instance.notify(f"Error: {str(exc)[:100]}", severity="error")
                return

            if success:
                self.app_instance.notify(f"'{event.media_title}' has been deleted", severity="information")

                if self.active_media_type:
                    search_term = self.search_panel.search_term
                    keyword_filter = self.search_panel.keyword_filter
                    self._perform_search(
                        self.active_media_type,
                        search_term,
                        keyword_filter
                    )

                if record_id is not None and self.selected_media_id == record_id:
                    self.selected_media_id = None
                    if self.runtime_state is not None:
                        self.runtime_state.selected_record_id = None
                    self.viewer_panel.clear_display()
                    self._show_empty_state()
            else:
                self.app_instance.notify(f"Failed to delete '{event.media_title}'", severity="error")
        else:
            logger.info(f"Media deletion cancelled for: {event.media_title}")
    
    @on(MediaUndeleteEvent)
    async def handle_media_undelete(self, event: MediaUndeleteEvent) -> None:
        """Handle undelete through the shared seam."""
        record = self._record_for_event(event)
        try:
            await self._scope_service().undelete_media(
                mode=self._record_backend(record),
                media_id=self._source_media_id(record, fallback=getattr(event, "media_id", None)),
            )
        except Exception as exc:
            logger.error(f"Error undeleting media {getattr(event, 'record_id', getattr(event, 'media_id', None))}: {exc}", exc_info=True)
            self.app_instance.notify(f"Error: {str(exc)[:100]}", severity="error")
            return

        if self.active_media_type:
            search_term = self.search_panel.search_term
            keyword_filter = self.search_panel.keyword_filter
            self._perform_search(
                self.active_media_type,
                search_term,
                keyword_filter
            )

    @on(MediaReadItLaterToggleEvent)
    def handle_read_it_later_toggle(self, event: MediaReadItLaterToggleEvent) -> None:
        """Handle viewer save/remove actions in a worker."""
        self.run_worker(self._handle_read_it_later_toggle_async(event), exclusive=True)

    async def _handle_read_it_later_toggle_async(self, event: MediaReadItLaterToggleEvent) -> None:
        """Mutate read-it-later state, refresh results, and clear filtered selection."""
        scope_service = self._scope_service()
        if scope_service is None:
            self.app_instance.notify("Media reading scope service is not available.", severity="error")
            return

        record = self._record_for_event(event)
        record_id = self._record_id(
            record,
            getattr(event, "record_id", None) or getattr(event, "media_id", None),
        )
        if record_id is None:
            self.app_instance.notify("Unable to determine media record for save action.", severity="error")
            return

        try:
            if event.save_for_later:
                updated = await self._maybe_await(
                    scope_service.save_to_read_it_later(
                        mode=self._record_backend(record),
                        media_id=self._source_media_id(record, fallback=getattr(event, "media_id", None)),
                    )
                )
            else:
                updated = await self._maybe_await(
                    scope_service.remove_from_read_it_later(
                        mode=self._record_backend(record),
                        media_id=self._source_media_id(record, fallback=getattr(event, "media_id", None)),
                    )
                )
        except Exception as exc:
            logger.error(f"Error toggling read-it-later for {record_id}: {exc}", exc_info=True)
            self.app_instance.notify(f"Error: {str(exc)[:100]}", severity="error")
            return

        merged = self._merge_record_detail(record_id, updated if isinstance(updated, dict) else None, save_for_later=event.save_for_later)
        if self.runtime_state is not None:
            self.runtime_state.detail_by_record_id[record_id] = merged
        if getattr(self.runtime_state, "selected_record_id", None) == record_id:
            self.viewer_panel.load_media(merged)

        try:
            await self._refresh_current_browse_results_async()
        except Exception as exc:
            logger.error(f"Error refreshing browse results after read-it-later toggle for {record_id}: {exc}", exc_info=True)
            self.app_instance.notify(f"Error loading media: {str(exc)[:100]}", severity="error")
            return

        if self._active_browse_subview() == "read-it-later" and not event.save_for_later:
            self._clear_selection_for_record(record_id)
    
    @on(MediaListCollapseEvent)
    def handle_list_collapse(self) -> None:
        """Handle list collapse toggle."""
        self.list_collapsed = not self.list_collapsed
        # Update button text
        try:
            button = self.viewer_panel.query_one("#collapse-media-list", Button)
            button.label = "▶" if self.list_collapsed else "◀"
        except Exception:
            pass
    
    @on(SidebarCollapseEvent)
    def handle_sidebar_collapse(self) -> None:
        """Handle sidebar collapse toggle."""
        self.sidebar_collapsed = not self.sidebar_collapsed
        # Update button text
        try:
            button = self.search_panel.query_one("#media-sidebar-toggle", Button)
            from ..Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE
            # Toggle between collapsed and expanded state
            if self.sidebar_collapsed:
                button.label = "▶"  # Arrow pointing right when collapsed
            else:
                button.label = get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE)
        except Exception:
            pass
    
    @on(MediaAnalysisRequestEvent)
    def handle_analysis_request(self, event: MediaAnalysisRequestEvent) -> None:
        """Handle media analysis request."""
        # Set the type_slug
        event.type_slug = self.active_media_type or ""
        
        async def perform_analysis():
            try:
                logger.info(f"Starting media analysis for media_id={event.media_id}")
                logger.debug(f"Analysis parameters: provider={event.provider}, model={event.model}, "
                           f"temperature={event.temperature}, top_p={event.top_p}, min_p={event.min_p}, "
                           f"max_tokens={event.max_tokens}")
                
                record = self._record_for_event(event)
                media_data = dict(record) if record else None

                if not media_data and self.app_instance.media_db and event.media_id not in (None, ""):
                    try:
                        media_data = self.app_instance.media_db.get_media_by_id(event.media_id)
                    except Exception as exc:
                        logger.debug(f"Local media lookup failed for analysis request {event.media_id}: {exc}")

                if not media_data:
                    logger.error(f"Media item not found for id={event.media_id}")
                    self.app_instance.notify("Media item not found", severity="error")
                    return
                
                logger.debug(f"Found media item: title='{media_data.get('title', 'Unknown')}', "
                           f"type={media_data.get('type', 'Unknown')}")
                
                # Use the same chat_wrapper as the chat window
                logger.info(f"Calling {event.provider} with model {event.model} for analysis")
                
                # Prepare the chat history format expected by chat_wrapper
                chat_history = []
                
                # Get API key using the same method as chat
                api_key_for_call = None
                provider_settings_key = event.provider.lower().replace(" ", "_")
                logger.debug(f"Looking for provider settings under key: {provider_settings_key}")
                
                provider_config_settings = self.app_instance.app_config.get("api_settings", {}).get(provider_settings_key, {})
                logger.debug(f"Provider config settings found: {bool(provider_config_settings)}")
                
                # First check direct api_key field
                if "api_key" in provider_config_settings:
                    config_api_key = provider_config_settings.get("api_key", "").strip()
                    if config_api_key and config_api_key != "<API_KEY_HERE>":
                        api_key_for_call = config_api_key
                        logger.debug(f"Using API key for '{event.provider}' from config file field.")
                    else:
                        logger.debug(f"API key field exists but is empty or placeholder")
                
                # If not found, check environment variable
                if not api_key_for_call:
                    env_var_name = provider_config_settings.get("api_key_env_var", "").strip()
                    logger.debug(f"Checking environment variable: {env_var_name}")
                    if env_var_name:
                        import os
                        env_api_key = os.environ.get(env_var_name, "").strip()
                        if env_api_key:
                            api_key_for_call = env_api_key
                            logger.debug(f"Using API key for '{event.provider}' from ENV var '{env_var_name}'.")
                        else:
                            logger.debug(f"Environment variable '{env_var_name}' not found or empty")
                
                # Check if key is required
                providers_requiring_key = ["openai", "anthropic", "google", "mistralai", "groq", "cohere", "openrouter", "huggingface", "deepseek"]
                if event.provider.lower() in providers_requiring_key and not api_key_for_call:
                    logger.error(f"API key required for provider '{event.provider}' but not found")
                    self.app_instance.notify(f"API Key for {event.provider} is missing.", severity="error")
                    return
                
                # Use chat_wrapper with the same parameters as the chat window
                logger.debug(f"Building LLM call with system_prompt length={len(event.system_prompt or '')}, "
                           f"user_prompt length={len(event.user_prompt or '')}")
                
                # Build the actual media content to send
                media_content_for_llm = {}
                content_text = media_data.get('content', '')
                if content_text:
                    # Format the content similar to how it's done in chat
                    media_content_for_llm = {
                        'title': media_data.get('title', 'Untitled'),
                        'author': media_data.get('author', 'Unknown'),
                        'content': content_text,
                        'type': media_data.get('type', 'Unknown'),
                        'url': media_data.get('url', '')
                    }
                    logger.debug(f"Prepared media content: title='{media_content_for_llm['title']}', "
                               f"content_length={len(content_text)}")
                else:
                    logger.warning("No content found in media item")
                
                def call_llm():
                    try:
                        logger.info("Making LLM API call...")
                        
                        # Combine the user prompt with media content if no content placeholder was used
                        final_user_prompt = event.user_prompt or ""
                        if media_content_for_llm and '{content}' not in (event.user_prompt or ''):
                            # If user didn't use {content} placeholder, append the content
                            logger.info("No {content} placeholder found in prompt, appending media content")
                            content_text = media_content_for_llm.get('content', '')
                            if content_text:
                                final_user_prompt = f"{event.user_prompt}\n\n---\n\nContent to analyze:\n\nTitle: {media_content_for_llm.get('title', 'Untitled')}\nAuthor: {media_content_for_llm.get('author', 'Unknown')}\nType: {media_content_for_llm.get('type', 'Unknown')}\n\n{content_text}"
                                logger.debug(f"Combined prompt length: {len(final_user_prompt)} chars")
                        
                        result = self.app_instance.chat_wrapper(
                            message=final_user_prompt,
                            history=chat_history,
                            media_content={},  # Empty since we're including content in the message
                            api_endpoint=event.provider,
                            api_key=api_key_for_call,
                            custom_prompt="",
                            temperature=event.temperature,
                            system_message=event.system_prompt or "",
                            streaming=False,  # No streaming for analysis
                            minp=event.min_p,
                            model=event.model,
                            topp=event.top_p,
                            topk=50,
                            llm_max_tokens=event.max_tokens,
                            llm_seed=None,
                            llm_stop=None,
                            llm_response_format=None,
                            llm_n=1,
                            llm_user_identifier=None,
                            llm_logprobs=False,
                            llm_top_logprobs=None,
                            llm_logit_bias=None,
                            llm_presence_penalty=0,
                            llm_frequency_penalty=0,
                            llm_tools=None,
                            llm_tool_choice=None,
                            llm_fixed_tokens_kobold=None,
                            current_image_input={},
                            selected_parts=[],
                            chatdict_entries=None,
                            max_tokens=500,
                            strategy="sorted_evenly",
                            strip_thinking_tags=False
                        )
                        logger.info(f"LLM API call completed, response type: {type(result)}, "
                                   f"response length: {len(str(result)) if result else 0}")
                        return result
                    except Exception as e:
                        logger.error(f"Error in LLM API call: {e}", exc_info=True)
                        raise
                
                # Run in thread since it's a sync function
                import asyncio
                logger.debug("Running LLM call in thread...")
                response = await asyncio.to_thread(call_llm)
                
                logger.info(f"Got response from LLM: type={type(response)}, "
                           f"is_string={isinstance(response, str)}, "
                           f"is_dict={isinstance(response, dict)}")
                
                # Extract the actual message content from the response
                response_text = None
                if isinstance(response, str):
                    response_text = response
                elif isinstance(response, dict):
                    # Handle OpenAI-style response format
                    if 'choices' in response and len(response['choices']) > 0:
                        choice = response['choices'][0]
                        if 'message' in choice and 'content' in choice['message']:
                            response_text = choice['message']['content']
                            logger.debug(f"Extracted message content from dict response: {len(response_text)} chars")
                        elif 'text' in choice:
                            response_text = choice['text']
                            logger.debug(f"Extracted text from dict response: {len(response_text)} chars")
                    # Handle direct content response
                    elif 'content' in response:
                        response_text = response['content']
                        logger.debug(f"Extracted content from dict response: {len(response_text)} chars")
                
                if response_text:
                    # Update display directly
                    try:
                        logger.debug("Updating analysis display with response")
                        analysis_display = self.viewer_panel.query_one("#analysis-display", Markdown)
                        await analysis_display.update(response_text)
                        # Update the viewer panel's current analysis state
                        self.viewer_panel.current_analysis = response_text
                        self.viewer_panel._update_analysis_button_states()
                        
                        # Add the new analysis as a temporary unsaved entry
                        if not any(a.get('analysis_content') == response_text for a in self.viewer_panel.all_analyses):
                            self.viewer_panel.all_analyses.insert(0, {
                                'version_number': 'unsaved',
                                'analysis_content': response_text,
                                'created_at': 'Just now (unsaved)',
                            })
                            self.viewer_panel.current_analysis_index = 0
                            self.viewer_panel._update_analysis_navigation()
                        
                        logger.info("Analysis display updated successfully")
                    except Exception as e:
                        logger.error(f"Error updating analysis display: {e}", exc_info=True)
                    
                    self.app_instance.notify("Analysis generated successfully", severity="information")
                else:
                    logger.error(f"Could not extract text from LLM response: response={response}")
                    self.app_instance.notify("Failed to generate analysis", severity="error")
                    # Reset analysis display on failure
                    try:
                        analysis_display = self.viewer_panel.query_one("#analysis-display", Markdown)
                        await analysis_display.update("*Analysis generation failed - no valid response text*")
                    except Exception as e:
                        logger.error(f"Error resetting analysis display: {e}")
                    
            except Exception as e:
                logger.error(f"Error performing analysis: {e}", exc_info=True)
                self.app_instance.notify(f"Error: {str(e)[:100]}", severity="error")
        
        # Run the analysis in a worker
        self.run_worker(perform_analysis(), exclusive=True)
    
    @on(MediaAnalysisSaveEvent)
    def handle_analysis_save(self, event: MediaAnalysisSaveEvent) -> None:
        """Handle saving new analysis."""
        event.type_slug = self.active_media_type or ""
        self.run_worker(self._handle_analysis_save_async(event), exclusive=True)

    async def _handle_analysis_save_async(self, event: MediaAnalysisSaveEvent) -> None:
        """Persist a new analysis version via the shared seam."""
        record = self._record_for_event(event)
        record_id = self._record_id(
            record,
            getattr(event, "record_id", None) or getattr(event, "media_id", None),
        )
        if record_id is None:
            self.app_instance.notify("Media item not found", severity="error")
            return

        record["id"] = record_id
        try:
            version_info = await self._scope_service().save_analysis_version(
                mode=self._record_backend(record),
                media_id=self._source_media_id(record, fallback=getattr(event, "media_id", None)),
                content=record.get("content", ""),
                analysis_content=event.analysis_content,
            )
        except ValueError as exc:
            self.app_instance.notify(str(exc), severity="warning")
            return
        except Exception as exc:
            logger.error(f"Error saving analysis for {record_id}: {exc}", exc_info=True)
            self.app_instance.notify(f"Error: {str(exc)[:100]}", severity="error")
            return

        if version_info:
            self.app_instance.notify("Analysis saved successfully", severity="information")
            self.viewer_panel.has_existing_analysis = True
            self.viewer_panel.current_analysis = event.analysis_content
            self.viewer_panel._update_analysis_button_states()
            await self._load_document_versions(record)
        else:
            self.app_instance.notify("Failed to save analysis", severity="error")
    
    @on(MediaAnalysisSaveAsNoteEvent)
    def handle_analysis_save_as_note(self, event: MediaAnalysisSaveAsNoteEvent) -> None:
        """Handle saving analysis as a new note."""
        try:
            if not self.app_instance.notes_db:
                self.app_instance.notify("Notes database not available", severity="error")
                return
            
            # Generate a title for the note
            note_title = f"Analysis: {event.media_title}"
            
            # Create the note content with metadata
            note_content = f"# {note_title}\n\n"
            note_content += f"*Generated from media: {event.media_title} (ID: {event.media_id})*\n\n"
            note_content += "---\n\n"
            note_content += event.analysis_content
            
            # Create the note
            note_id = self.app_instance.notes_db.create_note(
                title=note_title,
                content=note_content,
                tags=["media-analysis", f"media-id-{event.media_id}"]
            )
            
            if note_id:
                self.app_instance.notify(
                    f"Analysis saved as note: {note_title}",
                    severity="information"
                )
            else:
                self.app_instance.notify("Failed to save analysis as note", severity="error")
                
        except Exception as e:
            logger.error(f"Error saving analysis as note: {e}", exc_info=True)
            self.app_instance.notify(f"Error: {str(e)[:100]}", severity="error")
    
    @on(MediaAnalysisOverwriteEvent)
    def handle_analysis_overwrite(self, event: MediaAnalysisOverwriteEvent) -> None:
        """Handle overwriting existing analysis."""
        event.type_slug = self.active_media_type or ""
        self.run_worker(self._handle_analysis_overwrite_async(event), exclusive=True)

    async def _handle_analysis_overwrite_async(self, event: MediaAnalysisOverwriteEvent) -> None:
        """Persist an overwrite analysis version via the shared seam."""
        record = self._record_for_event(event)
        record_id = self._record_id(
            record,
            getattr(event, "record_id", None) or getattr(event, "media_id", None),
        )
        if record_id is None:
            self.app_instance.notify("Media item not found", severity="error")
            return

        record["id"] = record_id
        try:
            version_info = await self._scope_service().overwrite_analysis_version(
                mode=self._record_backend(record),
                media_id=self._source_media_id(record, fallback=getattr(event, "media_id", None)),
                content=record.get("content", ""),
                analysis_content=event.analysis_content,
            )
        except ValueError as exc:
            self.app_instance.notify(str(exc), severity="warning")
            return
        except Exception as exc:
            logger.error(f"Error overwriting analysis for {record_id}: {exc}", exc_info=True)
            self.app_instance.notify(f"Error: {str(exc)[:100]}", severity="error")
            return

        if version_info:
            self.app_instance.notify("Analysis overwritten successfully", severity="information")
            self.viewer_panel.has_existing_analysis = True
            self.viewer_panel.current_analysis = event.analysis_content
            self.viewer_panel._update_analysis_button_states()
            await self._load_document_versions(record)
        else:
            self.app_instance.notify("Failed to overwrite analysis", severity="error")
    
    @on(MediaAnalysisDeleteEvent)
    def handle_analysis_delete(self, event: MediaAnalysisDeleteEvent) -> None:
        """Handle deleting an analysis version."""
        event.type_slug = self.active_media_type or ""
        self.run_worker(self._handle_analysis_delete_async(event), exclusive=True)

    async def _handle_analysis_delete_async(self, event: MediaAnalysisDeleteEvent) -> None:
        """Delete an analysis version through the shared seam."""
        if not event.version_uuid:
            logger.info("No UUID provided, this appears to be a legacy analysis")
            self.app_instance.notify("Cannot delete legacy analysis from database", severity="warning")
            return

        record = self._record_for_event(event)
        try:
            success = await self._scope_service().delete_analysis_version(
                mode=self._record_backend(record),
                version_uuid=event.version_uuid,
            )
        except ValueError as exc:
            self.app_instance.notify(str(exc), severity="warning")
            return
        except Exception as exc:
            logger.error(f"Error deleting analysis {event.version_uuid}: {exc}", exc_info=True)
            self.app_instance.notify(f"Error: {str(exc)[:100]}", severity="error")
            return

        if success:
            self.app_instance.notify("Analysis deleted successfully", severity="information")
            await self._load_document_versions(record)
        else:
            logger.info("Could not delete document version, might be the last one")
            self.app_instance.notify("This is the last analysis version", severity="warning")
    
    def activate_media_type(self, type_slug: str, display_name: str) -> None:
        """Activate a media type and perform initial search."""
        logger.info(f"activate_media_type called: type_slug='{type_slug}', display_name='{display_name}'")
        self.active_media_type = type_slug
        if self.runtime_state is not None:
            self.runtime_state.active_media_type = type_slug
            self.runtime_state.search_term = ""
            self.runtime_state.keyword_filter = ""
            self.runtime_state.selected_record_id = None

        self._reset_invalid_saved_view_for_context()
        
        # Update navigation panel
        self.nav_panel.selected_type = type_slug
        
        # Update search panel
        self.search_panel.set_type_filter(type_slug, display_name)
        self._sync_saved_view_controls()
        
        # Clear viewer
        self.viewer_panel.clear_display()
        
        # Show empty state
        try:
            self._show_empty_state()
        except Exception:
            pass
        
        # Reset page to 1 when switching types
        self.list_panel.current_page = 1
        
        # Perform search
        logger.info(f"About to call _perform_search for type '{type_slug}'")
        self._perform_search(type_slug, "", "")
    
    def activate_initial_view(self) -> None:
        """Activate the initial view - called by app.py."""
        if not self.active_media_type and self.media_types:
            # Try "All Media" first
            if "All Media" in self.media_types:
                self.activate_media_type("all-media", "All Media")
            elif self.media_types:
                # Fall back to first available type
                from ..Utils.text import slugify
                first_type = self.media_types[0]
                self.activate_media_type(slugify(first_type), first_type)
    
    def update_search_results(self, results: List[Dict[str, Any]], page: int, total_pages: int) -> None:
        """Update search results in the list panel."""
        try:
            logger.info(f"Updating search results: {len(results)} items, page {page}/{total_pages}")
            if self.runtime_state is not None:
                self.runtime_state.browse_items = list(results)
            self.list_panel.load_items(results, page, total_pages)
        except Exception as e:
            logger.error(f"Error updating search results: {e}", exc_info=True)
    
    def watch_media_active_view(self, old_view: Optional[str], new_view: Optional[str]) -> None:
        """React to media_active_view changes from app.py button handlers."""
        if new_view and new_view.startswith("media-view-"):
            type_slug = new_view.replace("media-view-", "")
            # Find display name from media types
            display_name = type_slug.replace("-", " ").title()
            # Special case handling for known types
            if type_slug == "all-media":
                display_name = "All Media"
            elif type_slug == "analysis-review":
                display_name = "Analysis Review"
            elif type_slug == "collections-tags":
                display_name = "Collections/Tags"
            elif type_slug == "multi-item-review":
                display_name = "Multi-Item Review"
            
            self.activate_media_type(type_slug, display_name)
    
    def _perform_search(self, type_slug: str, search_term: str, keyword_filter: str) -> None:
        """Trigger media search in background."""
        logger.info(f"_perform_search called: type_slug='{type_slug}', search_term='{search_term}', keyword_filter='{keyword_filter}'")
        
        # Use run_worker with an async coroutine
        async def perform_search():
            logger.info(f"perform_search coroutine executing for type '{type_slug}'")
            try:
                scope_service = self._scope_service()
                if scope_service is None:
                    logger.error("Media reading scope service not available")
                    return
                
                # Skip search for special windows
                if type_slug in ["collections-tags", "multi-item-review"]:
                    logger.info(f"Skipping search for special window: {type_slug}")
                    return
                
                # Set loading state
                self.list_panel.set_loading(True)
                if self._reset_invalid_saved_view_for_context():
                    self._sync_saved_view_controls()

                results, total_matches = await self._execute_browse_query_async(
                    type_slug=type_slug,
                    search_term=search_term,
                    keyword_filter=keyword_filter,
                )
                
                logger.info(f"Search returned {len(results)} results, total matches: {total_matches}")
                
                if results:
                    # Calculate total pages
                    total_pages = (total_matches + self.list_panel.items_per_page - 1) // self.list_panel.items_per_page
                    total_pages = max(total_pages, 1)
                    
                    # Update the list panel
                    self.update_search_results(
                        results,
                        self.list_panel.current_page,
                        total_pages
                    )
                    
                    logger.info(f"Found {len(results)} items for type '{type_slug}' (page {self.list_panel.current_page}/{total_pages})")
                else:
                    # No results
                    self.update_search_results([], 1, 1)
                    logger.info(f"No media items found for type '{type_slug}'")
                
                # Always set loading false when done
                self.list_panel.set_loading(False)
                    
            except Exception as e:
                logger.error(f"Error during media search: {e}", exc_info=True)
                self.list_panel.set_loading(False)
                # Notify user of error
                self.app_instance.notify(f"Error loading media: {str(e)[:100]}", severity="error")
        
        # Run the worker
        self.run_worker(perform_search(), exclusive=True)

#
# End of MediaWindow_v2.py
##############################A###############A###############A###############A###############A###############A
