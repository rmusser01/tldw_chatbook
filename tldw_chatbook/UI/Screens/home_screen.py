"""Home dashboard screen for the master shell."""

import asyncio
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_session_settings import (
    build_console_settings_readiness,
    build_default_console_session_settings,
)
from tldw_chatbook.config import get_cli_setting, load_settings, save_setting_to_cli_config
from tldw_chatbook.Constants import LIBRARY_NAV_CONTEXT_NOTE_ID, TAB_CHAT, TAB_LIBRARY
from tldw_chatbook.Home.dashboard_state import (
    HOME_PRIMARY_ACTION_ID,
    HOME_RESUME_KIND_CONVERSATION,
    HOME_RESUME_KIND_NOTE,
    HOME_RESUME_LATEST_CONTROL_ID,
    HOME_START_CONVERSATION_CONTROL_ID,
    HomeContentSnapshot,
    HomeControl,
    HomeDashboard,
    HomeDashboardInput,
    HomeTriageState,
    apply_home_content_snapshot,
    build_home_triage_state,
    choose_home_selected_item,
    summarize_home_dashboard,
)
from tldw_chatbook.Home.home_rail_state import (
    HOME_RAIL_SECTION_IDS,
    HomeRailPreferences,
    coerce_home_rail_preferences,
    serialize_home_rail_preferences,
)
from tldw_chatbook.Widgets.Console.console_rail_section import (
    CONSOLE_RAIL_SECTION_TOGGLE_PREFIX,
    ConsoleRailSectionHeader,
)
from tldw_chatbook.Widgets.Home.home_canvas import HomeCanvas
from tldw_chatbook.Widgets.Home.home_rail import HOME_RAIL_ROW_PREFIX, HomeRail

from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen
from .settings_config_models import SettingsCategoryId


HOME_CONTROL_METHODS = {
    "home-approve": "approve_active_home_item",
    "home-reject": "reject_active_home_item",
    "home-pause": "pause_active_home_item",
    "home-resume": "resume_active_home_item",
    "home-retry": "retry_active_home_item",
    "home-open-details": "open_active_home_item_details",
    "home-open-in-console": "open_active_home_item_in_console",
    "home-open-chatbook-details": "open_active_home_item_details",
    "home-open-chatbook-in-console": "open_active_home_item_in_console",
    "home-review-flashcards": "open_home_flashcards_review",
}

HOME_CONTROL_METHODS_WITH_TARGET_ROUTE = {
    "home-open-details",
    "home-open-in-console",
    "home-open-chatbook-details",
    "home-open-chatbook-in-console",
}


def _home_runtime_status_label(state: HomeDashboardInput) -> str:
    source = str(state.runtime_source or "local").strip().lower()
    if source != "server":
        return "Local"
    server_label = str(state.server_label or "").strip()
    return f"Server: {server_label}" if server_label else "Server"


def _home_primary_action_context(action: object) -> dict[str, object]:
    if getattr(action, "action_id", None) == "fix_model_setup":
        return {"category": SettingsCategoryId.PROVIDERS_MODELS.value}
    return {}


# T190: max characters of a raw resume-candidate title kept on the resume
# control's Button label (truncated BEFORE markup-escaping so the escape
# backslashes can never be cut mid-sequence).
_HOME_RESUME_TITLE_MAX_CHARS = 40

# Sections load_settings() always injects into a disk-loaded config but which
# test fakes never carry -- the exact marker check Console's readiness
# staleness fix uses (chat_screen._CONSOLE_LIVE_CONFIG_MARKER_SECTIONS,
# task-177). A real boot snapshot is safe to refresh from disk; an injected
# hermetic test config must be honored verbatim.
_HOME_LIVE_CONFIG_MARKER_SECTIONS = ("general", "logging")


async def _await_home_seam_result(awaitable: Any) -> Any:
    """Await a seam's awaitable inside the worker thread's private loop."""
    return await awaitable


def _home_response_total(result: Any) -> int | None:
    """Extract a total count from a scope-service list response.

    Mirrors the shapes Library's rail counts consume: the local
    conversation service returns ``{"items", "pagination": {"total"}}``,
    the local media service ``{"items", "pagination": {"total_items"}}``.

    Args:
        result: The raw service response.

    Returns:
        The non-negative total, or ``None`` when the response carries no
        authoritative total (never falls back to a page-sample length --
        Home fetches limit-1 pages, so a sample count would be a lie).
    """
    if not isinstance(result, Mapping):
        return None
    pagination = result.get("pagination")
    candidates = []
    if isinstance(pagination, Mapping):
        candidates.extend((pagination.get("total"), pagination.get("total_items")))
    candidates.append(result.get("total"))
    for candidate in candidates:
        if isinstance(candidate, bool):
            continue
        try:
            if candidate is not None:
                return max(int(candidate), 0)
        except (TypeError, ValueError):
            continue
    return None


def _home_first_record(result: Any) -> Mapping[str, Any] | None:
    """Return the first record of a list response (list or items-mapping)."""
    if isinstance(result, Mapping):
        result = result.get("items")
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
        first = next(iter(result), None)
        return first if isinstance(first, Mapping) else None
    return None


def _home_record_timestamp(record: Mapping[str, Any] | None) -> datetime:
    """Parse a record's freshest timestamp for newest-wins comparison."""
    fallback = datetime.min.replace(tzinfo=timezone.utc)
    if not isinstance(record, Mapping):
        return fallback
    for key in ("last_modified", "updated_at", "created_at"):
        raw = record.get(key)
        if raw in (None, ""):
            continue
        if isinstance(raw, datetime):
            # DB layers may hand back datetimes directly; no str round-trip
            # (PR #608 review).
            return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
        text = str(raw).strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return fallback


def _home_resume_fields(
    latest_note: Mapping[str, Any] | None,
    latest_conversation: Mapping[str, Any] | None,
) -> tuple[str, str, str]:
    """Pick the newest resume candidate (note vs conversation).

    Args:
        latest_note: The newest local note record, if any.
        latest_conversation: The newest local conversation record, if any.

    Returns:
        ``(resume_kind, resume_id, resume_title)`` -- all empty when
        neither candidate has a usable id. Ties go to the conversation
        (Console resume is the tighter loop). The title stays RAW here;
        it is escaped once in ``build_home_resume_control``.
    """
    candidates = []
    if isinstance(latest_conversation, Mapping) and latest_conversation.get("id") not in (None, ""):
        candidates.append((HOME_RESUME_KIND_CONVERSATION, latest_conversation))
    if isinstance(latest_note, Mapping) and latest_note.get("id") not in (None, ""):
        candidates.append((HOME_RESUME_KIND_NOTE, latest_note))
    if not candidates:
        return "", "", ""
    kind, record = max(candidates, key=lambda pair: _home_record_timestamp(pair[1]))
    title = " ".join(str(record.get("title") or "").split())
    if len(title) > _HOME_RESUME_TITLE_MAX_CHARS:
        title = title[: _HOME_RESUME_TITLE_MAX_CHARS - 1].rstrip() + "…"
    return kind, str(record.get("id")), title


class HomeActionButton(Button):
    """Home button that emits press events even when app chrome hides layout."""

    def __init__(self, *args, fallback_press: Callable[[], None] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._fallback_press = fallback_press

    def press(self):
        if not self.display and self._fallback_press is not None:
            self._fallback_press()
            return self
        return super().press()


class HomeScreen(BaseAppScreen):
    """Dashboard, notifications, readiness, and next-best action surface."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "home", **kwargs)
        self._current_dashboard: HomeDashboard | None = None
        self._current_dashboard_input: HomeDashboardInput | None = None
        self._home_selected_row_id: str = ""
        # T152: the scoped canvas controls the user actually sees/presses
        # (``triage.canvas.actions``) -- kept alongside ``_current_dashboard``
        # (the UNSCOPED ``summarize_home_dashboard`` controls) because only
        # the scoped set carries a selection-aware ``target_id`` (e.g.
        # ``home-retry`` pointed at the SELECTED failed item rather than
        # just the first failed item in the list). See
        # ``_activate_home_control``.
        self._current_canvas_controls: tuple[HomeControl, ...] = ()
        # T190: cached content/readiness snapshot, refreshed by the
        # ``_refresh_home_content_snapshot`` worker off the compose path and
        # merged into every ``_build_dashboard_input``. Per-instance --
        # screens compose fresh on navigation, so every Home visit re-reads
        # the real seams.
        self._home_content_snapshot: HomeContentSnapshot | None = None

    def on_mount(self) -> None:
        super().on_mount()
        self._refresh_home_chatbook_artifact_snapshot()
        self._refresh_home_content_snapshot()
        self._refresh_home_active_work_cache()

    @work(exclusive=True, thread=True)
    def _refresh_home_chatbook_artifact_snapshot(self) -> None:
        adapter = getattr(self.app_instance, "home_active_work_adapter", None)
        refresh_flashcards_due = getattr(adapter, "refresh_flashcards_due_snapshot", None)
        if callable(refresh_flashcards_due):
            chachanotes_db = getattr(self.app_instance, "chachanotes_db", None)
            if getattr(chachanotes_db, "is_memory_db", False):
                # SQLite ``:memory:`` connections are thread-local -- the
                # flashcards-due provider ultimately queries ChaChaNotes
                # directly, and only the thread that created the DB has the
                # migrated schema. Running the refresh on THIS worker thread
                # would open a brand-new, unmigrated in-memory connection, so
                # hop back onto the UI thread for the in-memory case.
                # File-backed DBs keep the off-thread call -- that's the
                # whole point of this worker.
                self.app.call_from_thread(refresh_flashcards_due)
            else:
                refresh_flashcards_due()
        refresh_snapshot = getattr(adapter, "refresh_chatbook_artifact_snapshot", None)
        if not callable(refresh_snapshot):
            return
        refresh_snapshot()
        self.app.call_from_thread(self._refresh_after_chatbook_artifact_snapshot)

    def _refresh_after_chatbook_artifact_snapshot(self) -> None:
        if self.is_mounted:
            self.refresh(recompose=True)

    @work(exclusive=True, group="home-content-snapshot")
    async def _refresh_home_content_snapshot(self) -> None:
        """Refresh readiness + real content counts/recents off the compose path.

        T190: sources Console readiness from FRESH config (the same seams
        Console uses) and content counts/most-recent items from the same
        scope-service seams the Library rail uses, then syncs the triage
        surface in place. Runs in its own worker group so it never cancels
        (or is cancelled by) the chatbook-artifact snapshot worker.
        """
        snapshot = await self._build_home_content_snapshot()
        previous = self._home_content_snapshot
        self._home_content_snapshot = snapshot
        # Only re-sync when the snapshot actually changes what Home shows --
        # an all-default snapshot (nothing ready, nothing counted) must not
        # trigger a redundant adapter rebuild/refresh cycle.
        if self.is_mounted and snapshot != (previous or HomeContentSnapshot()):
            self._sync_home_triage()

    @work(exclusive=True, group="home-active-work-cache")
    async def _refresh_home_active_work_cache(self) -> None:
        """Warm the active-work adapter's TTL cache off the event loop.

        B3 (task-282): ``home_active_work_adapter.build_dashboard_input``
        used to run its watchlist/notification/server-event seam queries
        synchronously on the UI thread from every compose, triage sync,
        and rail click. The adapter now caches those fields with a short
        TTL (see ``LocalNotificationHomeActiveWorkAdapter``); this worker
        just keeps that cache warm via ``asyncio.to_thread`` so callers on
        the UI thread hit the cache instead of the DB/services. Runs in
        its own worker group so it never cancels (or is cancelled by) the
        content-snapshot/chatbook-artifact workers. Test doubles (e.g.
        ``RecordingHomeActiveWorkAdapter``) don't implement the async
        refresh hook and are silently skipped.
        """
        if getattr(self.app_instance, "_home_dashboard_test_input", None) is not None:
            # ``_build_dashboard_input`` returns the injected test input
            # without ever consulting the adapter, so warming its cache is
            # pure dead weight -- and for memory-backed seams the warm
            # compute runs inline on the event loop, where it measurably
            # delayed mount settling (phase6 power-user replay gate).
            return
        adapter = getattr(self.app_instance, "home_active_work_adapter", None)
        refresh = getattr(adapter, "refresh_active_work_cache_async", None)
        if not inspect.iscoroutinefunction(refresh):
            return
        try:
            refreshed = await refresh()
        except Exception as exc:
            logger.debug(f"Home active-work cache refresh failed: {exc}")
            return
        # Only re-sync when the adapter actually recomputed -- a fresh
        # cache (the common on-mount case: compose just cold-computed and
        # stored it) means nothing changed, and re-syncing would just burn
        # a redundant triage rebuild during mount settling.
        if refreshed and self.is_mounted:
            self._sync_home_triage()

    async def _build_home_content_snapshot(self) -> HomeContentSnapshot:
        """Assemble the T190 content snapshot from real seams, degrading quietly."""
        console_ready = await asyncio.to_thread(self._home_console_provider_ready)
        notes_service = getattr(self.app_instance, "notes_scope_service", None)
        conversation_service = getattr(
            self.app_instance, "chat_conversation_scope_service", None
        )
        media_service = getattr(self.app_instance, "media_reading_scope_service", None)
        notes_user_id = getattr(self.app_instance, "notes_user_id", None) or "default_user"

        note_count_result = await self._home_content_seam_call(
            getattr(notes_service, "count_notes", None),
            scope="local_note",
            user_id=notes_user_id,
        )
        notes_result = await self._home_content_seam_call(
            getattr(notes_service, "list_notes", None),
            scope="local_note",
            limit=1,
            user_id=notes_user_id,
        )
        conversations_result = await self._home_content_seam_call(
            getattr(conversation_service, "list_conversations", None),
            mode="local",
            # "all" spans global- and workspace-scoped conversations, same
            # as Library's rail count: Console chats persisted inside a
            # workspace session would be invisible under 'global'.
            scope_type="all",
            limit=1,
            offset=0,
        )
        media_result = await self._home_content_seam_call(
            getattr(media_service, "list_media_items", None),
            mode="local",
            page=1,
            results_per_page=1,
            include_keywords=False,
        )

        resume_kind, resume_id, resume_title = _home_resume_fields(
            _home_first_record(notes_result),
            _home_first_record(conversations_result),
        )
        return HomeContentSnapshot(
            console_ready=console_ready,
            conversation_count=_home_response_total(conversations_result),
            note_count=(
                note_count_result if isinstance(note_count_result, int) else None
            ),
            media_count=_home_response_total(media_result),
            resume_kind=resume_kind,
            resume_id=resume_id,
            resume_title=resume_title,
        )

    async def _home_content_seam_call(self, callable_obj: Any, **kwargs: Any) -> Any:
        """Invoke one scope-service seam safely for the content snapshot.

        Degrades to ``None`` on any failure (missing seam, policy denial,
        backend unavailable). SQLite ``:memory:`` connections are
        thread-local -- only the thread that created ChaChaNotes has the
        migrated schema -- so the in-memory case runs inline on this (UI)
        thread while file-backed DBs keep the off-thread call (same guard
        as ``_refresh_home_chatbook_artifact_snapshot`` and Library's
        ``_study_count_or_none``).
        """
        if not callable(callable_obj):
            return None
        chachanotes_db = getattr(self.app_instance, "chachanotes_db", None)
        try:
            if getattr(chachanotes_db, "is_memory_db", False):
                result = callable_obj(**kwargs)
                if inspect.isawaitable(result):
                    return await result
                return result

            def invoke_seam_in_worker() -> Any:
                result = callable_obj(**kwargs)
                if inspect.isawaitable(result):
                    # This thread has no event loop, and the awaitable must
                    # complete here so blocking async services stay off the
                    # UI loop (mirrors Library's _run_library_service_call).
                    return asyncio.run(_await_home_seam_result(result))  # policy-exception: worker-thread loop
                return result

            return await asyncio.to_thread(invoke_seam_in_worker)
        except Exception as exc:
            logger.debug(f"Home content snapshot seam call failed: {exc}")
            return None

    def _home_console_provider_ready(self) -> bool:
        """Return Console provider readiness from the freshest config.

        Reuses the exact readiness seams Console uses
        (``build_default_console_session_settings`` +
        ``build_console_settings_readiness``) over ``load_settings()``
        rather than the boot-time ``app_config`` snapshot -- the staleness
        bug just fixed for Console (task-177) must not be reintroduced on
        Home. An injected hermetic test config (no disk-load marker
        sections) is honored verbatim, same as Console's
        ``_provider_readiness_app_config``.
        """
        app_config = getattr(self.app_instance, "app_config", {}) or {}
        config: Mapping[str, object] = app_config if isinstance(app_config, Mapping) else {}
        if all(section in config for section in _HOME_LIVE_CONFIG_MARKER_SECTIONS):
            try:
                fresh = load_settings()
            except Exception:
                logger.debug(
                    "Home readiness refresh via load_settings() failed; using snapshot"
                )
                fresh = None
            if isinstance(fresh, Mapping) and fresh:
                config = fresh
        try:
            settings = build_default_console_session_settings(config)
            readiness = build_console_settings_readiness(settings, app_config=config)
        except Exception as exc:
            logger.debug(f"Home Console readiness check failed: {exc}")
            return False
        return bool(readiness.native_send_supported)

    def _build_dashboard_input(self) -> HomeDashboardInput:
        test_override = getattr(self.app_instance, "_home_dashboard_test_input", None)
        if test_override is not None:
            return test_override

        providers = getattr(self.app_instance, "providers_models", {}) or {}
        has_recent_work = bool(getattr(self.app_instance, "_screen_states", {}))
        dashboard_input = self.app_instance.home_active_work_adapter.build_dashboard_input(
            providers_models=providers,
            has_recent_work=has_recent_work,
        )
        manager = getattr(self.app_instance, "acp_runtime_process_manager", None)
        snapshot = getattr(manager, "snapshot", None)
        if callable(snapshot):
            raw_snapshot = snapshot()
            if isinstance(raw_snapshot, dict):
                dashboard_input = replace(
                    dashboard_input,
                    acp_ready=str(raw_snapshot.get("status") or "") == "running",
                )
        content_snapshot = self._home_content_snapshot
        if content_snapshot is not None:
            # T190: fold in fresh-config readiness + real content counts and
            # the most-recent resume candidate (see the snapshot worker).
            dashboard_input = apply_home_content_snapshot(
                dashboard_input, content_snapshot
            )
        return dashboard_input

    def compose_content(self) -> ComposeResult:
        """Compose the Home triage route: header, rail, focus canvas."""
        dashboard_input = self._build_dashboard_input()
        triage = build_home_triage_state(
            dashboard_input,
            selected_row_id=self._home_selected_row_id,
        )
        # Keep the legacy dashboard object as the defensive fallback for
        # control dispatch (count-only canvases with no selection have
        # their controls in both); the scoped canvas controls
        # (``_current_canvas_controls``) are what the user actually sees
        # and presses, and are tried first in ``_activate_home_control``.
        self._current_dashboard = summarize_home_dashboard(dashboard_input)
        self._current_dashboard_input = dashboard_input
        self._current_canvas_controls = triage.canvas.actions
        self._home_selected_row_id = triage.selected_row_id

        yield Static(
            triage.header_line,
            id="home-header-line",
            classes="destination-status-row",
        )
        triage_grid = Horizontal(
            id="home-triage-grid", classes="ds-panel destination-workbench"
        )
        triage_grid.styles.height = "1fr"
        triage_grid.styles.min_height = 12
        with triage_grid:
            rail = HomeRail(
                triage,
                self._home_rail_preferences(),
                id="home-rail",
                classes="destination-workbench-pane",
            )
            rail.styles.height = "100%"
            yield rail
            canvas = HomeCanvas(
                triage.canvas,
                action_button_factory=self._home_action_button,
                id="home-canvas",
                classes="destination-workbench-pane",
            )
            canvas.styles.height = "100%"
            yield canvas

    def _home_action_button(
        self, label: str, control_id: str, primary: bool = False
    ) -> HomeActionButton:
        """Build a canvas action button with the fallback-press wiring.

        Args:
            label: Visible button label.
            control_id: Button id (also the dispatch key for non-primary
                controls; see ``HOME_CONTROL_METHODS``).
            primary: Whether this control carries primary emphasis for the
                currently selected row (see ``HomeCanvasState.
                primary_control_id`` / ``_canvas_primary_control_id``).
        """
        classes = "home-canvas-action console-action-primary" if primary else "home-canvas-action"
        if control_id == HOME_PRIMARY_ACTION_ID:
            return HomeActionButton(
                label,
                id=HOME_PRIMARY_ACTION_ID,
                classes=classes,
                fallback_press=self._activate_home_primary_action,
            )
        return HomeActionButton(
            label,
            id=control_id,
            classes=classes,
            fallback_press=lambda control_id=control_id: (
                self._activate_home_control(control_id)
            ),
        )

    def _home_rail_preferences(self) -> HomeRailPreferences:
        """Read persisted Home rail section preferences, defensively.

        (C4) Same restart-persistence gap as Library's
        ``_library_rail_preferences``/``_load_library_search_history``:
        ``self.app_instance.app_config`` (from ``load_settings()``) can
        come back without a ``home`` section at all even when
        ``config.toml`` has persisted ``[home.rail_state]`` on disk -- so
        a freshly started app would otherwise always reopen every Home
        rail section at its hardcoded default instead of the user's
        last-chosen open/collapsed state. Falls back to a live
        ``get_cli_setting("home.rail_state")`` read of the CLI config file
        when ``app_config`` doesn't already carry a usable ``sections``
        dict; ``app_config`` wins whenever it does.
        """
        app_config = getattr(self.app_instance, "app_config", None)
        raw = None
        if isinstance(app_config, dict):
            home_config = app_config.get("home")
            if isinstance(home_config, dict):
                rail_state = home_config.get("rail_state")
                if isinstance(rail_state, dict):
                    raw = rail_state.get("sections")
        if not isinstance(raw, dict):
            try:
                # Dotted 1-arg form, same shape as Library's rail
                # preferences fallback: `get_cli_setting("home.rail_state")`
                # returns `config["home"]["rail_state"]` (the rail_state
                # sub-dict), not the "sections" dict directly.
                cli_rail_state = get_cli_setting("home.rail_state")
            except Exception:
                cli_rail_state = None
            if isinstance(cli_rail_state, dict):
                raw = cli_rail_state.get("sections")
        return coerce_home_rail_preferences(raw)

    def _set_home_rail_section(self, section_id: str, open_state: bool) -> None:
        """Persist one section preference and sync the rail body/header."""
        if section_id not in HOME_RAIL_SECTION_IDS:
            return
        from dataclasses import replace as dataclass_replace

        preferences = dataclass_replace(
            self._home_rail_preferences(), **{f"{section_id}_open": open_state}
        )
        serialized = serialize_home_rail_preferences(preferences)
        app_config = getattr(self.app_instance, "app_config", None)
        if isinstance(app_config, dict):
            home_config = app_config.get("home")
            if not isinstance(home_config, dict):
                home_config = {}
                app_config["home"] = home_config
            rail_state = home_config.get("rail_state")
            if not isinstance(rail_state, dict):
                rail_state = {}
                home_config["rail_state"] = rail_state
            rail_state["sections"] = serialized
        self._save_home_rail_preferences(serialized)
        try:
            body = self.query_one(f"#home-rail-section-body-{section_id}")
            header = self.query_one(
                f"#home-rail-section-header-{section_id}", ConsoleRailSectionHeader
            )
        except Exception:
            return
        body.styles.display = "block" if open_state else "none"
        header.sync_open(open_state)

    @work(thread=True)
    def _save_home_rail_preferences(self, serialized: dict[str, bool]) -> None:
        """Persist Home rail preferences without blocking the UI thread."""
        try:
            save_setting_to_cli_config("home.rail_state", "sections", serialized)
        except Exception:
            pass

    def _sync_home_triage(self) -> None:
        """Rebuild triage state and refresh rail + canvas in place."""
        dashboard_input = self._build_dashboard_input()
        triage = build_home_triage_state(
            dashboard_input,
            selected_row_id=self._home_selected_row_id,
        )
        self._current_dashboard = summarize_home_dashboard(dashboard_input)
        self._current_dashboard_input = dashboard_input
        self._current_canvas_controls = triage.canvas.actions
        self._home_selected_row_id = triage.selected_row_id
        try:
            self.query_one("#home-rail", HomeRail).sync_state(
                triage, self._home_rail_preferences()
            )
            self.query_one("#home-canvas", HomeCanvas).sync_state(triage.canvas)
            self.query_one("#home-header-line", Static).update(triage.header_line)
        except Exception:
            pass

    def _selected_home_item(self, dashboard_input: HomeDashboardInput):
        return choose_home_selected_item(dashboard_input)

    @on(Button.Pressed)
    def handle_home_button(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if not button_id:
            return

        if button_id.startswith(HOME_RAIL_ROW_PREFIX):
            event.stop()
            row_id = str(getattr(event.button, "row_id", "") or "")
            if row_id:
                self._home_selected_row_id = row_id
                self._sync_home_triage()
            return

        if button_id.startswith(f"{CONSOLE_RAIL_SECTION_TOGGLE_PREFIX}home-"):
            event.stop()
            section_id = button_id.removeprefix(
                f"{CONSOLE_RAIL_SECTION_TOGGLE_PREFIX}home-"
            )
            currently_open = bool(
                getattr(self._home_rail_preferences(), f"{section_id}_open", True)
            )
            self._set_home_rail_section(section_id, not currently_open)
            return

        if button_id == HOME_PRIMARY_ACTION_ID:
            self._activate_home_primary_action()
            return

        self._activate_home_control(button_id)

    def _activate_home_primary_action(self) -> None:
        dashboard = self._current_dashboard
        if dashboard is None:
            return
        prepare = getattr(self.app_instance, "prepare_home_primary_action", None)
        if callable(prepare):
            prepare(dashboard.next_action)
        self.post_message(
            NavigateToScreen(
                dashboard.next_action.target_route,
                screen_context=_home_primary_action_context(dashboard.next_action),
            )
        )

    def _activate_home_control(self, button_id: str) -> None:
        # T190: the idle-canvas controls are screen-level navigations
        # (mirroring the model-setup recovery card's routing), not
        # app-instance runtime hooks -- dispatch them before the generic
        # HOME_CONTROL_METHODS lookup would misreport them as unconnected.
        if button_id == HOME_START_CONVERSATION_CONTROL_ID:
            self.post_message(NavigateToScreen(TAB_CHAT))
            return
        if button_id == HOME_RESUME_LATEST_CONTROL_ID:
            self._activate_home_resume_latest()
            return
        dashboard = self._current_dashboard
        if dashboard is None:
            return
        # T152: resolve from the SELECTION-SCOPED canvas controls first --
        # the set the user actually sees and presses, whose target_id
        # reflects the SELECTED item (e.g. home-retry pointed at the
        # selected failed item, not just the first failed item in the
        # list). Fall back to the unscoped dashboard controls only when
        # the pressed control isn't there (defensive: count-only fallback
        # canvases with no selection have their controls in both sets).
        control = next(
            (item for item in self._current_canvas_controls if item.control_id == button_id),
            None,
        )
        if control is None:
            control = next((item for item in dashboard.controls if item.control_id == button_id), None)
        if control is None:
            return

        method_name = HOME_CONTROL_METHODS.get(control.control_id)
        method = getattr(self.app_instance, method_name, None) if method_name else None
        if callable(method):
            kwargs = {}
            if control.target_id is not None:
                kwargs["target_id"] = control.target_id
            if control.control_id in HOME_CONTROL_METHODS_WITH_TARGET_ROUTE:
                kwargs["target_route"] = control.target_route
            if kwargs:
                method(**kwargs)
            else:
                method()
        else:
            self.app_instance.notify(
                f"{control.label} is not connected yet.",
                severity="warning",
            )

    def _activate_home_resume_latest(self) -> None:
        """Route the resume-latest control to its one-click destination.

        Notes deep-link into the Library notes editor via the existing
        ``LIBRARY_NAV_CONTEXT_NOTE_ID`` navigation-context contract;
        conversations route to Console. Navigation always composes a fresh
        screen, so the deep link lands on a cleanly mounted surface.
        """
        control = next(
            (
                item
                for item in self._current_canvas_controls
                if item.control_id == HOME_RESUME_LATEST_CONTROL_ID
            ),
            None,
        )
        if control is None:
            return
        if control.target_route == TAB_LIBRARY and control.target_id:
            self.post_message(
                NavigateToScreen(
                    TAB_LIBRARY,
                    {LIBRARY_NAV_CONTEXT_NOTE_ID: control.target_id},
                )
            )
            return
        self.post_message(NavigateToScreen(TAB_CHAT))
