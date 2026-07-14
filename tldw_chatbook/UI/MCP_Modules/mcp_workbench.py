# tldw_chatbook/UI/MCP_Modules/mcp_workbench.py
"""MCP Hub workbench: rail + mode canvases + inspector assembly."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import ContentSwitcher, Static
from textual.worker import Worker

from tldw_chatbook.config import get_cli_setting
from tldw_chatbook.MCP.readiness import (
    HubAction,
    ReadinessSnapshot,
    as_checking,
    builtin_readiness,
    local_profile_readiness,
    server_target_readiness,
)
from tldw_chatbook.MCP.redaction import redact_args, redact_mapping
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
from tldw_chatbook.UI.MCP_Modules.mcp_profile_form import MCPProfileForm
from tldw_chatbook.UI.MCP_Modules.mcp_rail import MCPRail
from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode

# Sentinel distinguishing "key absent from a restore blob" from "key present
# with value None" -- see `_apply_view_state()`'s scope_ref handling.
_UNSET: Any = object()

MCP_HUB_MODES: dict[str, dict[str, str]] = {
    "servers": {"label": "Servers", "button_id": "mcp-mode-servers", "placeholder": ""},
    "tools": {
        "label": "Tools",
        "button_id": "mcp-mode-tools",
        "placeholder": (
            "Tools mode arrives in a later phase. Until then, a server's tools are "
            "listed in its Server detail, and tool actions run via Advanced in the inspector."
        ),
    },
    "permissions": {
        "label": "Permissions",
        "button_id": "mcp-mode-permissions",
        "placeholder": (
            "Permissions mode arrives in a later phase. MCP tools are not yet callable "
            "from chat, so there is nothing to permit yet."
        ),
    },
    "audit": {
        "label": "Audit",
        "button_id": "mcp-mode-audit",
        "placeholder": (
            "Audit mode arrives in a later phase. Action results appear inline in the "
            "inspector's Advanced section for now."
        ),
    },
}

_LEGACY_SECTIONS = [
    ("Overview", "overview"),
    ("Inventory", "inventory"),
    ("External Servers", "external_servers"),
    ("Governance", "governance"),
    ("Advanced", "advanced"),
]

# T5: local-profile lifecycle actions this workbench can dispatch, keyed by
# the short verb used throughout `_in_flight_action`/notifications. Maps to
# the typed T2 methods on `UnifiedMCPControlPlaneService` -- each raises with
# a user-ready message on failure and records its own attempt state, so the
# wrapper below must not re-record anything, just surface the result.
_LIFECYCLE_METHOD_NAMES: dict[str, str] = {
    "connect": "connect_local_profile",
    "test": "test_local_profile",
    "refresh": "refresh_local_profile",
    "disconnect": "disconnect_local_profile",
}

# Verb map from the inspector's HubActionRequested action to the lifecycle
# verb keys above -- only these three ever originate from the readiness
# action buttons (disconnect is a detail-view-only action, wired in T7).
_HUB_ACTION_TO_LIFECYCLE_VERB: dict[HubAction, str] = {
    HubAction.CONNECT: "connect",
    HubAction.VALIDATE: "test",
    HubAction.REFRESH_DISCOVERY: "refresh",
}

# Past-tense verb used in the success notification, e.g. "docs: connected — 3 tools."
_LIFECYCLE_PAST_TENSE: dict[str, str] = {
    "connect": "connected",
    "test": "checked",
    "refresh": "refreshed",
    "disconnect": "disconnected",
}


def _redact_external_server_record(record: Any) -> Any:
    """Redact one external-server record before it can reach the legacy
    Advanced renderer (frozen `render_external_servers_section()` in
    unified_mcp_sections.py).

    That renderer keys local records by "name", which local profile dicts
    never have (they use "profile_id"), so its `item.get(key) or item`
    fallback prints the FULL RAW DICT per entry -- CLI args and env values
    included -- whenever the key doesn't match. Non-Mapping records (already
    a shape the renderer can't consume sensibly) pass through unchanged.
    """
    if not isinstance(record, Mapping):
        return record
    record = dict(record)
    args = record.get("args")
    if isinstance(args, (list, tuple)):
        # redact_args handles `--api-key VALUE` / `key=value` CLI-arg shapes
        # that redact_mapping's generic key-based redaction below doesn't
        # reach (args is a plain list of strings, not a mapping).
        record["args"] = redact_args([str(a) for a in args])
    return redact_mapping(record)


def _redact_external_servers_list(records: Any) -> Any:
    if not isinstance(records, list):
        return records
    return [_redact_external_server_record(r) for r in records]


class _AdvancedSectionShim:
    """Shields the inspector's legacy Advanced pane from two local-source gaps.

    `UnifiedMCPControlPlaneService.load_section()` returns a dict for every
    section except local-source "external_servers", which comes back as a
    bare list (mirroring `LocalMCPControlService.get_external_servers()`).
    The renderers in `unified_mcp_sections.py` all assume a Mapping, and
    `MCPInspector.set_service_context()`/`on_select_changed()` schedule the
    section load as a worker with Textual's default `exit_on_error=True` —
    that one shape mismatch (or a raised exception) there would crash the
    whole app, not just the Advanced pane. Normalize and fail closed here
    instead, at the integration seam this task owns, without touching
    mcp_inspector.py.

    Second gap: `render_external_servers_section()` (also frozen) keys
    records by "name", which local profile dicts never have -- its fallback
    then prints each FULL RAW DICT, secrets included (CLI args, env values).
    Records are redacted here, at this same seam, before the renderer ever
    sees them -- on both the bare-list local path and any dict payload that
    already carries an "external_servers" list (the server-source shape).
    """

    def __init__(self, service: Any) -> None:
        self._service = service

    def __getattr__(self, name: str) -> Any:
        return getattr(self._service, name)

    async def load_section(self, section: str | None = None) -> dict[str, Any]:
        try:
            payload = await self._service.load_section(section)
        except Exception as exc:
            logger.warning(f"MCP workbench advanced section load failed: {exc}")
            return {"source": "local", "section": section or "overview", "error": str(exc)}
        if isinstance(payload, dict):
            if isinstance(payload.get("external_servers"), list):
                payload = dict(payload)
                payload["external_servers"] = _redact_external_servers_list(
                    payload["external_servers"]
                )
            return payload
        if isinstance(payload, list) and section == "external_servers":
            # UnifiedMCPControlPlaneService.load_section() only returns a
            # bare list for the local-source "external_servers" section
            # (LocalMCPControlService.get_external_servers()); every other
            # section already comes back as a dict. render_external_servers_section
            # reads this key as a list.
            return {
                "source": "local",
                "section": section,
                "external_servers": _redact_external_servers_list(payload),
            }
        return {"source": "local", "section": section or "overview"}


class MCPWorkbench(Container):
    """Assembles the Phase 1 MCP Hub. Read-only over the control-plane service."""

    class ModeChanged(Message, namespace="mcp_workbench"):
        """Posted by `set_mode()` whenever the active mode actually changes,
        so the hosting screen can keep its mode-chip highlight in sync.
        `set_mode` is the single emission point: it covers every path that
        changes the mode without going through `MCPScreen._activate_mode()`
        (a click or keybinding) -- state restore and inspector hub actions
        ("Open tool catalog"/"Open audit") alike. The screen's chip sync is
        idempotent, so the redundant notification on the _activate_mode
        path is harmless."""

        def __init__(self, mode: str) -> None:
            super().__init__()
            self.mode = mode

    DEFAULT_CSS = """
    MCPWorkbench {
        width: 100%;
        height: 1fr;
        min-height: 0;
    }
    #mcp-hub-grid {
        width: 100%;
        height: 100%;
        min-height: 0;
    }
    #mcp-hub-canvas {
        width: 5fr;
        min-width: 38;
        height: 100%;
        min-height: 0;
    }
    """

    def __init__(self, app_instance: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._app_instance = app_instance
        self._active_mode = "servers"
        self._source = "local"
        self._selected_server_key: str | None = None
        self._scope: str = "personal"
        self._scope_ref: str | None = None
        self._snapshots: list[ReadinessSnapshot] = []
        # T6: raw local-profile catalog records keyed by profile_id, kept in
        # sync with `_snapshots` by `_collect_snapshots()` -- readiness
        # snapshots don't carry every field the add/edit form needs
        # (profile_id/command/args/env_placeholders/env_literals), so
        # `show_form()` on an EDIT_CONFIG hub action looks the record up
        # here instead.
        self._catalog_records: dict[str, dict[str, Any]] = {}
        # T5: in-flight local-profile lifecycle operations, keyed by
        # server_key ("local:<profile_id>"). While a key is present here,
        # `_snapshot_for_display()`/`_sync_children()` render that server as
        # CHECKING (see `as_checking()`) regardless of its last-known
        # readiness, and the inspector shows a Cancel button instead of the
        # normal action set.
        self._in_flight: dict[str, Worker] = {}
        # The lifecycle verb ("connect"/"test"/"refresh"/"disconnect") each
        # in-flight key is running, so the CHECKING badge's message and the
        # eventual notification can say what's actually happening.
        self._in_flight_action: dict[str, str] = {}
        self._pending_view_state: dict[str, Any] | None = None
        # Guards the post-mount restore race: `on_mount` awaits `reload()`
        # inline, but a caller (e.g. the destination screen) can call
        # `set_initial_view_state()` while that reload is still in flight.
        # Without this flag, the reload's own `_sync_children()` and a
        # concurrently scheduled restore worker can race to write
        # `_source`/`_selected_server_key`/`_scope`/`_scope_ref` last.
        self._reloading: bool = False

    @property
    def active_mode(self) -> str:
        return self._active_mode

    @property
    def app_instance(self) -> Any:
        if self._app_instance is not None:
            return self._app_instance
        try:
            return self.app
        except Exception:
            return None

    def _service(self) -> Any:
        return getattr(self.app_instance, "unified_mcp_service", None)

    def compose(self) -> ComposeResult:
        with Horizontal(id="mcp-hub-grid", classes="destination-workbench"):
            yield MCPRail(
                source=self._source,
                snapshots=[],
                selected_server_key=None,
                scope_options=[("Personal", "personal")],
                scope_value=self._scope,
                scope_ref_options=[],
                scope_ref_value=self._scope_ref,
                id="mcp-hub-rail",
                classes="destination-workbench-pane",
            )
            with ContentSwitcher(
                initial="mcp-mode-canvas-servers",
                id="mcp-hub-canvas",
                classes="destination-workbench-pane",
            ):
                yield MCPServersMode(id="mcp-mode-canvas-servers")
                for mode, spec in MCP_HUB_MODES.items():
                    if mode == "servers":
                        continue
                    with Vertical(id=f"mcp-mode-canvas-{mode}"):
                        yield Static(
                            spec["placeholder"],
                            classes="ds-recovery-callout",
                            markup=False,
                        )
            yield MCPInspector(id="mcp-hub-inspector", classes="destination-workbench-pane")

    async def on_mount(self) -> None:
        await self.reload()

    # -- data loading ---------------------------------------------------------

    async def reload(self) -> None:
        self._reloading = True
        try:
            service = self._service()
            if service is not None:
                try:
                    context = await service.load_context()
                    self._source = context.selected_source or "local"
                    if (
                        self._source == "server"
                        and context.selected_active_server_id
                        and self._selected_server_key is None
                    ):
                        self._selected_server_key = f"server:{context.selected_active_server_id}"
                    if context.selected_scope is not None:
                        self._scope = context.selected_scope
                    if context.selected_scope_ref is not None:
                        self._scope_ref = context.selected_scope_ref
                except Exception as exc:
                    logger.warning(f"MCP workbench context load failed: {exc}")
            self._snapshots = await self._collect_snapshots()
            await self._sync_children()
            inspector = self.query_one(MCPInspector)
            inspector.set_service_context(
                _AdvancedSectionShim(service) if service is not None else None,
                _LEGACY_SECTIONS,
            )
        finally:
            self._reloading = False
        # Consume any view state that arrived while this reload was in
        # flight (see `set_initial_view_state()`), so it is applied exactly
        # once and always after this reload's own `_sync_children()`.
        await self._consume_pending_view_state()

    async def _collect_snapshots(self) -> list[ReadinessSnapshot]:
        snapshots: list[ReadinessSnapshot] = []
        service = self._service()
        if self._source == "local":
            snapshots.append(
                builtin_readiness(
                    enabled=bool(get_cli_setting("mcp", "enabled", False)),
                    expose_tools=bool(get_cli_setting("mcp", "expose_tools", True)),
                    expose_resources=bool(get_cli_setting("mcp", "expose_resources", True)),
                    expose_prompts=bool(get_cli_setting("mcp", "expose_prompts", True)),
                )
            )
            if service is not None:
                # T5: `local_external_catalog()` (T2) additionally attaches
                # each record's persisted `runtime_state` (last connect/test/
                # refresh attempt), which `local_profile_readiness()` uses to
                # surface a specific failure reason instead of the generic
                # "not currently connected". Fall back to the Phase 1 path
                # for any service that doesn't expose it yet (older fakes).
                catalog_loader = getattr(service, "local_external_catalog", None)
                try:
                    if callable(catalog_loader):
                        records = await catalog_loader()
                    else:
                        records = await service.load_section("external_servers")
                except Exception as exc:
                    logger.warning(f"MCP local profile listing failed: {exc}")
                    records = []
                if isinstance(records, list):  # local source returns a bare list
                    snapshots.extend(local_profile_readiness(r) for r in records)
                    self._catalog_records = {
                        str(r.get("profile_id")): dict(r)
                        for r in records
                        if isinstance(r, Mapping) and r.get("profile_id")
                    }
        else:
            target_store = getattr(service, "target_store", None)
            if target_store is not None:
                snapshots.extend(
                    server_target_readiness(t) for t in target_store.list_targets()
                )
        return snapshots

    def _snapshot_for(self, server_key: str | None) -> ReadinessSnapshot | None:
        if server_key is None:
            return None
        for snap in self._snapshots:
            if snap.server_key == server_key:
                return snap
        return None

    def _display_snapshot(self, snapshot: ReadinessSnapshot) -> ReadinessSnapshot:
        """Overlay the in-flight CHECKING state onto a snapshot for rendering.

        `self._snapshots` itself always holds the last *derived* readiness
        (from `_collect_snapshots()`) so a cancelled or failed lifecycle
        action has something correct to fall back to -- this wraps it with
        `as_checking()` only for display, purely based on whether the key is
        currently in `self._in_flight`.
        """
        if snapshot.server_key in self._in_flight:
            action = self._in_flight_action.get(snapshot.server_key, "update")
            return as_checking(snapshot, action)
        return snapshot

    def _snapshot_for_display(self, server_key: str | None) -> ReadinessSnapshot | None:
        snapshot = self._snapshot_for(server_key)
        return None if snapshot is None else self._display_snapshot(snapshot)

    async def _sync_children(self) -> None:
        """Push current state into the rail/canvas/inspector children.

        Awaited end to end -- `MCPInspector.update_readiness()` must fully
        finish its remove+mount cycle (see mcp_inspector.py) before this
        coroutine returns, so that Textual's message pump cannot dequeue a
        second selection event and start another `_sync_children()` call
        while the first's inspector refresh is still settling
        (`DuplicateIds` regression; see test_mcp_inspector.py).
        """
        display_snapshots = [self._display_snapshot(snap) for snap in self._snapshots]
        rail = self.query_one(MCPRail)
        rail.sync_state(
            source=self._source,
            snapshots=display_snapshots,
            selected_server_key=self._selected_server_key,
            scope_options=[("Personal", "personal")],
            scope_value=self._scope,
            scope_ref_options=[],
            scope_ref_value=self._scope_ref,
        )
        canvas = self.query_one(MCPServersMode)
        await canvas.update_overview(display_snapshots)
        selected = self._snapshot_for_display(self._selected_server_key)
        canvas.show_detail(selected)
        await self.query_one(MCPInspector).update_readiness(selected)

    # -- modes & view state ---------------------------------------------------

    def set_mode(self, mode: str) -> None:
        if mode not in MCP_HUB_MODES:
            mode = "servers"
        mode_changed = mode != self._active_mode
        self._active_mode = mode
        self.query_one(ContentSwitcher).current = f"mcp-mode-canvas-{mode}"
        if mode_changed:
            # Single emission point for mode changes (see ModeChanged) --
            # covers restore and inspector hub-action paths, which bypass
            # the screen's _activate_mode chip sync.
            self.post_message(self.ModeChanged(mode))

    def get_view_state(self) -> dict[str, Any]:
        return {
            "mode": self.active_mode,
            "source": self._source,
            "selected_server_key": self._selected_server_key,
            "scope": self._scope,
            "scope_ref": self._scope_ref,
        }

    def set_initial_view_state(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        if self.is_mounted:
            # Always stash the latest requested state so a reload already in
            # flight (see `reload()`) can pick it up when it finishes,
            # instead of racing it with a worker started here.
            self._pending_view_state = dict(state)
            if not self._reloading:
                self.run_worker(
                    self._consume_pending_view_state(),
                    group="mcp-workbench-restore",
                    exclusive=True,
                )
        else:
            self._pending_view_state = dict(state)

    async def _consume_pending_view_state(self) -> None:
        """Apply `_pending_view_state` exactly once, then clear it."""
        state = self._pending_view_state
        self._pending_view_state = None
        if state:
            await self._apply_view_state(state)

    async def _apply_view_state(self, state: dict[str, Any]) -> None:
        # Tolerant restore: unknown keys ignored; legacy panel shape accepted.
        source = state.get("source") or state.get("selected_source")
        if source in ("local", "server") and source != self._source:
            await self._switch_source(str(source))
        # I2: a restored non-"servers" mode must also move the screen's
        # chip highlight -- set_mode() itself posts ModeChanged on any
        # actual change (single emission point), so no extra post here.
        self.set_mode(str(state.get("mode") or "servers"))
        server_key = state.get("selected_server_key")
        if isinstance(server_key, str) and self._snapshot_for(server_key) is not None:
            self._selected_server_key = server_key
        scope = state.get("scope") or state.get("selected_scope")
        if isinstance(scope, str) and scope:
            self._scope = scope
        # T7 carry-over: distinguish "key absent" (keep the current
        # scope_ref untouched) from "key present with value None" (an
        # explicit clear). `dict.get(key, _UNSET)` is required here because
        # `state.get("scope_ref")` alone can't tell "absent" from
        # "present-but-None" apart -- both return None.
        if "scope_ref" in state:
            raw_scope_ref = state["scope_ref"]
        elif "selected_scope_ref" in state:
            raw_scope_ref = state["selected_scope_ref"]
        else:
            raw_scope_ref = _UNSET
        if raw_scope_ref is not _UNSET:
            self._scope_ref = None if raw_scope_ref is None else str(raw_scope_ref)
        await self._sync_children()

    # -- event wiring -----------------------------------------------------------

    async def _switch_source(self, source: str) -> None:
        service = self._service()
        if service is not None:
            try:
                await service.select_source(source)
            except Exception as exc:
                logger.warning(f"MCP source switch failed: {exc}")
        self._source = source
        self._selected_server_key = None
        self._snapshots = await self._collect_snapshots()
        await self._sync_children()

    async def on_mcp_rail_source_changed(self, event: MCPRail.SourceChanged) -> None:
        event.stop()
        await self._switch_source(event.source)

    async def on_mcp_rail_server_selected(self, event: MCPRail.ServerSelected) -> None:
        event.stop()
        self._selected_server_key = event.server_key
        service = self._service()
        if (
            service is not None
            and event.server_key is not None
            and event.server_key.startswith("server:")
            and "/" not in event.server_key
        ):
            try:
                await service.select_server_target(event.server_key.split(":", 1)[1])
            except Exception as exc:
                logger.warning(f"MCP server target selection failed: {exc}")
        await self._sync_children()

    async def on_mcp_rail_scope_changed(self, event: MCPRail.ScopeChanged) -> None:
        event.stop()
        # C1 defense in depth: a no-op ScopeChanged (already-tracked scope +
        # scope_ref) must not round-trip to the service or resync children.
        # The primary fix is the rail's own mount-echo guard (mcp_rail.py),
        # but this dedup means a stray duplicate here can't self-sustain a
        # recompose storm even if some future caller posts one.
        if (event.scope, event.scope_ref) == (self._scope, self._scope_ref):
            return
        service = self._service()
        if service is not None:
            try:
                await service.select_scope(event.scope, event.scope_ref)
            except Exception as exc:
                logger.warning(f"MCP scope selection failed: {exc}")
        self._scope = event.scope
        self._scope_ref = event.scope_ref
        # No `_sync_children()` here: nothing scope-dependent renders in
        # Phase 1 (the rail's scope selects reflect it purely from the last
        # explicit sync_state() call), and resyncing would recompose the
        # rail -> remount its Selects -> another mount-echo -> another
        # ScopeChanged, which is exactly the storm this handler used to feed.

    async def on_mcp_servers_mode_server_row_selected(
        self, event: MCPServersMode.ServerRowSelected
    ) -> None:
        event.stop()
        self._selected_server_key = event.server_key
        await self._sync_children()

    async def on_mcp_inspector_hub_action_requested(
        self, event: MCPInspector.HubActionRequested
    ) -> None:
        event.stop()
        if event.action is HubAction.VIEW_DETAILS and event.server_key:
            self._selected_server_key = event.server_key
            self.set_mode("servers")
            await self._sync_children()
        elif event.action is HubAction.OPEN_TOOL_CATALOG:
            self.set_mode("tools")
        elif event.action is HubAction.OPEN_AUDIT:
            self.set_mode("audit")
        elif (
            event.action in _HUB_ACTION_TO_LIFECYCLE_VERB
            and event.server_key
            and event.server_key.startswith("local:")
        ):
            profile_id = event.server_key.split(":", 1)[1]
            self._start_lifecycle(
                event.server_key, profile_id, _HUB_ACTION_TO_LIFECYCLE_VERB[event.action]
            )
        elif (
            event.action is HubAction.EDIT_CONFIG
            and event.server_key
            and event.server_key.startswith("local:")
        ):
            profile_id = event.server_key.split(":", 1)[1]
            record = self._catalog_records.get(profile_id)
            await self.query_one(MCPServersMode).show_form(record)

    async def on_mcp_servers_mode_add_server_requested(
        self, event: MCPServersMode.AddServerRequested
    ) -> None:
        event.stop()
        await self.query_one(MCPServersMode).show_form(None)

    def on_mcp_profile_form_submit_requested(
        self, event: MCPProfileForm.SubmitRequested
    ) -> None:
        """Dispatch a profile save in the background.

        Synchronous (not `async def`), mirroring `_start_lifecycle()`: the
        handler itself must return immediately so Textual's message pump
        stays responsive while the save runs -- the actual `await
        service.save_local_profile(...)` happens inside the worker coroutine
        below.
        """
        event.stop()
        self.run_worker(
            self._save_local_profile(dict(event.payload)),
            group="mcp-profile-save",
            exclusive=True,
        )

    async def _save_local_profile(self, payload: dict[str, Any]) -> None:
        service = self._service()
        if service is None:
            return
        try:
            await service.save_local_profile(payload)
        except ValueError as exc:
            try:
                form = self.query_one(MCPProfileForm)
            except Exception:
                form = None
            if form is not None:
                form.show_error(str(exc))
            return
        except Exception as exc:
            logger.warning(f"MCP profile save failed: {exc}")
            self.app.notify(f"Save failed: {exc}", severity="error")
            return
        canvas = self.query_one(MCPServersMode)
        await canvas.hide_form()
        self.app.notify(f"Saved {payload.get('profile_id')}.")
        self._snapshots = await self._collect_snapshots()
        await self._sync_children()

    async def on_mcp_profile_form_cancelled(self, event: MCPProfileForm.Cancelled) -> None:
        event.stop()
        await self.query_one(MCPServersMode).hide_form()

    def on_mcp_inspector_cancel_requested(self, event: MCPInspector.CancelRequested) -> None:
        """Cancel an in-flight lifecycle worker.

        Synchronous (not `async def`): `Worker.cancel()` is itself
        synchronous, and the caller (Textual's message pump, or a test
        calling this directly) doesn't need to await anything here -- the
        display resync is fired off as its own worker below instead of being
        awaited inline.
        """
        event.stop()
        worker = self._in_flight.pop(event.server_key, None)
        self._in_flight_action.pop(event.server_key, None)
        if worker is None:
            # Stale cancel: the operation already finished and popped itself
            # (its own completion toast + resync have run). Toasting
            # "Cancelled." here would falsely claim a completed operation
            # was stopped -- silent no-op instead.
            return
        worker.cancel()
        self.app.notify("Cancelled.")
        self.run_worker(self._sync_children(), group="mcp-lifecycle-sync", exclusive=True)

    # -- lifecycle actions (T5: connect/test/refresh/disconnect) --------------

    def _start_lifecycle(self, server_key: str, profile_id: str, action: str) -> None:
        """Dispatch a local-profile lifecycle action in the background.

        Synchronous (not `async def`): must register `self._in_flight`
        synchronously, before returning to the caller, so a `CancelRequested`
        arriving right after this call (or a second click of the same
        action) reliably observes the worker that was just started -- if
        this were `async def` and awaited only later, the bookkeeping below
        wouldn't run until the event loop actually scheduled this coroutine,
        leaving a window where the guard/cancel logic would see stale state.
        """
        if server_key in self._in_flight:
            self.app.notify(f"{profile_id}: {action} already running.", severity="warning")
            return
        service = self._service()
        method_name = _LIFECYCLE_METHOD_NAMES.get(action)
        method = getattr(service, method_name, None) if service is not None and method_name else None
        if not callable(method):
            logger.warning(
                f"MCP workbench: no lifecycle method for action={action!r} "
                f"(server_key={server_key!r})"
            )
            return
        coro = method(profile_id)
        worker = self.run_worker(
            self._lifecycle_wrapper(server_key, profile_id, action, coro),
            group="mcp-lifecycle",
            exclusive=False,
        )
        self._in_flight[server_key] = worker
        self._in_flight_action[server_key] = action
        # Render the CHECKING badge + inspector Cancel button immediately --
        # decoupled from the lifecycle worker above, which may be sitting on
        # a slow (or, in tests, gated) network/subprocess call and must not
        # block this optimistic UI update.
        self.run_worker(self._sync_children(), group="mcp-lifecycle-sync", exclusive=True)

    async def _lifecycle_wrapper(
        self, server_key: str, profile_id: str, action: str, coro: Any
    ) -> None:
        """Run one lifecycle coroutine, then always clean up and resync.

        The T2 typed methods (`connect_local_profile` etc.) already record
        their own attempt state and raise a user-ready message on failure --
        this must not duplicate that recording, only surface the outcome and
        drop the in-flight marker. `except Exception` deliberately does not
        catch `asyncio.CancelledError` (a `BaseException` since Python 3.8):
        a cancelled worker skips straight to `finally`, which is exactly the
        cleanup `on_mcp_inspector_cancel_requested()` needs and which the
        cancel handler's own notify()/resync above already covers, so no
        redundant "cancelled" notification is sent from here.
        """
        try:
            result = await coro
        except Exception as exc:
            self.app.notify(f"{profile_id}: {action} failed — {exc}", severity="error")
        else:
            verb = _LIFECYCLE_PAST_TENSE.get(action, action)
            tool_count = self._lifecycle_tool_count(result)
            if tool_count is None:
                self.app.notify(f"{profile_id}: {verb}.")
            else:
                noun = "tool" if tool_count == 1 else "tools"
                self.app.notify(f"{profile_id}: {verb} — {tool_count} {noun}.")
        finally:
            self._in_flight.pop(server_key, None)
            self._in_flight_action.pop(server_key, None)
            self._snapshots = await self._collect_snapshots()
            await self._sync_children()

    @staticmethod
    def _lifecycle_tool_count(result: Any) -> int | None:
        """Best-effort tool count from a lifecycle result for the success notice.

        `connect_local_profile`/`refresh_local_profile` return a dict with a
        `tools` list; `test_local_profile` returns a dict with a `tools`
        *count* (int); `disconnect_local_profile` returns a bare bool. Any
        other shape just omits the tool count from the notification.
        """
        if not isinstance(result, Mapping):
            return None
        tools = result.get("tools")
        if isinstance(tools, list):
            return len(tools)
        if isinstance(tools, int) and not isinstance(tools, bool):
            return tools
        return None
