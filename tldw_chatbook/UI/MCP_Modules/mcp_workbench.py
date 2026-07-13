# tldw_chatbook/UI/MCP_Modules/mcp_workbench.py
"""MCP Hub workbench: rail + mode canvases + inspector assembly."""

from __future__ import annotations

from typing import Any

from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import ContentSwitcher, Static

from tldw_chatbook.config import get_cli_setting
from tldw_chatbook.MCP.readiness import (
    HubAction,
    ReadinessSnapshot,
    builtin_readiness,
    local_profile_readiness,
    server_target_readiness,
)
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
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


class _AdvancedSectionShim:
    """Shields the inspector's legacy Advanced pane from one local-source shape gap.

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
            return payload
        if isinstance(payload, list) and section == "external_servers":
            # UnifiedMCPControlPlaneService.load_section() only returns a
            # bare list for the local-source "external_servers" section
            # (LocalMCPControlService.get_external_servers()); every other
            # section already comes back as a dict. render_external_servers_section
            # reads this key as a list.
            return {"source": "local", "section": section, "external_servers": payload}
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
            self._sync_children()
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
                try:
                    records = await service.load_section("external_servers")
                except Exception as exc:
                    logger.warning(f"MCP local profile listing failed: {exc}")
                    records = []
                if isinstance(records, list):  # local source returns a bare list
                    snapshots.extend(local_profile_readiness(r) for r in records)
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

    def _sync_children(self) -> None:
        rail = self.query_one(MCPRail)
        rail.sync_state(
            source=self._source,
            snapshots=self._snapshots,
            selected_server_key=self._selected_server_key,
            scope_options=[("Personal", "personal")],
            scope_value=self._scope,
            scope_ref_options=[],
            scope_ref_value=self._scope_ref,
        )
        canvas = self.query_one(MCPServersMode)
        canvas.update_overview(self._snapshots)
        selected = self._snapshot_for(self._selected_server_key)
        canvas.show_detail(selected)
        self.query_one(MCPInspector).update_readiness(selected)

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
        self._sync_children()

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
        self._sync_children()

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
        self._sync_children()

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

    def on_mcp_servers_mode_server_row_selected(
        self, event: MCPServersMode.ServerRowSelected
    ) -> None:
        event.stop()
        self._selected_server_key = event.server_key
        self._sync_children()

    def on_mcp_inspector_hub_action_requested(
        self, event: MCPInspector.HubActionRequested
    ) -> None:
        event.stop()
        if event.action is HubAction.VIEW_DETAILS and event.server_key:
            self._selected_server_key = event.server_key
            self.set_mode("servers")
            self._sync_children()
        elif event.action is HubAction.OPEN_TOOL_CATALOG:
            self.set_mode("tools")
        elif event.action is HubAction.OPEN_AUDIT:
            self.set_mode("audit")
