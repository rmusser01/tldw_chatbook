# tldw_chatbook/UI/MCP_Modules/mcp_workbench.py
"""MCP Hub workbench: rail + mode canvases + inspector assembly."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import ContentSwitcher, Static
from textual.worker import Worker

from tldw_chatbook.config import get_cli_setting, save_setting_to_cli_config
from tldw_chatbook.MCP.mcp_import import ImportCandidate
from tldw_chatbook.MCP.readiness import (
    HubAction,
    ReadinessSnapshot,
    as_checking,
    builtin_readiness,
    local_profile_readiness,
    server_external_record_readiness,
    server_target_readiness,
)
from tldw_chatbook.MCP.redaction import redact_args, redact_mapping
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
from tldw_chatbook.UI.MCP_Modules.mcp_profile_form import MCPImportPanel, MCPProfileForm
from tldw_chatbook.UI.MCP_Modules.mcp_rail import MCPRail
from tldw_chatbook.UI.MCP_Modules.mcp_server_mutations import MCPServerMutationsPanel
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

# T9: success-notify copy per server-mutation action name. A generic
# "<last segment> saved." fallback would read as "Create saved."/"Delete
# saved." for the slot actions -- ambiguous about *what* was created or
# deleted -- so every wired action gets its own sentence instead.
_SERVER_MUTATION_MESSAGES: dict[str, str] = {
    "external_server.create": "External server created.",
    "external_server.update": "External server updated.",
    "external_server.slot.create": "Credential slot added.",
    "external_server.slot.secret.set": "Secret set.",
    "external_server.slot.secret.clear": "Secret cleared.",
    "external_server.slot.delete": "Credential slot deleted.",
}


def _import_summary(succeeded: list[str], failed: list[tuple[str, str]]) -> str:
    """One notify-ready sentence covering a whole import batch.

    Every candidate is attempted regardless of an earlier failure (T8: "a
    failing save produces the summary notify without aborting the rest") --
    this renders whatever mix of successes/failures resulted into a single
    toast instead of one per candidate.
    """
    parts: list[str] = []
    if succeeded:
        parts.append(f"Imported {len(succeeded)}: {', '.join(succeeded)}.")
    if failed:
        failed_desc = ", ".join(f"{profile_id} ({error})" for profile_id, error in failed)
        parts.append(f"Failed {len(failed)}: {failed_desc}.")
    return " ".join(parts) if parts else "Nothing to import."


def _import_severity(succeeded: list[str], failed: list[tuple[str, str]]) -> str:
    if failed and not succeeded:
        return "error"
    if failed:
        return "warning"
    return "information"


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
        # T6: True while a profile-save worker is in flight. Mirrors
        # `_start_lifecycle()`'s synchronous-registration pattern: set in the
        # (sync) SubmitRequested handler before the worker is dispatched and
        # cleared in the worker's `finally`, so a second Save arriving in the
        # same pump window reliably observes it. Without this, dispatching
        # every submit through `run_worker(..., exclusive=True)` let a second
        # click CANCEL the in-flight save mid-write.
        self._profile_save_in_flight: bool = False
        # T8: True while an mcpServers-import apply worker is in flight. Same
        # synchronous-registration guard as `_profile_save_in_flight`: set
        # before the worker is dispatched, cleared in its `finally`, so a
        # second Import click during a slow batch can't dispatch a second
        # overlapping apply worker.
        self._profile_import_in_flight: bool = False
        # T7: True while a profile-delete worker is in flight. Same
        # synchronous-registration guard as `_profile_save_in_flight` above:
        # set before the worker is dispatched, cleared in its `finally`, so
        # a second DeleteConfirmed arriving in the same pump window (e.g. a
        # double-click on "Confirm delete" before the button unmounts)
        # cannot cancel the in-flight delete mid-write.
        self._profile_delete_in_flight: bool = False
        # T9: True while an external-server-record mutation worker
        # (external_server.create/update or a credential-slot action) is in
        # flight. Same synchronous-registration guard as the local-profile
        # flags above: set before dispatch, cleared in the worker's
        # `finally`.
        self._server_mutation_in_flight: bool = False
        # T9: whether the active server target's scope permits
        # `external_server.*` mutations (team/org/system-admin only --
        # `service.available_actions()` returns `[]` for "personal"). Reset
        # to False for local source and recomputed for server source in
        # `_collect_snapshots()` (whenever a target's external-servers
        # section is loaded, which pins the service context's
        # `selected_section` to "external_servers" -- see that method's
        # docstring for why this is read directly off `available_actions()`
        # instead of a synthetic select-then-restore round trip) and again,
        # cheaply, on scope changes (`on_mcp_rail_scope_changed`).
        self._server_mutations_available: bool = False
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
        # T12 review fix: identity of the object the inspector's Advanced
        # pane was last rebound to (see `_rebind_inspector_advanced_context`).
        # None until the first rebind so the mount-time reload() always
        # binds.
        self._advanced_rebind_key: tuple[Any, ...] | None = None
        self._pending_view_state: dict[str, Any] | None = None
        # T7: `_sync_children()` now flows through `MCPServersMode.show_detail()`,
        # which (since T7 added the detail toolbar) performs its own awaited
        # remove_children()/mount_all() cycle on `#mcp-detail-toolbar` --
        # real suspension points a concurrently *running* `_sync_children()`
        # call can interleave with. `_start_lifecycle()` deliberately fires
        # two independent workers (the immediate "mcp-lifecycle-sync" resync
        # for the optimistic CHECKING badge, and "mcp-lifecycle"'s own
        # finally-triggered resync once the lifecycle call completes) --
        # different worker groups, so Textual's `exclusive=True` cancellation
        # within one group does not serialize them against each other. This
        # lock does: a second `_sync_children()` call simply waits for the
        # first's remove+mount cycle to finish (harmless -- it repaints with
        # whatever the latest state is once it acquires the lock) instead of
        # racing it and raising DuplicateIds.
        #
        # DELIBERATELY BROAD (T7 review adjudication): review proposed
        # narrowing this to a toolbar-local lock per the widget-local
        # `MCPInspector._refresh_lock` convention; kept broad because beyond
        # the DuplicateIds fix it also guarantees whole-triad consistency --
        # rail, overview table, and inspector always render from the same
        # snapshot generation within one locked pass, so a torn state where
        # the rail shows refresh A while the table shows refresh B cannot
        # occur -- and it serializes the DataTable clear()/add_row() cycle in
        # `update_overview()` against concurrent workers. Relationship to
        # `MCPInspector._refresh_lock`: that one is the inspector's own
        # widget-local guard for its external callers (worker-driven Advanced
        # refreshes racing pump-driven ones); on the `_sync_children()` path
        # it is always acquired AFTER this lock (via `update_readiness()`
        # inside the locked block), and no code path acquires them in the
        # opposite order -- same ordering everywhere, no AB-BA deadlock.
        self._sync_children_lock = asyncio.Lock()
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
                            # T12: phase placeholders are informational, not a
                            # recovery/alarm condition -- the warning chrome
                            # was semantic dilution of the single alarm color
                            # (UX-inputs #4). T13 adds the `.ds-info-callout`
                            # CSS rule itself; this is the class-name swap.
                            classes="ds-info-callout",
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
            self._rebind_inspector_advanced_context(service)
        finally:
            self._reloading = False
        # Consume any view state that arrived while this reload was in
        # flight (see `set_initial_view_state()`), so it is applied exactly
        # once and always after this reload's own `_sync_children()`.
        await self._consume_pending_view_state()

    def _selected_target_id(self) -> str | None:
        """The server-target id implied by `_selected_server_key`.

        Handles both a target row directly selected ("server:main") and an
        external-record row beneath it ("server:main/docs") -- both drill
        into the same target's external-servers listing.
        """
        key = self._selected_server_key
        if not key or not key.startswith("server:"):
            return None
        remainder = key.split(":", 1)[1]
        return remainder.split("/", 1)[0] if remainder else None

    def _active_service_target_id(self) -> str | None:
        """The target id server-source operations would actually run against.

        UI selection wins when present, but `run_action`'s server branch
        resolves its target from the SERVICE context
        (`_require_active_server_target()` reads
        `context.selected_active_server_id`), not from the workbench's local
        selection -- and the two genuinely diverge: Add-server is only ever
        reachable from the overview, where `_selected_server_key` is None
        while the service still remembers the last-activated target. Falling
        back to the service context here keeps everything derived from this
        id (external-record loading, the post-create drill, the Add-server
        tooltip's target naming) consistent with where a mutation would
        really land.
        """
        target_id = self._selected_target_id()
        if target_id is not None:
            return target_id
        service = self._service()
        context = getattr(service, "context", None) if service is not None else None
        active = getattr(context, "selected_active_server_id", None)
        return str(active) if active else None

    def _active_target_label(self) -> str | None:
        """Human label for `_active_service_target_id()`'s target, or None.

        Prefers the target store's configured label; falls back to the raw
        id so the Add-server tooltip can always name a resolvable target.
        """
        target_id = self._active_service_target_id()
        if target_id is None:
            return None
        target_store = getattr(self._service(), "target_store", None)
        if target_store is not None:
            try:
                for target in target_store.list_targets():
                    if str(getattr(target, "server_id", "")) == target_id:
                        label = getattr(target, "label", None)
                        return str(label) if label else target_id
            except Exception as exc:
                logger.warning(f"MCP target label lookup failed: {exc}")
        return target_id

    def _rebind_inspector_advanced_context(self, service: Any) -> None:
        """Push the current source/target into the inspector's Advanced pane.

        T12 (UX-inputs #1): "rebind or reset the section content whenever
        the selection changes so reopening never shows a previous object's
        facts; and label the object the content describes." Calling
        `set_service_context()` again resets the Advanced section back to
        its first entry and reloads it against the (possibly new) service
        context -- the same full rebind `reload()` already did on mount --
        so this is called from every place that changes which object the
        service context refers to: `reload()` itself, `_switch_source()`,
        and `_select_server_key()`.

        Review fix: rebinding is deduplicated on the OBJECT's identity, not
        on every call -- the UX-inputs text requires a rebind on selection
        CHANGE, and e.g. reclicking the already-selected rail row (or a
        no-op reload) is not a change; unconditionally rebinding there wiped
        the user's Advanced browsing state (section snapping back to
        Overview). Mirrors the C1 ScopeChanged dedup precedent in this
        file. The Advanced object is the local control plane (local source,
        regardless of which row is selected) or the active server target
        (server source) -- so the key is the source plus, for server source,
        the active target id/label; the service's identity is included so a
        swapped-in service (e.g. None -> real) always rebinds.
        """
        target_label = self._active_target_label()
        if self._source == "server":
            identity: Any = (self._active_service_target_id(), target_label)
        else:
            identity = None
        key = (id(service) if service is not None else None, self._source, identity)
        if key == self._advanced_rebind_key:
            return
        self._advanced_rebind_key = key
        self.query_one(MCPInspector).set_service_context(
            _AdvancedSectionShim(service) if service is not None else None,
            _LEGACY_SECTIONS,
            source=self._source,
            target_label=target_label,
        )

    @staticmethod
    def _is_external_record_key(server_key: str | None) -> bool:
        if not server_key or not server_key.startswith("server:"):
            return False
        remainder = server_key.split(":", 1)[1]
        return "/" in remainder

    def _compute_server_mutations_available(self, service: Any) -> bool:
        """Whether `external_server.*` mutation actions are usable right now.

        `available_actions()` only returns the `external_server.*` set when
        the service context's `selected_section` is "external_servers"
        (mirrors the legacy Advanced panel/inspector -- see
        mcp_inspector.py's `_load_advanced_section` C2 comment). Rather than
        issuing a synthetic `select_section("external_servers")` +
        `available_actions()` + restore-previous-section round trip purely
        to answer this question, this piggybacks on the read that
        `_collect_snapshots()` already performs for real, functional reasons
        whenever a server target is selected: loading that target's
        external-servers section (to render its record rows) pins
        `selected_section` to "external_servers" as a side effect of real
        navigation, so `available_actions()` called right after is accurate
        with no extra round trip and no context left mutated beyond what the
        UI was already doing. When no target is ACTIVE at all
        (`_active_service_target_id()` is None -- neither a UI selection nor
        a service-remembered target), that load never ran and
        `selected_section` may be stale -- this then reads as unavailable,
        which happens to also be the honest answer: without an active
        target, `external_server.create` has nowhere to attach anyway (and
        the Add-server button additionally carries its own no-target gate,
        see `MCPServersMode._update_add_server_button()`).
        """
        if service is None:
            return False
        loader = getattr(service, "available_actions", None)
        if not callable(loader):
            return False
        try:
            actions = loader() or []
        except Exception as exc:
            logger.warning(f"MCP available_actions check failed: {exc}")
            return False
        return any(
            isinstance(a, Mapping) and a.get("name") == "external_server.create"
            for a in actions
        )

    async def _collect_snapshots(self) -> list[ReadinessSnapshot]:
        snapshots: list[ReadinessSnapshot] = []
        service = self._service()
        if self._source == "local":
            self._server_mutations_available = False
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
            # T9: with an ACTIVE target (a target/external-record row
            # selected in the UI, or -- review fix -- the service context's
            # remembered active target when nothing is visibly selected),
            # also load and append that target's external-server records --
            # they appear in rail/table beneath the target, keyed
            # "server:<target>/<ext>". Gating on the service's notion of
            # active (not just the UI selection) is what lets a freshly
            # created record show up immediately: Add-server runs from the
            # overview with no selection at all.
            target_id = self._active_service_target_id()
            if service is not None and target_id is not None:
                try:
                    payload = await service.load_section("external_servers")
                except Exception as exc:
                    logger.warning(f"MCP external server listing failed: {exc}")
                    payload = None
                records = payload.get("external_servers") if isinstance(payload, Mapping) else None
                if isinstance(records, list):
                    snapshots.extend(
                        server_external_record_readiness(r, server_id=target_id)
                        for r in records
                        if isinstance(r, Mapping)
                    )
            self._server_mutations_available = self._compute_server_mutations_available(service)
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

        Wrapped in `_sync_children_lock` (T7): that guards the *pump*
        ordering above, but `_start_lifecycle()` also fires this method from
        two independent worker groups (the immediate optimistic-CHECKING
        resync and the lifecycle wrapper's own completion resync), which can
        genuinely run concurrently as separate asyncio tasks -- the lock
        serializes those too, so two overlapping calls' remove+mount cycles
        (now real suspension points, since `MCPServersMode.show_detail()`
        rebuilds the detail toolbar) can't interleave on the same widgets.
        """
        async with self._sync_children_lock:
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
            await canvas.update_overview(
                display_snapshots,
                source=self._source,
                mutations_available=self._server_mutations_available,
                mutation_target_label=self._active_target_label(),
            )
            selected = self._snapshot_for_display(self._selected_server_key)
            await self._show_selected_detail(canvas, selected)
            await self.query_one(MCPInspector).update_readiness(selected)

    async def _show_selected_detail(
        self, canvas: MCPServersMode, selected: ReadinessSnapshot | None
    ) -> None:
        """Route the selected snapshot to the read-only detail pane or,
        for an external-server record when mutations are available, to the
        `MCPServerMutationsPanel` edit-mode host (T9).

        Credential slots are fetched fresh on every selection (not cached)
        -- they can change from other clients/sessions, and this only runs
        on an actual selection change, not on every keystroke.
        """
        if (
            selected is not None
            and self._is_external_record_key(selected.server_key)
            and self._server_mutations_available
        ):
            record = dict((selected.detail or {}).get("raw") or {})
            record.setdefault("server_id", selected.server_key.rsplit("/", 1)[-1])
            slots = await self._fetch_credential_slots(record.get("server_id"))
            await canvas.show_server_mutations(record, slots)
            return
        await canvas.show_detail(selected, mutations_available=self._server_mutations_available)

    async def _fetch_credential_slots(self, server_id: Any) -> list[dict[str, Any]]:
        service = self._service()
        if service is None or not server_id:
            return []
        try:
            result = await service.run_action(
                "external_server.slots.list", {"server_id": server_id}
            )
        except Exception as exc:
            logger.warning(f"MCP credential slot listing failed: {exc}")
            return []
        slots = result.get("credential_slots") if isinstance(result, Mapping) else None
        return [dict(s) for s in slots if isinstance(s, Mapping)] if isinstance(slots, list) else []

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
            # T7 review fix: a mode change is an "other interaction" per the
            # arm-then-confirm contract, so it must disarm any pending delete
            # confirmation -- the ContentSwitcher hides the servers canvas
            # without unmounting it, so nothing else resets the arm state on
            # a Servers -> Tools -> Servers round-trip. Dispatched as a
            # worker (set_mode is sync); no-op when unarmed.
            self.run_worker(
                self._disarm_canvas_delete(),
                group="mcp-detail-disarm",
                exclusive=True,
            )

    async def _disarm_canvas_delete(self) -> None:
        # Under `_sync_children_lock`: `disarm_delete()` rebuilds the detail
        # toolbar (awaited remove+mount), which must not interleave with a
        # concurrently running `_sync_children()` doing the same via
        # `show_detail()` -- same DuplicateIds hazard the lock exists for.
        async with self._sync_children_lock:
            await self.query_one(MCPServersMode).disarm_delete()

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
        self._rebind_inspector_advanced_context(service)

    async def on_mcp_rail_source_changed(self, event: MCPRail.SourceChanged) -> None:
        event.stop()
        await self._switch_source(event.source)

    async def _select_server_key(self, server_key: str | None) -> None:
        """Shared selection path for both the rail and the overview table.

        T9: previously only the rail's handler informed the service which
        target is active (`select_server_target`) and re-collected
        snapshots; the table's row-click handler just resynced from the
        existing `_snapshots`. That gap didn't matter in Phase 1 (nothing
        was target-scoped), but now that `_collect_snapshots()` loads a
        selected target's external-server records off the service's *active*
        target, a table-driven selection had no way to make it active --
        both entry points now share this one path.
        """
        self._selected_server_key = server_key
        service = self._service()
        if (
            service is not None
            and server_key is not None
            and server_key.startswith("server:")
            and "/" not in server_key
        ):
            try:
                await service.select_server_target(server_key.split(":", 1)[1])
            except Exception as exc:
                logger.warning(f"MCP server target selection failed: {exc}")
        if self._source == "server":
            self._snapshots = await self._collect_snapshots()
        await self._sync_children()
        self._rebind_inspector_advanced_context(service)

    async def on_mcp_rail_server_selected(self, event: MCPRail.ServerSelected) -> None:
        event.stop()
        await self._select_server_key(event.server_key)

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
        #
        # T9: mutation availability IS scope-dependent though -- recompute it
        # cheaply (no snapshot/rail/detail resync, just the Add-server
        # button's gating) so a scope change alone doesn't leave it stale.
        if self._source == "server":
            self._server_mutations_available = self._compute_server_mutations_available(service)
            self.query_one(MCPServersMode).set_mutations_available(
                self._server_mutations_available,
                mutation_target_label=self._active_target_label(),
            )

    async def on_mcp_servers_mode_server_row_selected(
        self, event: MCPServersMode.ServerRowSelected
    ) -> None:
        event.stop()
        await self._select_server_key(event.server_key)

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
        await self._open_add_server(notify_if_gated=False)

    async def open_add_server_form(self) -> None:
        """Open the Add-server form/panel from outside the overview button.

        T13: entry point for the `a` keybinding, which -- unlike a
        `Button.Pressed` on the overview's own (already gate-disabled)
        Add-server button -- can fire while server-source mutations are
        gated off. Reachable-while-gated means silently no-opping (the
        button-press behavior) would leave the user with no explanation, so
        this notifies with the button's own gate copy instead.
        """
        await self._open_add_server(notify_if_gated=True)

    async def _open_add_server(self, *, notify_if_gated: bool) -> None:
        canvas = self.query_one(MCPServersMode)
        if self._source == "server":
            # T9: mirrors `MCPServersMode._update_add_server_button()`'s gate
            # precedence -- scope gate first, then the no-active-target gate.
            # A real Button.Pressed can't reach here while either fails (the
            # button is disabled), but a defensive check costs nothing.
            if not self._server_mutations_available:
                if notify_if_gated:
                    self._notify_add_server_gated(canvas)
                return
            if self._active_service_target_id() is None:
                if notify_if_gated:
                    self._notify_add_server_gated(canvas)
                return
            await canvas.show_server_mutations(None, [])
        else:
            await canvas.show_form(None)

    def _notify_add_server_gated(self, canvas: MCPServersMode) -> None:
        """Surface the overview Add-server button's own gate tooltip as a notification.

        Reuses whatever `MCPServersMode._update_add_server_button()` already
        computed rather than duplicating the gate copy, so the notification
        and the button's own explanation can never drift apart.
        """
        try:
            button = canvas.query_one("#mcp-add-server")
        except Exception:
            button = None
        message = str(button.tooltip) if button is not None and button.tooltip else (
            "Adding a server is unavailable right now."
        )
        self.app.notify(message, severity="warning")

    async def on_mcp_servers_mode_import_servers_requested(
        self, event: MCPServersMode.ImportServersRequested
    ) -> None:
        event.stop()
        # T8: existing catalog ids drive the panel's overwrite warnings --
        # `_catalog_records` is kept in sync with `_snapshots` by
        # `_collect_snapshots()` (Task 6).
        await self.query_one(MCPServersMode).show_import(set(self._catalog_records))

    async def on_mcp_servers_mode_disconnect_requested(
        self, event: MCPServersMode.DisconnectRequested
    ) -> None:
        """Route the detail toolbar's Disconnect button through the same
        `_start_lifecycle()` dispatch T5 wired for connect/test/refresh --
        disconnect is a detail-view-only action, so it never comes through
        `HubActionRequested`/`_HUB_ACTION_TO_LIFECYCLE_VERB` like those three.
        """
        event.stop()
        if event.server_key and event.server_key.startswith("local:"):
            profile_id = event.server_key.split(":", 1)[1]
            self._start_lifecycle(event.server_key, profile_id, "disconnect")

    def on_mcp_servers_mode_builtin_flag_changed(
        self, event: MCPServersMode.BuiltinFlagChanged
    ) -> None:
        """Dispatch a built-in server enable/expose toggle in the background.

        Synchronous (not `async def`), mirroring
        `on_mcp_servers_mode_delete_confirmed()`/`on_mcp_profile_form_
        submit_requested()`: the handler returns immediately so the message
        pump stays responsive while the config write + catalog reload run.
        No in-flight guard (unlike those two): each Checkbox already
        displays its own last-known value between toggles, so a rapid
        second toggle -- of the same or a different flag -- simply cancels
        the still-running worker via `exclusive=True` (safe: `Checkbox.
        Changed` is idempotent config state, not an append-only mutation)
        and starts fresh from the latest event.
        """
        event.stop()
        self.run_worker(
            self._save_builtin_flag(event.key, event.value),
            group="mcp-builtin-flag",
            exclusive=True,
        )

    async def _save_builtin_flag(self, key: str, value: bool) -> None:
        """Persist one `[mcp]` enable/expose flag, then reload the catalog.

        The write itself is the blocking part (TOML read-modify-write to
        disk, `save_setting_to_cli_config()` in config.py) -- offloaded via
        `asyncio.to_thread` rather than Textual's `@work(thread=True)`
        decorator (the fire-and-forget precedent at
        library_screen.py:5534's `_save_library_rail_preferences()`)
        because this call, unlike that one, MUST follow the write with
        async work that touches live widgets (`_collect_snapshots()` +
        `_sync_children()`, so the built-in row's readiness badge and this
        detail pane's own checkboxes reflect the change). Keeping both
        steps in one coroutine dispatched via `run_worker(coroutine, ...)`
        mirrors this file's own `_load_import_file()` (`asyncio.to_thread`
        for a blocking `Path.read_text` followed by an in-coroutine UI
        update) instead of adding a `call_from_thread` marshaling hop back
        onto the event loop that a sync `@work(thread=True)` method would
        need for the same follow-up.
        """
        try:
            saved = await asyncio.to_thread(save_setting_to_cli_config, "mcp", key, value)
        except Exception as exc:
            logger.warning(f"MCP built-in flag save failed: {exc}")
            self.app.notify(f"Failed to save {key}: {exc}", severity="error")
            return
        if not saved:
            self.app.notify(f"Failed to save {key}.", severity="error")
            return
        self._snapshots = await self._collect_snapshots()
        await self._sync_children()

    def on_mcp_servers_mode_delete_confirmed(
        self, event: MCPServersMode.DeleteConfirmed
    ) -> None:
        """Dispatch a profile delete in the background.

        Synchronous (not `async def`), mirroring
        `on_mcp_profile_form_submit_requested()`: the handler itself must
        return immediately so Textual's message pump stays responsive while
        the delete runs -- the actual `await
        service.delete_local_profile(...)` happens inside the worker
        coroutine below. `_profile_delete_in_flight` (set here,
        synchronously, before dispatch; cleared in the worker's `finally`)
        is what makes a double confirm safe.
        """
        event.stop()
        if not event.server_key or not event.server_key.startswith("local:"):
            return
        profile_id = event.server_key.split(":", 1)[1]
        if self._profile_delete_in_flight:
            self.app.notify(f"{profile_id}: delete already running.", severity="warning")
            return
        self._profile_delete_in_flight = True
        self.run_worker(
            self._delete_local_profile(event.server_key, profile_id),
            group="mcp-profile-delete",
            exclusive=True,
        )

    async def _delete_local_profile(self, server_key: str, profile_id: str) -> None:
        try:
            service = self._service()
            if service is None:
                return
            try:
                await service.delete_local_profile(profile_id)
            except Exception as exc:
                logger.warning(f"MCP profile delete failed: {exc}")
                self.app.notify(f"Delete failed: {exc}", severity="error")
                return
            self.app.notify(f"Deleted {profile_id}.")
            if self._selected_server_key == server_key:
                self._selected_server_key = None
            self._snapshots = await self._collect_snapshots()
            await self._sync_children()
        finally:
            self._profile_delete_in_flight = False

    def on_mcp_profile_form_submit_requested(
        self, event: MCPProfileForm.SubmitRequested
    ) -> None:
        """Dispatch a profile save in the background.

        Synchronous (not `async def`), mirroring `_start_lifecycle()`: the
        handler itself must return immediately so Textual's message pump
        stays responsive while the save runs -- the actual `await
        service.save_local_profile(...)` happens inside the worker coroutine
        below. The `_profile_save_in_flight` guard (set here, synchronously,
        before dispatch; cleared in the worker's `finally`) is what makes a
        double Save safe: without it, `exclusive=True` alone let a second
        submit CANCEL the in-flight save mid-write and start over.
        """
        event.stop()
        if self._profile_save_in_flight:
            self.app.notify("Save already running.", severity="warning")
            return
        self._profile_save_in_flight = True
        self.run_worker(
            self._save_local_profile(dict(event.payload), warning=event.warning),
            group="mcp-profile-save",
            exclusive=True,
        )

    def _form_or_none(self) -> MCPProfileForm | None:
        try:
            return self.query_one(MCPProfileForm)
        except Exception:
            return None

    async def _save_local_profile(
        self, payload: dict[str, Any], warning: str | None = None
    ) -> None:
        """Run one profile save; on success, also re-surface the form's args
        secret-lint `warning` as a toast (I4 follow-up). The in-form
        `#mcp-form-args-warning` Static only survives FAILED saves -- the
        success path's `hide_form()` below unmounts the whole form
        sub-second after the warning rendered, so without this toast the
        user would never see it on exactly the path where the secret
        actually got persisted into a profile's args.
        """
        try:
            service = self._service()
            if service is None:
                return
            try:
                await service.save_local_profile(payload)
            except ValueError as exc:
                # Store-validation copy is user-ready. If the form is gone
                # (e.g. cancelled while the save was in flight), the failure
                # must still surface -- never vanish silently.
                form = self._form_or_none()
                if form is not None:
                    form.show_error(str(exc))
                else:
                    self.app.notify(str(exc), severity="error")
                return
            except Exception as exc:
                logger.warning(f"MCP profile save failed: {exc}")
                # Route through show_error when possible: it also re-enables
                # the form's Save button (disabled at submit) for a retry.
                form = self._form_or_none()
                if form is not None:
                    form.show_error(f"Save failed: {exc}")
                else:
                    self.app.notify(f"Save failed: {exc}", severity="error")
                return
            canvas = self.query_one(MCPServersMode)
            await canvas.hide_form()
            self.app.notify(f"Saved {payload.get('profile_id')}.")
            if warning:
                self.app.notify(warning, severity="warning")
            self._snapshots = await self._collect_snapshots()
            await self._sync_children()
        finally:
            self._profile_save_in_flight = False

    async def on_mcp_profile_form_cancelled(self, event: MCPProfileForm.Cancelled) -> None:
        event.stop()
        await self.query_one(MCPServersMode).hide_form()

    # -- T9: server-source external-server + credential-slot mutations --------

    def _mutations_panel_or_none(self) -> MCPServerMutationsPanel | None:
        try:
            return self.query_one(MCPServerMutationsPanel)
        except Exception:
            return None

    def on_mcp_server_mutations_submit_requested(
        self, event: MCPServerMutationsPanel.SubmitRequested
    ) -> None:
        """Dispatch one `run_action(action, payload)` call in the background.

        Synchronous (not `async def`), mirroring
        `on_mcp_profile_form_submit_requested()`: `_server_mutation_in_flight`
        is set here, before dispatch, so a second Save/Add-slot/Set-secret
        press arriving in the same pump window is reliably swallowed with a
        warning toast instead of racing the in-flight call.
        """
        event.stop()
        if self._server_mutation_in_flight:
            self.app.notify("Save already running.", severity="warning")
            return
        self._server_mutation_in_flight = True
        self.run_worker(
            self._run_server_mutation(event.action, dict(event.payload)),
            group="mcp-server-mutation",
            exclusive=True,
        )

    async def _run_server_mutation(self, action: str, payload: dict[str, Any]) -> None:
        try:
            service = self._service()
            if service is None:
                return
            try:
                await service.run_action(action, payload)
            except Exception as exc:
                logger.warning(f"MCP server mutation failed ({action}): {exc}")
                panel = self._mutations_panel_or_none()
                if panel is not None:
                    panel.show_error(str(exc))
                else:
                    self.app.notify(f"{action} failed: {exc}", severity="error")
                return
            self.app.notify(
                _SERVER_MUTATION_MESSAGES.get(
                    action, f"{action.rsplit('.', 1)[-1].replace('_', ' ').title()} saved."
                )
            )
            if action == "external_server.create":
                # Drill straight into the record just created -- credential
                # setup is the natural next step, and `_sync_children()`
                # below will fetch its slots and show the mutation panel in
                # edit mode (T9's `_show_selected_detail()`). Review fix:
                # derived from the SERVICE's active target, because create
                # only ever runs from the overview where the local UI
                # selection is None -- `_selected_target_id()` alone made
                # this branch dead.
                target_id = self._active_service_target_id()
                server_id = payload.get("server_id")
                if target_id and server_id:
                    self._selected_server_key = f"server:{target_id}/{server_id}"
            self._snapshots = await self._collect_snapshots()
            await self._sync_children()
        finally:
            self._server_mutation_in_flight = False

    async def on_mcp_server_mutations_cancelled(
        self, event: MCPServerMutationsPanel.Cancelled
    ) -> None:
        """Close the mutations panel AND clear the selection that opened it.

        I2 fix: `show_server_mutations()` never updates `_detail_snapshot`
        (it hosts an edit-mode panel for an external-server record instead
        of routing through `show_detail()`), so `hide_form()`'s own
        `_detail_snapshot is None` check can't be trusted to land back on
        the overview -- and, worse, `_selected_server_key` was left pointing
        at the external record, so the very next `_sync_children()` (a
        background lifecycle completion, the `r` keybinding, a
        runtime-backend refresh) would call `_show_selected_detail()` again
        and re-open this same panel out of nowhere.
        Routes through the exact same path `ServerRowSelected(None)` uses
        (`_select_server_key(None)`): clears `_selected_server_key`, then a
        full resync lands on the overview with the table cursor restored.
        `hide_form()` runs first purely to close the panel/unmount its
        widgets; the resync right after is what settles the correct
        final container visibility regardless of `hide_form()`'s own
        (possibly stale) guess.
        """
        event.stop()
        await self.query_one(MCPServersMode).hide_form()
        await self._select_server_key(None)

    # -- T8: mcpServers import (paste or file) ---------------------------------

    def _import_panel_or_none(self) -> MCPImportPanel | None:
        try:
            return self.query_one(MCPImportPanel)
        except Exception:
            return None

    async def on_mcp_import_panel_file_requested(
        self, event: MCPImportPanel.FileRequested
    ) -> None:
        event.stop()
        from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen, Filters

        def on_file_selected(file_path: Any) -> None:
            if file_path:
                self.run_worker(
                    self._load_import_file(str(file_path)),
                    group="mcp-import-file",
                    exclusive=True,
                )

        await self.app.push_screen(
            EnhancedFileOpen(
                location=".",
                title="Select MCP config JSON",
                filters=Filters(("JSON", lambda p: p.suffix.lower() == ".json")),
                context="mcp_import",
            ),
            callback=on_file_selected,
        )

    async def _load_import_file(self, file_path: str) -> None:
        try:
            text = await asyncio.to_thread(Path(file_path).read_text, encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            # UnicodeDecodeError (a ValueError subclass, not an OSError) is
            # raised by `read_text` for any non-UTF-8 file -- e.g. a
            # Claude-Desktop config saved with a BOM/legacy encoding. Left
            # uncaught, it escapes this worker and, with Textual's default
            # `exit_on_error=True`, takes down the whole app (C1).
            self.app.notify(f"Could not read {file_path}: {exc}", severity="error")
            return
        panel = self._import_panel_or_none()
        if panel is not None:
            panel.set_file_text(text)

    async def on_mcp_import_panel_cancelled(self, event: MCPImportPanel.Cancelled) -> None:
        event.stop()
        await self.query_one(MCPServersMode).hide_form()

    def on_mcp_import_panel_import_requested(
        self, event: MCPImportPanel.ImportRequested
    ) -> None:
        """Dispatch a batch of candidate saves in the background.

        Synchronous (not `async def`), mirroring
        `on_mcp_profile_form_submit_requested()`: `_profile_import_in_flight`
        is set here, before dispatch, so a second Import press arriving in
        the same pump window is reliably swallowed with a warning toast
        instead of racing the in-flight batch.
        """
        event.stop()
        if self._profile_import_in_flight:
            self.app.notify("Import already running.", severity="warning")
            return
        self._profile_import_in_flight = True
        self.run_worker(
            self._apply_import(list(event.candidates)),
            group="mcp-profile-import",
            exclusive=True,
        )

    async def _apply_import(self, candidates: list[ImportCandidate]) -> None:
        try:
            service = self._service()
            if service is None:
                return
            succeeded: list[str] = []
            failed: list[tuple[str, str]] = []
            for candidate in candidates:
                try:
                    await service.save_local_profile(candidate.to_payload())
                except Exception as exc:
                    logger.warning(f"MCP import failed for {candidate.profile_id}: {exc}")
                    failed.append((candidate.profile_id, str(exc)))
                else:
                    succeeded.append(candidate.profile_id)
            self.app.notify(_import_summary(succeeded, failed), severity=_import_severity(succeeded, failed))
            canvas = self.query_one(MCPServersMode)
            await canvas.hide_form()
            self._snapshots = await self._collect_snapshots()
            await self._sync_children()
        finally:
            self._profile_import_in_flight = False

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
