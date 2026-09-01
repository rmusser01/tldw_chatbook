"""ADR-099 guards for Terminal's user-only, process-memory boundary."""

from __future__ import annotations

import ast
import dataclasses
import json
import re
from collections.abc import Mapping
from pathlib import Path

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _wait_for_selector
from tldw_chatbook.Agents.agent_models import ToolCatalogEntry, ToolSchema
from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
from tldw_chatbook.Agents.tool_catalog import LIBRARY_RESERVED_TOOL_NAMES
from tldw_chatbook.Agents.run_log import RunLogWriter
from tldw_chatbook.Chat.console_agent_bridge import _compose_run_registry_and_allowed
from tldw_chatbook.Chat.console_library_policy import ConsoleAssistantLibraryAccess
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail, ExchangeCapture
from tldw_chatbook.Chat.console_exchange_export import project_exchange_export
from tldw_chatbook.Chat.trace_export_profiles import TraceExportProfile
from tldw_chatbook.config import load_settings
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.Terminal.contracts import (
    AdmissionGate,
    BackendIdentity,
    CleanupAttempt,
    CleanupProof,
    TerminalLaunchRequest,
    TerminalProjection,
)
from tldw_chatbook.Terminal.session_manager import TerminalSessionManager
from tldw_chatbook.UI.Navigation.screen_state_store import (
    RuntimeIdentity,
    ScreenStateStore,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService


REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCT_ROOT = REPO_ROOT / "tldw_chatbook"
TERMINAL_ROOT = PRODUCT_ROOT / "Terminal"
TERMINAL_IMPORTERS = frozenset(
    {
        "tldw_chatbook/Utils/input_validation.py",
        "tldw_chatbook/Widgets/Console/console_terminal_session_modal.py",
        "tldw_chatbook/Widgets/Console/console_terminal_workspace.py",
        "tldw_chatbook/UI/Console_Modules/terminal.py",
        "tldw_chatbook/UI/Console_Modules/wiring.py",
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "tldw_chatbook/UI/Screens/settings_screen.py",
        "tldw_chatbook/app.py",
    }
)
TERMINAL_RUNTIME_OWNERS = frozenset(
    {
        "tldw_chatbook/UI/Console_Modules/wiring.py",
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "tldw_chatbook/UI/Screens/settings_screen.py",
        "tldw_chatbook/app.py",
    }
)
FORBIDDEN_SINKS = (
    "model_tool_schemas",
    "model_tool_catalog",
    "mcp_permission_storage",
    "provider_messages",
    "provider_history",
    "console_conversation_messages",
    "agent_runs",
    "run_logs",
    "exports",
    "workspace_persistence",
    "config",
    "database_writes",
    "app_snapshot",
    "workspace_snapshot",
    "conversation_snapshot",
    "crash_recovery_snapshot",
)
CONCRETE_SINKS = FORBIDDEN_SINKS
SESSION_NAME = "TERM-PRIVACY-NAME-22512"
START_DIRECTORY_MARKER = "TERM-PRIVACY-PATH-22512"
TERMINAL_TOOL_WORDS = frozenset({"terminal", "pty", "conpty"})


class _Backend:
    def start(
        self, request: TerminalLaunchRequest, admission: AdmissionGate
    ) -> BackendIdentity:
        assert request.name == SESSION_NAME
        assert START_DIRECTORY_MARKER in request.start_directory
        assert admission.admitted is True
        return BackendIdentity(session_id=admission.token)

    def write(self, data: bytes) -> None:
        del data

    def resize(self, columns: int, rows: int) -> None:
        del columns, rows

    def request_priority_close(self) -> None:
        return None

    def finalize_shutdown(self) -> None:
        return None

    def cleanup(self, attempt: CleanupAttempt) -> CleanupProof:
        del attempt
        return CleanupProof(True, True, True)


class _CatalogProvider:
    """Small provider double used through the production run composer."""

    def __init__(self, source: str, name: str) -> None:
        self.entry = ToolCatalogEntry(
            id=f"{source}:{name}",
            name=name,
            one_line_description=f"{source} composition probe",
            source=source,
        )

    def list_catalog(self) -> list[ToolCatalogEntry]:
        return [self.entry]

    def load_schema(self, tool_id: str) -> ToolSchema:
        assert tool_id == self.entry.id
        return ToolSchema(
            id=tool_id,
            name=self.entry.name,
            description=self.entry.one_line_description,
            parameters={"type": "object", "properties": {}},
        )

    def invoke(self, tool_id: str, args: dict) -> object:
        raise AssertionError(f"privacy characterization must not invoke {tool_id}")


def _tree(path: Path, source: str | None = None) -> ast.Module:
    return ast.parse(
        source if source is not None else path.read_text(encoding="utf-8"),
        filename=str(path),
    )


def _imports_terminal(path: Path, source: str) -> bool:
    for node in ast.walk(_tree(path, source)):
        if isinstance(node, ast.Import) and any(
            alias.name == "tldw_chatbook.Terminal"
            or alias.name.startswith("tldw_chatbook.Terminal.")
            for alias in node.names
        ):
            return True
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        if module == "tldw_chatbook.Terminal" or module.startswith(
            "tldw_chatbook.Terminal."
        ):
            return True
        if module == "tldw_chatbook" and any(
            alias.name == "Terminal" for alias in node.names
        ):
            return True
        if node.level and any(alias.name == "Terminal" for alias in node.names):
            return True
        if node.level and module.split(".", 1)[0] == "Terminal":
            return True
    return False


def _files_referencing_manager_owner() -> set[str]:
    owners: set[str] = set()
    names = {"terminal_session_manager", "_terminal_session_manager_shutdown_task"}
    for path in PRODUCT_ROOT.rglob("*.py"):
        if path.is_relative_to(TERMINAL_ROOT):
            continue
        source = path.read_text(encoding="utf-8")
        if not any(name in source for name in names):
            continue
        tree = _tree(path, source)
        if any(
            (isinstance(node, ast.Name) and node.id in names)
            or (isinstance(node, ast.Attribute) and node.attr in names)
            or (isinstance(node, ast.Constant) and node.value in names)
            for node in ast.walk(tree)
        ):
            owners.add(path.relative_to(REPO_ROOT).as_posix())
    return owners


def _manager_construction_sites(root: Path = PRODUCT_ROOT) -> dict[str, int]:
    sites: dict[str, int] = {}
    for path in root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = _tree(path, source)
        constructor_names = {"TerminalSessionManager"}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if not (node.module or "").endswith("Terminal.session_manager"):
                continue
            constructor_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "TerminalSessionManager"
            )
        count = sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Name)
                and node.func.id in constructor_names
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == "TerminalSessionManager"
            )
        )
        if count:
            sites[path.relative_to(root).as_posix()] = count
    return sites


def _contains_terminal_material(value: object, markers: tuple[str, ...]) -> bool:
    value_type = type(value)
    if value_type.__module__.startswith("tldw_chatbook.Terminal"):
        return True
    if isinstance(value, str):
        return any(marker in value for marker in markers)
    if isinstance(value, bytes):
        return any(marker.encode() in value for marker in markers)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return any(
            _contains_terminal_material(getattr(value, field.name), markers)
            for field in dataclasses.fields(value)
        )
    if isinstance(value, Mapping):
        return any(
            _contains_terminal_material(key, markers)
            or _contains_terminal_material(item, markers)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_terminal_material(item, markers) for item in value)
    return False


def _assert_no_terminal_model_tool(
    catalog: object,
    schemas: object,
) -> None:
    rendered = json.dumps(
        {"catalog": catalog, "schemas": schemas},
        default=lambda value: dataclasses.asdict(value)
        if dataclasses.is_dataclass(value)
        else repr(value),
        sort_keys=True,
    ).casefold()
    words = {
        part
        for token in re.findall(r"[a-z0-9_]+", rendered)
        for part in token.split("_")
        if part
    }
    found = sorted(words.intersection(TERMINAL_TOOL_WORDS))
    assert not found, f"model-visible Terminal tool vocabulary found: {found}"


def _production_model_catalog() -> tuple[list[object], list[object], tuple[str, ...]]:
    local_provider = _CatalogProvider("local", "local_probe")
    virtual_provider = _CatalogProvider("virtual_cli", "virtual_cli_probe")
    raw_provider = _CatalogProvider("raw_shell", "raw_shell_probe")
    mcp_provider = _CatalogProvider("mcp", "mcp_probe")
    library_provider = LibraryToolProvider(object())
    library_authority = library_provider.issue_builtin_authority(
        reserved_names=LIBRARY_RESERVED_TOOL_NAMES,
        assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
    )
    profile_provider = _CatalogProvider("profile", "profile_probe")
    registry, allowed, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed(
            {
                "available_skills": [
                    {
                        "name": "skill_probe",
                        "description": "skill composition probe",
                        "argument_hint": "optional probe text",
                    }
                ]
            },
            local_provider=local_provider,
            virtual_cli_provider=virtual_provider,
            raw_shell_provider=raw_provider,
            mcp_provider=mcp_provider,
            library_provider=library_provider,
            library_authority=library_authority,
            profile_provider=profile_provider,
        )
    )
    catalog = registry.list_catalog()
    schemas = [registry.load_schema(entry.id) for entry in catalog]
    assert {entry.source for entry in catalog} >= {
        "builtin",
        "local",
        "virtual_cli",
        "raw_shell",
        "library",
        "profile",
        "skill",
        "mcp",
    }
    return catalog, schemas, allowed


def _assert_sinks_are_terminal_free(
    sinks: Mapping[str, list[object]],
    projection: TerminalProjection,
    *,
    expected_sinks: tuple[str, ...],
) -> None:
    markers = (projection.session_id, SESSION_NAME, START_DIRECTORY_MARKER)
    assert set(sinks) == set(expected_sinks)
    for sink_name, payloads in sinks.items():
        assert payloads, f"{sink_name} was not reached"
        assert not _contains_terminal_material(payloads, markers), (
            f"persistent Terminal material reached {sink_name}"
        )


def _real_boundary_payloads(
    tmp_path: Path,
    *,
    manager: TerminalSessionManager,
    projection: TerminalProjection,
) -> dict[str, object]:
    """Exercise each concrete sink while the app-owned session is live."""
    assert manager.projection(projection.session_id) is projection
    catalog, schemas, allowed_tools = _production_model_catalog()

    permission_store = MCPPermissionStore(tmp_path / "mcp-permissions.json")
    permission_store.set_global_default("deny")

    chat_store = ConsoleChatStore()
    session = chat_store.ensure_session()
    chat_store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="ordinary conversation boundary probe",
    )
    controller = ConsoleChatController(
        store=chat_store,
        provider_gateway=object(),  # provider execution is not used here
        agent_runtime_enabled=False,
    )
    provider_messages = controller._provider_messages_for_session(session.id)

    agent_runs = AgentRunsDB(tmp_path / "agent-runs.sqlite", client_id="privacy")
    try:
        run_rows = agent_runs.list_runs("ordinary-conversation")
    finally:
        agent_runs.close()

    run_log_root = tmp_path / "run-log-root"
    run_log = RunLogWriter(root=run_log_root)
    run_log.bind("ordinary-run")
    assert run_log.append(
        run_id="ordinary-run",
        kind="primary",
        type="model",
        content="ordinary run-log boundary probe",
    ) == 1
    run_log.close()
    run_log_payloads = [
        path.read_bytes()
        for path in sorted(run_log_root.rglob("*"))
        if path.is_file()
    ]
    assert run_log_payloads

    exchange_export = project_exchange_export(
        ExchangeCapture(
            run_tag="ordinary-export",
            seq=1,
            created_at="2026-09-01T00:00:00Z",
            provider="test-provider",
            model="test-model",
            endpoint=None,
            request={"messages_payload": [{"role": "user", "content": "hello"}]},
            response={"content": "ordinary response"},
            status="complete",
            usage_json=json.dumps({"input_tokens": 1, "output_tokens": 1}),
            omitted_keys=(),
            capture_detail=CaptureDetail.SAFE,
        ),
        TraceExportProfile.SAFE_SUMMARY,
    )

    workspace_db_path = tmp_path / "workspaces.sqlite"
    workspace_db = WorkspaceDB(workspace_db_path, client_id="privacy")
    workspace_registry = LocalWorkspaceRegistryService(workspace_db)
    try:
        workspace_registry.create_workspace(
            workspace_id="ordinary-workspace",
            name="Ordinary workspace",
        )
        workspace_rows = workspace_registry.list_workspaces()
    finally:
        workspace_db.close()

    screen_state_store = ScreenStateStore()
    runtime_identity = RuntimeIdentity(active_source="local")
    screen_state_store.save(
        "chat",
        {"native_console_state": {"version": 1, "ordinary": True}},
        runtime_identity,
    )
    app_snapshot = screen_state_store.restore("chat", runtime_identity)
    assert app_snapshot is not None

    durable_paths = (
        tmp_path / "mcp-permissions.json",
        tmp_path / "agent-runs.sqlite",
        workspace_db_path,
    )
    database_payloads = [path.read_bytes() for path in durable_paths]
    database_payloads.extend(run_log_payloads)

    payloads = {
        "model_tool_schemas": {"schemas": schemas, "allowed": allowed_tools},
        "model_tool_catalog": catalog,
        "mcp_permission_storage": permission_store.load(),
        "provider_messages": provider_messages,
        "provider_history": list(provider_messages),
        "console_conversation_messages": chat_store.messages_for_session(session.id),
        "agent_runs": {"query_reached": True, "rows": run_rows},
        "run_logs": run_log_payloads,
        "exports": exchange_export,
        "workspace_persistence": workspace_rows,
        "config": load_settings(),
        "database_writes": database_payloads,
        "app_snapshot": app_snapshot,
        "workspace_snapshot": {
            "active_workspace_id": chat_store.workspace_context.active_workspace_id,
            "persisted_rows": workspace_rows,
        },
        "conversation_snapshot": {
            "session": session,
            "messages": chat_store.messages_for_session(session.id),
        },
        "crash_recovery_snapshot": chat_store.dispatch_recovery_for_session(
            session.id
        ),
    }
    assert manager.projection(projection.session_id) is projection
    return payloads


def _concrete_boundary_sinks(
    tmp_path: Path,
    *,
    manager: TerminalSessionManager,
    projection: TerminalProjection,
) -> dict[str, list[object]]:
    concrete = _real_boundary_payloads(
        tmp_path,
        manager=manager,
        projection=projection,
    )
    assert set(concrete) == set(CONCRETE_SINKS)
    return {sink_name: [concrete[sink_name]] for sink_name in CONCRETE_SINKS}


def _fake_boundary_sinks() -> dict[str, list[object]]:
    """Reach every forbidden seam for the mutation-sensitive negative control."""
    return {
        sink_name: [{"fake_boundary_probe_reached": sink_name}]
        for sink_name in FORBIDDEN_SINKS
    }


def _create_live_session(
    tmp_path: Path,
) -> tuple[TerminalSessionManager, TerminalProjection, list[tuple[TerminalProjection, ...]]]:
    start_directory = tmp_path / START_DIRECTORY_MARKER
    start_directory.mkdir()
    manager = TerminalSessionManager(
        lambda: True,
        _Backend,
        screen_model_factory=lambda _columns, _rows: object(),
    )
    local_render_snapshots: list[tuple[TerminalProjection, ...]] = []
    manager.subscribe(lambda: local_render_snapshots.append(manager.projections()))
    assert manager.arm(acknowledge_disclosure=True).armed is True
    result = manager.create_session(
        TerminalLaunchRequest(
            name=SESSION_NAME,
            shell="default",
            start_directory=str(start_directory),
            columns=80,
            rows=24,
        )
    )
    assert result.admitted is True
    assert result.projection is not None
    assert local_render_snapshots
    assert any(result.projection in snapshot for snapshot in local_render_snapshots)
    return manager, result.projection, local_render_snapshots


def test_terminal_imports_and_runtime_ownership_stay_in_local_ui_layers() -> None:
    importers: set[str] = set()
    for path in PRODUCT_ROOT.rglob("*.py"):
        if path.is_relative_to(TERMINAL_ROOT):
            continue
        source = path.read_text(encoding="utf-8")
        if "Terminal" not in source or not _imports_terminal(path, source):
            continue
        importers.add(path.relative_to(REPO_ROOT).as_posix())

    assert importers == TERMINAL_IMPORTERS
    assert _files_referencing_manager_owner() == TERMINAL_RUNTIME_OWNERS
    assert _manager_construction_sites() == {"app.py": 1}


def test_manager_constructor_scan_detects_an_extra_owner(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text(
        "TerminalSessionManager(lambda: True, object)\n",
        encoding="utf-8",
    )
    nested = tmp_path / "UI" / "Screens"
    nested.mkdir(parents=True)
    (nested / "chat_screen.py").write_text(
        "import tldw_chatbook.Terminal.session_manager as sessions\n"
        "sessions.TerminalSessionManager(lambda: True, object)\n",
        encoding="utf-8",
    )

    assert _manager_construction_sites(tmp_path) == {
        "UI/Screens/chat_screen.py": 1,
        "app.py": 1,
    }


@pytest.mark.parametrize(
    "source",
    (
        "from tldw_chatbook import Terminal\n",
        "from .. import Terminal\n",
    ),
)
def test_terminal_import_scan_detects_package_imports(source: str) -> None:
    assert _imports_terminal(Path("mutation.py"), source) is True


def test_terminal_package_is_not_registered_as_a_model_tool() -> None:
    catalog, schemas, allowed_tools = _production_model_catalog()
    catalog_source = (PRODUCT_ROOT / "Agents" / "tool_catalog.py").read_text(
        encoding="utf-8"
    )

    assert catalog
    assert schemas
    assert "TerminalSessionManager" not in catalog_source
    assert "console_terminal" not in catalog_source
    _assert_no_terminal_model_tool(catalog, {"schemas": schemas, "allowed": allowed_tools})
    assert not _contains_terminal_material(
        {"catalog": catalog, "schemas": schemas},
        (SESSION_NAME, START_DIRECTORY_MARKER),
    )


def test_model_tool_guard_rejects_a_realistic_terminal_registration() -> None:
    with pytest.raises(AssertionError, match="terminal"):
        _assert_no_terminal_model_tool(
            [{"id": "builtin:read_terminal", "name": "read_terminal"}],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "read_terminal",
                        "description": "Read a persistent Terminal session.",
                    },
                }
            ],
        )


@pytest.mark.asyncio
async def test_app_owned_terminal_session_leaves_live_console_boundaries_free(
    tmp_path: Path,
) -> None:
    app = _build_test_app(
        config_overrides={"console": {"raw_cli_permitted": True}}
    )
    _configure_native_ready_console(app)
    app.terminal_session_manager.finalize_shutdown()
    manager, projection, local_render_snapshots = _create_live_session(tmp_path)
    app.terminal_session_manager = manager
    try:
        async with app.run_test(size=(160, 48)) as pilot:
            chat = ChatScreen(app)
            await app.push_screen(chat)
            app._initial_screen_pushed = True
            app.current_tab = "chat"
            await _wait_for_selector(chat, pilot, "#console-native-composer")

            chat_store = chat._ensure_console_chat_store()
            session = chat_store.ensure_session()
            chat_store.append_message(
                session.id,
                role=ConsoleMessageRole.USER,
                content="mounted conversation boundary probe",
            )
            controller = chat._ensure_console_chat_controller()
            provider_messages = controller._provider_messages_for_session(session.id)
            app_snapshot = chat.save_state()
            runtime_identity = app._current_runtime_identity()
            app.screen_state_store.save("chat", app_snapshot, runtime_identity)

            sinks = _concrete_boundary_sinks(
                tmp_path,
                manager=manager,
                projection=projection,
            )
            sinks.update(
                {
                    "provider_messages": [provider_messages],
                    "provider_history": [list(provider_messages)],
                    "console_conversation_messages": [
                        chat_store.messages_for_session(session.id)
                    ],
                    "config": [app.app_config],
                    "app_snapshot": [app_snapshot],
                    "workspace_snapshot": [chat_store.workspace_context],
                    "conversation_snapshot": [
                        {
                            "session": session,
                            "messages": chat_store.messages_for_session(session.id),
                        }
                    ],
                    "crash_recovery_snapshot": [
                        chat_store.dispatch_recovery_for_session(session.id)
                    ],
                }
            )
            _assert_sinks_are_terminal_free(
                sinks,
                projection,
                expected_sinks=CONCRETE_SINKS,
            )
            assert not _contains_terminal_material(
                app.screen_state_store.restore("chat", runtime_identity),
                (projection.session_id, SESSION_NAME, START_DIRECTORY_MARKER),
            )
            assert any(
                snapshot and snapshot[0].name == SESSION_NAME
                for snapshot in local_render_snapshots
            ), "safe local render projections may contain the visible session name"
    finally:
        manager.disarm()
        assert manager.wait_for_cleanup(projection.session_id, timeout_seconds=1)
        manager.finalize_shutdown()


def test_privacy_guard_fails_for_projection_in_every_sink(tmp_path: Path) -> None:
    sinks = _fake_boundary_sinks()
    manager, projection, _local_render_snapshots = _create_live_session(tmp_path)
    try:
        _assert_sinks_are_terminal_free(
            sinks,
            projection,
            expected_sinks=FORBIDDEN_SINKS,
        )
        for sink_name in FORBIDDEN_SINKS:
            sinks[sink_name].append(projection)
            with pytest.raises(AssertionError, match=re.escape(sink_name)):
                _assert_sinks_are_terminal_free(
                    sinks,
                    projection,
                    expected_sinks=FORBIDDEN_SINKS,
                )
            assert sinks[sink_name].pop() is projection
            _assert_sinks_are_terminal_free(
                sinks,
                projection,
                expected_sinks=FORBIDDEN_SINKS,
            )

        sinks["exports"].append(dataclasses.asdict(projection))
        with pytest.raises(AssertionError, match="exports"):
            _assert_sinks_are_terminal_free(
                sinks,
                projection,
                expected_sinks=FORBIDDEN_SINKS,
            )
    finally:
        manager.disarm()
        assert manager.wait_for_cleanup(projection.session_id, timeout_seconds=1)
        manager.finalize_shutdown()
