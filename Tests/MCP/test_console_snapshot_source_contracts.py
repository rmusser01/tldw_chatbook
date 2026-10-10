"""Native source callback contracts substituted by Console joined workers."""

import asyncio
import dis
import inspect
import threading
from types import MethodType, SimpleNamespace

import pytest

from Tests.Chat.test_console_async_mcp_snapshot import (
    _MaximumProbe,
    _loop_projection,
)
from Tests.Chat import test_console_async_mcp_snapshot as snapshot_controls
from Tests.MCP import test_external_catalog_worker_ownership as catalog_controls
from Tests.private_profile import private_profile_test
from tldw_chatbook.Backup_Recovery import storage_admission
from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
from tldw_chatbook.MCP.local_store import LocalMCPStore
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.MCP.unified_control_plane_service import (
    UnifiedMCPControlPlaneService,
)


mcp_sources = snapshot_controls.mcp_sources
local_root = snapshot_controls.local_root
catalog_store = snapshot_controls.catalog_store
snapshot_case = snapshot_controls.snapshot_case


class _AdmittedBundleProbe(_MaximumProbe):
    """Hold the original admitted bundle body before its dynamic load lookup."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.code = inspect.unwrap(LocalMCPStore.get_catalog_bundle).__code__
        self.first_entered = threading.Event()

    def observe(self, frame, event, arg):
        super().observe(frame, event, arg)
        if (
            event == "call"
            and frame.f_code is self.permission_code
            and frame.f_locals.get("self") is self.permissions
        ):
            self.first_entered.set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,consumer",
    [
        ("instance", "maximum"),
        ("class", "maximum"),
        ("cold_class", "maximum"),
        ("instance", "catalog"),
        ("class", "local"),
    ],
)
async def test_custom_permission_switch_keeps_original_deny_and_caller_contract(
    snapshot_case, monkeypatch, kind, consumer
):
    case = snapshot_case
    _loop_projection(case)
    loop = asyncio.get_running_loop()
    thread = threading.current_thread()
    observed = []

    def deny(*args):
        observed.append(threading.current_thread())
        if consumer == "catalog":
            assert threading.current_thread() is not thread
            with pytest.raises(RuntimeError):
                asyncio.get_running_loop()
        else:
            assert threading.current_thread() is thread
            assert asyncio.get_running_loop() is loop
        return True

    if kind == "instance":
        monkeypatch.setattr(case.permissions, "get_kill_switch", deny)
    else:
        monkeypatch.setattr(MCPPermissionStore, "get_kill_switch", deny)
        if kind == "cold_class":
            case.service._permission_store = None
    if consumer == "maximum":
        captured = await case.controller.capture_turn_configuration_snapshot(
            case.session.id
        )
        assert not captured.mcp_definition_maximum
    elif consumer == "catalog":
        from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider

        provider = MCPToolProvider(service=case.service, main_loop=loop)
        await provider.compose_catalog()
        assert not provider.list_catalog()
    else:
        from Tests.Chat.test_console_local_review_hook import _test_execution_context

        async def no_catalog(*args, **kwargs):
            return None

        case.controller._compose_mcp_provider = no_catalog
        context = _test_execution_context(
            case.controller._scratch_spaces.snapshot(case.session.id),
            session_id=case.session.id,
            tool_configuration={"local_tools_enabled": True},
        )
        _, _, local, _ = await case.controller._compose_agent_request_providers(
            session_id=case.session.id,
            project_selection=None,
            project_authority_guard=None,
            turn_context=context,
            admitted_roots=(),
        )
        assert local is None
    assert observed
    assert all((item is thread) is (consumer != "catalog") for item in observed)


@pytest.mark.parametrize("kind", ["permission_switch", "catalog", "load", "projection"])
def test_joined_source_rejects_borrowed_actual_receiver(
    snapshot_case, monkeypatch, kind
):
    from tldw_chatbook.MCP.console_snapshot import standard_console_sources

    case = snapshot_case
    assert standard_console_sources(case.service)
    if kind == "permission_switch":
        other = MCPPermissionStore(case.permissions.path)
        target, name = case.permissions, "get_kill_switch"
    elif kind == "projection":
        other = LocalMCPControlService(store=case.source, manifest_provider=lambda: {})
        target, name = case.local, "_project_external_catalog"
    else:
        other = LocalMCPStore(case.source.path)
        target, name = (
            case.source,
            "get_external_catalog" if kind == "catalog" else "load",
        )
    callback = getattr(other, name)
    assert isinstance(callback, MethodType) and callback.__self__ is other
    monkeypatch.setattr(target, name, callback)
    assert not standard_console_sources(case.service)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["instance", "class"])
async def test_custom_native_store_load_preserves_caller_loop(
    snapshot_case, monkeypatch, kind
):
    case = snapshot_case
    _loop_projection(case)
    original = case.source.load
    state = original()
    loop = asyncio.get_running_loop()
    thread = threading.current_thread()
    observed = []

    def caller_load(*args):
        # Instance customization calls the original installed native reader.
        # Class customization supplies a prepared custom view; original outer
        # joined-reader admission and the permission reader remain installed.
        result = original() if kind == "instance" else state
        observed.append(threading.current_thread())
        assert threading.current_thread() is thread
        assert asyncio.get_running_loop() is loop
        return result

    monkeypatch.setattr(
        case.source if kind == "instance" else LocalMCPStore, "load", caller_load
    )
    captured = await case.controller.capture_turn_configuration_snapshot(
        case.session.id
    )
    assert observed and all(item is thread for item in observed)
    assert "local:one::first" in captured.mcp_definition_maximum


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["store_instance", "store_class", "projection_instance", "projection_class"]
)
async def test_custom_external_source_filter_preserves_definition_maximum(
    snapshot_case, monkeypatch, kind
):
    case = snapshot_case
    _loop_projection(case)
    thread = threading.current_thread()
    loop = asyncio.get_running_loop()
    observed = []

    def filtered(*args, **kwargs):
        observed.append(threading.current_thread())
        assert threading.current_thread() is thread
        assert asyncio.get_running_loop() is loop
        return []

    if kind.startswith("store"):
        target = case.source if kind.endswith("instance") else LocalMCPStore
        name = "get_external_catalog"
    else:
        target = case.local if kind.endswith("instance") else LocalMCPControlService
        name = "_project_external_catalog"
    monkeypatch.setattr(target, name, filtered)
    captured = await case.controller.capture_turn_configuration_snapshot(
        case.session.id
    )
    assert "local:one::first" not in captured.mcp_definition_maximum
    assert captured.mcp_definition_maximum
    assert observed and all(item is thread for item in observed)


@pytest.mark.asyncio
async def test_catalog_worker_preserves_custom_filtered_source(
    snapshot_case, monkeypatch
):
    case = snapshot_case
    observed = []

    def filtered():
        observed.append(threading.current_thread())
        return []

    monkeypatch.setattr(case.source, "get_external_catalog", filtered)
    assert await case.service.local_external_catalog() == []
    assert observed == [threading.current_thread()]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["permission_store", "execution_log"])
@private_profile_test
async def test_public_source_descriptor_replaced_before_first_helper_import(
    request, monkeypatch, kind
):
    import sys

    assert "tldw_chatbook.MCP.console_snapshot" not in sys.modules
    case = request.getfixturevalue("snapshot_case")
    assert "tldw_chatbook.MCP.console_snapshot" not in sys.modules
    _loop_projection(case)
    thread = threading.current_thread()
    observed = []
    if kind == "permission_store":
        alternate = MCPPermissionStore(case.permissions.path)
        original = alternate.load

        def narrowed_load():
            result = original()
            observed.append(threading.current_thread())
            assert threading.current_thread() is thread
            result["profiles"]["default"]["global_default"] = "deny"
            return result

        monkeypatch.setattr(alternate, "load", narrowed_load)
        descriptor = property(lambda owner: alternate)
    else:
        case.permissions.set_tool_state(
            "local:one", "first", "allow", definition_hash="0" * 64
        )
        descriptor = property(lambda owner: None)
    monkeypatch.setattr(UnifiedMCPControlPlaneService, kind, descriptor)
    from tldw_chatbook.MCP.console_snapshot import standard_console_sources

    captured = await case.controller.capture_turn_configuration_snapshot(
        case.session.id
    )
    if kind == "permission_store":
        assert not captured.mcp_definition_maximum
        assert observed and all(item is thread for item in observed)
    else:
        assert captured.mcp_definition_maximum
        assert case.permissions.get_tool_entry("local:one", "first")["config_changed"]
        assert case.service._execution_log is None
    assert not standard_console_sources(case.service)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind",
    [
        "permission_switch",
        "permission_switch_descriptor",
        "store_load",
        "permission_descriptor",
        "log_descriptor",
    ],
)
async def test_source_dependency_retarget_during_native_body_is_refused(
    snapshot_case, monkeypatch, kind
):
    case = snapshot_case
    _loop_projection(case)
    observed = []
    probe = _AdmittedBundleProbe(case.source, case.permissions, hold=True)
    with probe.installed():
        pending = asyncio.create_task(
            case.controller.capture_turn_configuration_snapshot(case.session.id)
        )
        try:
            # Qualify real progress at each original source boundary. Both
            # witnesses retain the original 4 s entry bound; no native read,
            # timer, guard or overall application budget is shortened/skipped.
            await catalog_controls._worker_entered(
                SimpleNamespace(
                    entered=probe.first_entered, read_threads=probe.permission_threads
                ),
                pending,
            )
            await catalog_controls._worker_entered(probe, pending)
            if kind == "permission_switch":
                monkeypatch.setattr(case.permissions, "get_kill_switch", lambda: True)
            elif kind == "permission_switch_descriptor":

                def changed_getter(owner):
                    observed.append(threading.current_thread())
                    return lambda: True

                monkeypatch.setattr(
                    MCPPermissionStore, "get_kill_switch", property(changed_getter)
                )
            elif kind == "store_load":
                original = case.source.load

                def replaced_load():
                    observed.append(threading.current_thread())
                    return original()

                monkeypatch.setattr(case.source, "load", replaced_load)
            else:
                name = (
                    "permission_store"
                    if kind == "permission_descriptor"
                    else "execution_log"
                )
                original = getattr(UnifiedMCPControlPlaneService, name)
                monkeypatch.setattr(
                    UnifiedMCPControlPlaneService, name, property(original.fget)
                )
            probe.release.set()
            captured = await pending
            assert (
                not observed
            ), "changed source callback ran before the ownership fence"
            assert not captured.mcp_definition_maximum
            assert all(
                lease not in storage_admission._live_leases for lease in probe.leases
            )
        finally:
            await catalog_controls._settle(pending, probe)


@pytest.mark.asyncio
@pytest.mark.parametrize("kill", [False, True])
async def test_original_joined_source_still_reads_fresh_native_switch(
    snapshot_case, kill
):
    from tldw_chatbook.MCP.console_snapshot import standard_console_sources

    case = snapshot_case
    _loop_projection(case)
    case.permissions.set_kill_switch(kill)
    assert standard_console_sources(case.service)
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        captured = await case.controller.capture_turn_configuration_snapshot(
            case.session.id
        )
    assert bool(captured.mcp_definition_maximum) is not kill
    assert len(probe.permission_calls) == 1
    assert len(probe.read_threads) == (0 if kill else 1)
    assert all(
        thread is not threading.current_thread() for _, thread in probe.permission_calls
    )


@pytest.mark.parametrize("kind", ["foreign", "proxy", "callback"])
def test_private_bundle_reader_rejects_unqualified_captured_load(snapshot_case, kind):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    source = snapshot_case.source
    original = source.load
    if kind == "foreign":
        reader = LocalMCPStore(source.path).load
        assert reader.__self__ is not source
    elif kind == "proxy":

        class LoadProxy:
            __func__ = property(lambda self: original.__func__)
            __self__ = property(lambda self: source)

            def __call__(self):
                return original()

        reader = LoadProxy()
        assert reader.__func__ is original.__func__ and reader.__self__ is source
    else:

        def reader():
            return original()

    with pytest.raises(RecoveryRequired, match="mcp_source_selection_changed"):
        source.get_catalog_bundle(_captured_load=reader)


def test_private_bundle_reader_preserves_original_native_load(snapshot_case):
    source = snapshot_case.source
    probe = catalog_controls._ReadProbe(source)
    with probe.installed():
        bundle = source.get_catalog_bundle(_captured_load=source.load)
    assert len(probe.read_threads) == 1
    assert bundle["profiles"][0]["profile_id"] == "one"
    assert bundle["discovery_snapshots"]["one"] == {"tools": [{"name": "first"}]}
    assert probe.leases and all(
        lease not in storage_admission._live_leases for lease in probe.leases
    )


@pytest.mark.asyncio
async def test_exact_native_custom_catalog_keeps_late_inventory_receiver(
    snapshot_case, monkeypatch
):
    from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider

    case = snapshot_case
    loop = asyncio.get_running_loop()
    caller = threading.current_thread()
    observed = []

    def current_inventory():
        observed.append(threading.current_thread())
        return {"tools": [{"name": "current_inventory"}]}

    current_local = LocalMCPControlService(
        store=case.source, manifest_provider=current_inventory
    )

    async def custom_catalog():
        assert asyncio.get_running_loop() is loop
        assert threading.current_thread() is caller
        await asyncio.sleep(0)
        case.service.local_service = current_local
        return []

    class PermissionCustodyProbe(_MaximumProbe):
        def observe(self, frame, event, arg):
            super().observe(frame, event, arg)
            if (
                event == "call"
                and frame.f_code is self.permission_code
                and frame.f_locals.get("self") is case.permissions
            ):
                with storage_admission._changed:
                    for state in tuple(
                        catalog_controls.raw_participants._states.values()
                    ):
                        if state.source is case.permissions:
                            self.leases.extend(state.leases)

    monkeypatch.setattr(case.service, "local_external_catalog", custom_catalog)
    probe = PermissionCustodyProbe(case.source, case.permissions)
    provider = MCPToolProvider(service=case.service, main_loop=loop)
    with probe.installed():
        await provider.compose_catalog()
    assert observed and all(item is not caller for item in observed)
    assert any("current_inventory" in row.name for row in provider.list_catalog())
    assert probe.permission_calls
    assert probe.leases and all(
        lease not in storage_admission._live_leases for lease in probe.leases
    )


@pytest.mark.asyncio
async def test_standard_catalog_refuses_changed_class_descriptor_before_getter(
    snapshot_case, monkeypatch
):
    from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider

    case = snapshot_case
    _loop_projection(case)
    original = case.service.local_external_catalog
    observed = []
    probe = _AdmittedBundleProbe(case.source, case.permissions, hold=True)
    provider = MCPToolProvider(
        service=case.service, main_loop=asyncio.get_running_loop()
    )
    with probe.installed():
        pending = asyncio.create_task(provider.compose_catalog())
        try:
            await catalog_controls._worker_entered(
                SimpleNamespace(
                    entered=probe.first_entered, read_threads=probe.permission_threads
                ),
                pending,
            )
            await catalog_controls._worker_entered(probe, pending)

            def changed_getter(owner):
                observed.append(threading.current_thread())
                return original

            monkeypatch.setattr(
                UnifiedMCPControlPlaneService,
                "local_external_catalog",
                property(changed_getter),
            )
            probe.release.set()
            error = None
            try:
                await pending
            except catalog_controls.bootstrap.RecoveryRequired as exc:
                error = exc
            assert not observed, "changed catalog getter ran before the static fence"
            assert error is not None and str(error) == "mcp_source_selection_changed"
            assert probe.leases and all(
                lease not in storage_admission._live_leases for lease in probe.leases
            )
        finally:
            await catalog_controls._settle(pending, probe)


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer", ["maximum", "local"])
async def test_unrelated_custom_catalog_descriptor_is_not_read_by_preparation(
    snapshot_case, monkeypatch, consumer
):
    case = snapshot_case
    _loop_projection(case)
    original = case.service.local_external_catalog
    observed = []

    def custom_getter(owner):
        observed.append(threading.current_thread())
        return original

    monkeypatch.setattr(
        UnifiedMCPControlPlaneService, "local_external_catalog", property(custom_getter)
    )
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        if consumer == "maximum":
            captured = await case.controller.capture_turn_configuration_snapshot(
                case.session.id
            )
            assert "local:one::first" in captured.mcp_definition_maximum
        else:
            from Tests.Chat.test_console_local_review_hook import (
                _test_execution_context,
            )

            context = _test_execution_context(
                case.controller._scratch_spaces.snapshot(case.session.id),
                session_id=case.session.id,
                tool_configuration={"local_tools_enabled": True},
            )
            local, review = await case.controller._compose_local_provider_async(
                session_id=case.session.id, turn_context=context, admitted_roots=()
            )
            assert local is not None and callable(review)
    assert not observed, "unrelated custom catalog property was dereferenced"
    assert probe.permission_calls and all(
        thread is not threading.current_thread() for _, thread in probe.permission_calls
    )


class _EmptyMCPCompositionProbe(catalog_controls._ReadProbe):
    """Count original constructor/catalog work without replacing stock code."""

    def __init__(self, case):
        from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider

        super().__init__(case.source)
        self.service = case.service
        self.constructor_code = MCPToolProvider.__init__.__code__
        self.catalog_code = MCPToolProvider._compose_catalog.__code__
        self.catalog_start = next(
            instruction.offset
            for instruction in dis.get_instructions(self.catalog_code)
            if instruction.opname == "RESUME" and instruction.arg == 0
        )
        self.provider_constructions = 0
        self.catalog_preparations = 0

    def observe(self, frame, event, arg):
        if event == "call":
            if (
                frame.f_code is self.constructor_code
                and frame.f_locals.get("service") is self.service
            ):
                self.provider_constructions += 1
            if (
                frame.f_code is self.catalog_code
                and frame.f_lasti == self.catalog_start
            ):
                receiver = frame.f_locals.get("self")
                if getattr(receiver, "_service", None) is self.service:
                    self.catalog_preparations += 1
        super().observe(frame, event, arg)


def _require_stock_empty_route(case):
    from tldw_chatbook.Agents import mcp_tool_provider as provider_source

    composition = provider_source.capture_standard_controller_composition(
        provider_source.MCPToolProvider, case.service
    )
    assert composition is not None, "fixture did not qualify the stock MCP route"


@pytest.mark.asyncio
@pytest.mark.parametrize("publish_counts", [True, False], ids=["dispatch", "preview"])
async def test_empty_stock_mcp_maximum_skips_catalog_work(
    snapshot_case, publish_counts, record_property
):
    """An empty stock bound must avoid construction and real catalog reads."""
    case = snapshot_case
    _loop_projection(case)
    _require_stock_empty_route(case)
    case.app.console_mcp_tool_count = 3
    case.app.console_mcp_not_connected_count = 1
    probe = _EmptyMCPCompositionProbe(case)
    with probe.installed():
        provider = await case.controller._compose_mcp_provider(
            case.session.id,
            publish_counts=publish_counts,
            maximum_tool_ids=frozenset(),
            plugin_maximum=None,
        )
    record_property("mcp_provider_constructions", probe.provider_constructions)
    record_property("mcp_catalog_preparations", probe.catalog_preparations)
    record_property("mcp_catalog_source_reads", len(probe.read_threads))
    assert provider is None
    assert (
        case.app.console_mcp_tool_count,
        case.app.console_mcp_not_connected_count,
    ) == ((None, None) if publish_counts else (3, 1))
    assert probe.provider_constructions == 0
    assert probe.catalog_preparations == 0
    assert probe.read_threads == []


@pytest.mark.asyncio
async def test_empty_mcp_live_clears_previous_counts(snapshot_case):
    case = snapshot_case
    _loop_projection(case)
    _require_stock_empty_route(case)
    with _EmptyMCPCompositionProbe(case).installed() as original:
        populated = await case.controller._compose_mcp_provider(case.session.id)
    assert populated is not None and populated.list_catalog()
    assert original.provider_constructions == 1 and original.read_threads
    assert case.app.console_mcp_tool_count > 0
    with _EmptyMCPCompositionProbe(case).installed() as empty:
        provider = await case.controller._compose_mcp_provider(
            case.session.id, maximum_tool_ids=frozenset()
        )
    assert provider is None
    assert case.app.console_mcp_tool_count is None
    assert case.app.console_mcp_not_connected_count is None
    assert empty.provider_constructions == 0 and not empty.read_threads


class _CustomEmptyMaximum(frozenset):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "maximum,has_provider",
    [
        (None, True),
        (frozenset({"local:one::first"}), True),
        (_CustomEmptyMaximum(), False),
    ],
    ids=["unset", "nonempty", "custom-empty-set"],
)
async def test_empty_mcp_maximum_keeps_ordinary_bounds(
    snapshot_case, maximum, has_provider
):
    case = snapshot_case
    _loop_projection(case)
    _require_stock_empty_route(case)
    with _EmptyMCPCompositionProbe(case).installed() as probe:
        provider = await case.controller._compose_mcp_provider(
            case.session.id, maximum_tool_ids=maximum
        )
    assert (provider is not None) is has_provider
    assert probe.provider_constructions == 1
    assert probe.catalog_preparations == 1 and probe.read_threads
    assert all(lease not in storage_admission._live_leases for lease in probe.leases)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["factory", "service", "factory-defaults"])
async def test_empty_mcp_maximum_keeps_ordinary_fallback(
    snapshot_case, monkeypatch, kind
):
    """Custom and changed inputs must retain original native preparation."""
    from tldw_chatbook.Agents import mcp_tool_provider as provider_source
    from tldw_chatbook.Chat import console_chat_controller as controller_source

    case = snapshot_case
    _loop_projection(case)
    _require_stock_empty_route(case)
    if kind == "factory":

        class CustomProvider(provider_source.MCPToolProvider):
            pass

        monkeypatch.setattr(controller_source, "MCPToolProvider", CustomProvider)
        factory = CustomProvider
    elif kind == "service":

        class CustomService(UnifiedMCPControlPlaneService):
            pass

        case.service = CustomService(
            target_store=None,
            context_store=None,
            local_service=case.local,
            server_service=None,
        )
        case.service._permission_store = case.permissions
        case.app.unified_mcp_service = case.service
        factory = provider_source.MCPToolProvider
    else:
        factory = provider_source.MCPToolProvider
        original = factory.__init__.__kwdefaults__
        assert original is not None
        monkeypatch.setattr(factory.__init__, "__kwdefaults__", dict(original))
    assert (
        provider_source.capture_standard_controller_composition(factory, case.service)
        is None
    )
    with _EmptyMCPCompositionProbe(case).installed() as probe:
        provider = await case.controller._compose_mcp_provider(
            case.session.id, maximum_tool_ids=frozenset()
        )
    assert provider is None
    assert probe.provider_constructions == 1
    assert probe.catalog_preparations == 1 and probe.read_threads


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plugin_maximum", [{}, {"plugin_run_id": "pending:console-mcp"}]
)
async def test_empty_mcp_maximum_preserves_plugin_route(snapshot_case, plugin_maximum):
    case = snapshot_case
    _loop_projection(case)
    _require_stock_empty_route(case)
    assert case.controller._skills_service is None
    # An unavailable service refuses before platform-specific plugin imports.
    expected_error, reason = PermissionError, "plugin_mcp_authority_unavailable"
    with _EmptyMCPCompositionProbe(case).installed() as probe:
        with pytest.raises(expected_error, match=reason):
            await case.controller._compose_mcp_provider(
                case.session.id,
                maximum_tool_ids=frozenset(),
                plugin_maximum=plugin_maximum,
            )
    assert probe.provider_constructions == 1
    assert probe.catalog_preparations == 1
    # Plugin authority is still checked; the empty MCP ceiling needs no external catalog.
    assert probe.read_threads == []
