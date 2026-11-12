"""Stock preflight permission reads retain native custody off the input loop."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import sys
import threading
from functools import partial
from types import MethodType, SimpleNamespace

import pytest
import pytest_asyncio

from Tests.Chat import test_console_configuration_capture as source_controls
from Tests.Chat.test_console_configuration_worker_lifetime import _until
from Tests.Chat.test_console_light_surface_context_meter import _Captured, _openai
from Tests.Chat.test_console_light_tool_surface import LIGHT, fresh_profile_gates
from tldw_chatbook.Agents.plain_chat_tool_surface import (
    light_surface_configured,
    tool_kill_switch_engaged,
)
from tldw_chatbook.Backup_Recovery import raw_participants, storage_admission
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_context_compaction import (
    EffectiveMemoryKind,
    EffectiveMemoryResult,
)
from tldw_chatbook.Chat.console_runtime import _native_tools_enabled_for_app
from tldw_chatbook.Chat.console_tool_surface_preview import ToolSurfacePreview
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.Utils.token_counter import ContextWindowResolution

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]
catalog_store = source_controls.catalog_store
local_root = source_controls.local_root
mcp_sources = source_controls.mcp_sources
snapshot_case = source_controls.snapshot_case
runtime_case = source_controls.runtime_case


class _OriginalPermissionRead:
    """Hold the original payload body without replacing checked callables."""

    def __init__(self, permission, *, fail=False):
        self.permission = permission
        self.code = inspect.unwrap(MCPPermissionStore._load_locked).__code__
        self.entered = threading.Event()
        self.release = threading.Event()
        self.returned = threading.Event()
        self.thread = None
        self.calls = 0
        self.leases = []
        self.timed_out = False
        self.fail = fail

    def start(self, code, _offset):
        frame = sys._getframe(1)
        if frame.f_locals.get("self") is not self.permission:
            return
        self.calls += 1
        if self.entered.is_set():
            return
        self.thread = threading.current_thread()
        with storage_admission._changed:
            for state in tuple(raw_participants._states.values()):
                if state.source is self.permission:
                    self.leases.extend(state.leases)
        self.entered.set()
        self.timed_out = not self.release.wait(8)
        if self.fail:
            raise OSError("original permission read unavailable")

    def finish(self, code, _offset, _result):
        if sys._getframe(1).f_locals.get("self") is self.permission:
            self.returned.set()

    @contextlib.contextmanager
    def installed(self):
        monitor = sys.monitoring
        slot = next(value for value in range(6) if monitor.get_tool(value) is None)
        monitor.use_tool_id(slot, "preflight-original-permission")
        events = monitor.events.PY_START | monitor.events.PY_RETURN
        try:
            monitor.register_callback(slot, monitor.events.PY_START, self.start)
            monitor.register_callback(slot, monitor.events.PY_RETURN, self.finish)
            monitor.set_local_events(slot, self.code, events)
            yield self
        finally:
            self.release.set()
            monitor.set_local_events(slot, self.code, 0)
            monitor.register_callback(slot, monitor.events.PY_START, None)
            monitor.register_callback(slot, monitor.events.PY_RETURN, None)
            monitor.free_tool_id(slot)

    def assert_retired(self):
        assert self.returned.is_set() and not self.timed_out
        assert self.leases, "the original permission body had no native admission"
        with storage_admission._changed:
            assert all(
                lease not in storage_admission._live_leases for lease in self.leases
            )


@pytest_asyncio.fixture
async def preview_case(runtime_case, monkeypatch):
    from tldw_chatbook import config

    case = runtime_case
    assert config.save_settings_to_cli_config(fresh_profile_gates())
    case.app.app_config = {"console": {"native_tool_calls": True}}
    bridge = ConsoleAgentBridge(
        agent_runs_db=SimpleNamespace(),
        store=case.store,
        provider_gateway=case.controller.provider_gateway,
        native_tools_enabled=partial(_native_tools_enabled_for_app, case.app),
        light_tool_surface=light_surface_configured,
        tool_kill_switch=partial(tool_kill_switch_engaged, case.app),
    )
    case.bridge = bridge
    case.resolution = _openai(ContextWindowResolution(1_047_576, "model catalog", True))
    case.messages = [{"role": "user", "content": "Reply briefly"}]
    case.prepared = []

    def prepare(_resolution, messages, *, tools=(), **_kwargs):
        case.prepared.append((messages, list(tools), threading.current_thread()))
        raise _Captured

    with monkeypatch.context() as patch:
        patch.setattr(case.controller, "_agent_bridge", bridge)
        patch.setattr(case.controller, "_context_repository", object())
        patch.setattr(case.controller, "_compaction_service", object())
        patch.setattr(
            case.controller.provider_gateway,
            "prepare_chat_request",
            prepare,
            raising=False,
        )
        patch.setattr(case.session, "ephemeral", False)
        patch.setattr(
            case.controller, "_durable_context_snapshots", lambda *_a, **_k: (object(),)
        )
        patch.setattr(
            case.controller,
            "_select_session_effective_memory",
            lambda *_a: EffectiveMemoryResult(EffectiveMemoryKind.RAW),
        )
        yield case


async def _preflight(case, phase="capacity"):
    case.session.persisted_conversation_id = (
        "preview-history" if phase == "history" else None
    )
    return await case.controller._apply_conversation_memory_preflight(
        session_id=case.session.id,
        resolution=case.resolution,
        provider_messages=case.messages,
        assistant_message_id="",
        agent_tools_enabled=True,
        assessment_sink=lambda *_args: None,
    )


@pytest.mark.parametrize("phase", ["capacity", "history"])
@pytest.mark.parametrize("engaged", [False, True])
async def test_original_preflight_permission_read_preserves_loop_progress_and_surface(
    preview_case,
    phase,
    engaged,
):
    case = preview_case
    case.permissions.set_kill_switch(engaged)
    expected = case.bridge.preview_tool_surface(
        case.resolution, session_id=case.session.id, messages=case.messages
    )
    probe = _OriginalPermissionRead(case.permissions)
    loop = asyncio.get_running_loop()
    loop_thread = threading.current_thread()
    progressed = threading.Event()
    stop = threading.Event()
    while_held = []

    def progress():
        while_held.append(not probe.release.is_set())
        progressed.set()

    def release_independently():
        try:
            while not probe.entered.wait(0.01):
                if stop.is_set():
                    return
            loop.call_soon_threadsafe(progress)
            progressed.wait(0.5)
        finally:
            probe.release.set()

    releaser = threading.Thread(target=release_independently)
    try:
        with probe.installed():
            releaser.start()
            with pytest.raises(_Captured):
                await _preflight(case, phase)
            await asyncio.sleep(0)
        assert probe.entered.is_set()
        assert probe.thread is not loop_thread, (
            "permission admission still blocks the input loop"
        )
        assert while_held == [True]
        probe.assert_retired()
        assert (
            not case.controller._preparation_reads
            and not case.runtime._preparation_reads
        )
        messages, tools, thread = case.prepared.pop()
        assert thread is loop_thread
        assert messages == expected.request_messages(case.messages)
        assert tools == list(expected.tools)
        assert {row["function"]["name"] for row in tools} == (
            LIGHT - {"ask_user"} if engaged else LIGHT
        )
    finally:
        stop.set()
        probe.release.set()
        releaser.join(timeout=2)
        assert not releaser.is_alive()


@pytest.mark.parametrize("interruption", ["cancel", "source-change"])
async def test_preflight_permission_read_retires_before_interruption(
    preview_case, interruption
):
    case = preview_case
    probe = _OriginalPermissionRead(case.permissions)
    task = None
    with probe.installed():
        try:
            task = asyncio.create_task(_preflight(case, "history"))
            assert await _until(lambda: probe.entered.is_set() or task.done())
            assert probe.thread is not threading.current_thread(), (
                "native read must leave the loop"
            )
            assert (
                case.controller._preparation_reads and case.runtime._preparation_reads
            )
            if interruption == "cancel":
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
            else:
                case.app.unified_mcp_service = SimpleNamespace(
                    get_kill_switch=lambda: False
                )
            await asyncio.sleep(0)
            assert not task.done() and not probe.returned.is_set()
            probe.release.set()
            with pytest.raises(
                asyncio.CancelledError if interruption == "cancel" else RecoveryRequired
            ):
                await task
            probe.assert_retired()
            assert not case.prepared
            assert (
                not case.controller._preparation_reads
                and not case.runtime._preparation_reads
            )
        finally:
            probe.release.set()
            case.app.unified_mcp_service = case.service
            if task is not None:
                with contextlib.suppress(
                    asyncio.CancelledError, _Captured, RecoveryRequired
                ):
                    await task


@pytest.mark.parametrize(
    "target",
    ["preview", "assessor", "native", "light", "kill", "service", "permission"],
)
async def test_custom_preflight_callbacks_keep_signature_and_loop_affinity(
    preview_case, monkeypatch, target
):
    case = preview_case
    seen = []
    loop = asyncio.get_running_loop()
    thread = threading.current_thread()

    def observed():
        seen.append((threading.current_thread(), asyncio.get_running_loop()))

    if target == "preview":

        def preview(resolution, *, session_id, messages, turn_skill_bindings=()):
            observed()
            return ToolSurfacePreview()

        monkeypatch.setattr(case.bridge, "preview_tool_surface", preview)
    elif target == "assessor":

        def assess(
            *,
            session_id,
            owner,
            resolution,
            provider_messages,
            prepare,
            agent_tools_enabled,
            assessment_sink,
            turn_skill_bindings=(),
        ):
            observed()
            raise _Captured

        monkeypatch.setattr(case.controller, "_assess_request_capacity_only", assess)
    else:

        def callback():
            observed()
            return target != "kill"

        if target == "service":
            monkeypatch.setattr(case.service, "get_kill_switch", callback)
        elif target == "permission":

            def load():
                observed()
                return {"kill_switch": False}

            monkeypatch.setattr(case.permissions, "load", load)
        else:
            monkeypatch.setattr(
                case.bridge,
                {
                    "native": "_native_tools_enabled",
                    "light": "_light_tool_surface",
                    "kill": "_tool_kill_switch",
                }[target],
                callback,
            )
    with pytest.raises(_Captured):
        await _preflight(case)
    assert seen == [(thread, loop)]


async def test_each_preflight_reads_the_switch_again_off_on_off(preview_case):
    case = preview_case
    observations = []
    for engaged in (False, True, False):
        case.permissions.set_kill_switch(engaged)
        probe = _OriginalPermissionRead(case.permissions)
        probe.release.set()
        with probe.installed(), pytest.raises(_Captured):
            await _preflight(case)
        assert probe.calls == 1 and probe.thread is not threading.current_thread()
        observations.append({row["function"]["name"] for row in case.prepared.pop()[1]})
    assert observations == [LIGHT, LIGHT - {"ask_user"}, LIGHT]


async def test_full_surface_preflight_does_not_read_light_kill_switch(
    preview_case, monkeypatch
):
    case = preview_case
    seen = []
    thread = threading.current_thread()

    def native():
        seen.append(("native", threading.current_thread()))
        return True

    def full():
        seen.append(("light", threading.current_thread()))
        return False

    monkeypatch.setattr(case.bridge, "_native_tools_enabled", native)
    monkeypatch.setattr(case.bridge, "_light_tool_surface", full)
    probe = _OriginalPermissionRead(case.permissions)
    probe.release.set()
    with probe.installed(), pytest.raises(_Captured):
        await _preflight(case)
    assert probe.calls == 0
    assert seen == [("native", thread), ("light", thread)]


async def test_preflight_preserves_original_switch_failure_policy(
    preview_case, monkeypatch
):
    case = preview_case
    original_callback = case.bridge._tool_kill_switch

    def unavailable():
        raise OSError("permission read unavailable")

    for service, expected in (
        (SimpleNamespace(), LIGHT),
        (SimpleNamespace(get_kill_switch=None), LIGHT),
        (SimpleNamespace(get_kill_switch=unavailable), LIGHT - {"ask_user"}),
    ):
        monkeypatch.setattr(case.app, "unified_mcp_service", service)
        with pytest.raises(_Captured):
            await _preflight(case)
        assert {row["function"]["name"] for row in case.prepared.pop()[1]} == expected
    monkeypatch.setattr(case.app, "unified_mcp_service", case.service)
    monkeypatch.setattr(case.bridge, "_tool_kill_switch", unavailable)
    with pytest.raises(_Captured):
        await _preflight(case)
    assert {row["function"]["name"] for row in case.prepared.pop()[1]} == LIGHT - {
        "ask_user"
    }
    monkeypatch.setattr(case.bridge, "_tool_kill_switch", original_callback)
    direct = _OriginalPermissionRead(case.permissions, fail=True)
    direct.release.set()
    with direct.installed():
        expected = case.bridge.preview_tool_surface(
            case.resolution, session_id=case.session.id, messages=case.messages
        )
    owned = _OriginalPermissionRead(case.permissions, fail=True)
    owned.release.set()
    with owned.installed(), pytest.raises(_Captured):
        await _preflight(case)
    assert direct.calls == owned.calls == 1
    assert direct.thread is threading.current_thread()
    assert owned.thread is not threading.current_thread()
    assert case.prepared.pop()[1] == list(expected.tools)
    with storage_admission._changed:
        assert all(
            lease not in storage_admission._live_leases
            for lease in [*direct.leases, *owned.leases]
        )


def _minimal_preflight_controller(name):
    """No native source is needed to test callback routing and headless fallback."""
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_context_compaction import ConsoleCompactionPreflight

    controller = ConsoleChatController.__new__(ConsoleChatController)
    owner = SimpleNamespace(
        id="minimal", ephemeral=False, persisted_conversation_id=None
    )
    seen = []

    def preview(resolution, *, session_id, messages, turn_skill_bindings=()):
        return ToolSurfacePreview(tools=({"name": name},))

    def prepare(resolution, messages, *, tools, **kwargs):
        seen.extend(tools)
        raise _Captured

    controller.app = SimpleNamespace()
    controller._agent_bridge = SimpleNamespace(preview_tool_surface=preview)
    controller.store = SimpleNamespace(sessions=lambda: [owner])
    controller.provider_gateway = SimpleNamespace(prepare_chat_request=prepare)
    dependencies = {
        key: (lambda: None)
        for key in inspect.signature(ConsoleCompactionPreflight).parameters
    }
    dependencies.update(
        read_controller__agent_bridge=lambda: controller._agent_bridge,
        read_controller__preflight_tool_surface=lambda: (
            controller._preflight_tool_surface
        ),
        read_controller__assess_request_capacity_only=lambda: (
            controller._assess_request_capacity_only
        ),
        read_controller__context_repository=lambda: object(),
        read_controller__compaction_service=lambda: object(),
        read_controller_provider_gateway=lambda: controller.provider_gateway,
        read_controller_store=lambda: controller.store,
    )
    controller._compaction_preflight = ConsoleCompactionPreflight(**dependencies)
    return controller, seen


async def _minimal_preflight(controller):
    return await controller._apply_conversation_memory_preflight(
        session_id="minimal",
        resolution=None,
        provider_messages=[{"role": "user", "content": "."}],
        assistant_message_id="",
        agent_tools_enabled=True,
        assessment_sink=lambda *_args: None,
    )


async def test_headless_preflight_keeps_existing_custom_preview():
    controller, seen = _minimal_preflight_controller("headless")
    controller.app = None
    with pytest.raises(_Captured):
        await _minimal_preflight(controller)
    assert seen == [{"name": "headless"}]


@pytest.mark.parametrize("binding", ["foreign-original", "reader-custom"])
async def test_preflight_never_passes_its_surface_to_another_assessor(binding):
    first, first_seen = _minimal_preflight_controller("first")
    second, second_seen = _minimal_preflight_controller("second")
    if binding == "foreign-original":
        first._assess_request_capacity_only = second._assess_request_capacity_only
    else:

        def custom(
            self,
            *,
            session_id,
            owner,
            resolution,
            provider_messages,
            prepare,
            agent_tools_enabled,
            assessment_sink,
            turn_skill_bindings=(),
        ):
            second_seen.append({"name": "second"})
            raise _Captured

        first._compaction_preflight.read_controller__assess_request_capacity_only = (
            lambda: MethodType(custom, first)
        )
    with pytest.raises(_Captured):
        await _minimal_preflight(first)
    if binding == "foreign-original":
        assert first_seen == [{"name": "second"}]
        assert not second_seen
    else:
        assert not first_seen
        assert second_seen == [{"name": "second"}]


@pytest.mark.parametrize("changed", ["assessor", "gateway", "prepare"])
async def test_preflight_refuses_callback_replacement_during_native_read(
    preview_case, changed
):
    case = preview_case
    probe = _OriginalPermissionRead(case.permissions)
    task = None
    old_gateway = case.controller.provider_gateway
    old_prepare = old_gateway.prepare_chat_request
    old_assessor = case.controller._assess_request_capacity_only

    def unexpected(*args, **kwargs):
        raise AssertionError("changed callback was used")

    with probe.installed():
        try:
            task = asyncio.create_task(_preflight(case))
            assert await _until(lambda: probe.entered.is_set() or task.done())
            assert probe.thread is not threading.current_thread()
            if changed == "assessor":
                case.controller._assess_request_capacity_only = unexpected
            elif changed == "gateway":
                case.controller.provider_gateway = SimpleNamespace(
                    prepare_chat_request=unexpected
                )
            else:
                old_gateway.prepare_chat_request = unexpected
            probe.release.set()
            with pytest.raises(RecoveryRequired):
                await task
            assert not case.prepared
            probe.assert_retired()
        finally:
            probe.release.set()
            case.controller.provider_gateway = old_gateway
            old_gateway.prepare_chat_request = old_prepare
            case.controller._assess_request_capacity_only = old_assessor
            if task is not None:
                with contextlib.suppress(RecoveryRequired, _Captured):
                    await task


async def test_changed_switch_callback_body_stays_on_original_loop(
    preview_case, monkeypatch
):
    case = preview_case
    case.app.preview_callback_threads = []

    def changed(app):
        app.preview_callback_threads.append(__import__("threading").current_thread())
        return True

    monkeypatch.setattr(tool_kill_switch_engaged, "__code__", changed.__code__)
    with pytest.raises(_Captured):
        await _preflight(case)
    assert case.app.preview_callback_threads == [threading.current_thread()]
    assert {row["function"]["name"] for row in case.prepared.pop()[1]} == LIGHT - {
        "ask_user"
    }
