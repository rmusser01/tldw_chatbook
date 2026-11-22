"""Stock preflight permission reads retain native custody off the input loop."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import sys
import threading
import time
from functools import partial
from types import MethodType, SimpleNamespace

import pytest
import pytest_asyncio

from Tests.Chat import test_console_configuration_capture as source_controls
from Tests.Chat.test_console_configuration_worker_lifetime import _until
from Tests.Chat.test_console_light_surface_context_meter import _Captured, _openai
from Tests.Chat.test_console_light_tool_surface import LIGHT, fresh_profile_gates
from tldw_chatbook.Agents.plain_chat_tool_surface import (
    AppToolKillSwitch,
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
        # Exact stock construction in ConsoleRuntime.ensure_agent_bridge.
        tool_kill_switch=AppToolKillSwitch(case.app),
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
        assert (
            probe.thread is not loop_thread
        ), "permission admission still blocks the input loop"
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
            assert (
                probe.thread is not threading.current_thread()
            ), "native read must leave the loop"
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


async def test_legacy_partial_switch_keeps_owned_permission_read(preview_case):
    case = preview_case
    case.bridge._tool_kill_switch = partial(tool_kill_switch_engaged, case.app)
    probe = _OriginalPermissionRead(case.permissions)
    probe.release.set()
    with probe.installed(), pytest.raises(_Captured):
        await _preflight(case)
    assert probe.calls == 1 and probe.thread is not threading.current_thread()
    probe.assert_retired()


async def test_changed_runtime_switch_sources_keep_original_loop_affinity(
    preview_case, monkeypatch
):
    from tldw_chatbook.Agents import plain_chat_tool_surface

    case = preview_case
    case.app.preview_callback_threads = []

    def changed_call(self):
        self.app.preview_callback_threads.append(
            __import__("threading").current_thread()
        )
        return True

    def changed_reader(app):
        app.preview_callback_threads.append(__import__("threading").current_thread())
        return True

    for changed in ("call-body", "reader-binding"):
        case.app.preview_callback_threads.clear()
        with monkeypatch.context() as patch:
            if changed == "call-body":
                patch.setattr(
                    AppToolKillSwitch.__call__, "__code__", changed_call.__code__
                )
            else:
                patch.setattr(
                    plain_chat_tool_surface, "tool_kill_switch_engaged", changed_reader
                )
            with pytest.raises(_Captured):
                await _preflight(case)
        assert case.app.preview_callback_threads == [threading.current_thread()]
        assert {row["function"]["name"] for row in case.prepared.pop()[1]} == LIGHT - {
            "ask_user"
        }


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


@pytest.mark.parametrize(
    "changed",
    ["assessor", "gateway", "prepare", "switch-owner", "switch-body", "switch-reader"],
)
async def test_preflight_refuses_callback_replacement_during_native_read(
    preview_case, changed, monkeypatch
):
    from tldw_chatbook.Agents import plain_chat_tool_surface

    case = preview_case
    probe = _OriginalPermissionRead(case.permissions)
    task = None
    old_gateway = case.controller.provider_gateway
    old_prepare = old_gateway.prepare_chat_request
    old_assessor = case.controller._assess_request_capacity_only
    switch = case.bridge._tool_kill_switch

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
            elif changed == "prepare":
                old_gateway.prepare_chat_request = unexpected
            elif changed == "switch-owner":
                object.__setattr__(switch, "app", SimpleNamespace())
            elif changed == "switch-body":

                def changed_call(self):
                    return False

                monkeypatch.setattr(
                    AppToolKillSwitch.__call__, "__code__", changed_call.__code__
                )
            else:
                monkeypatch.setattr(
                    plain_chat_tool_surface,
                    "tool_kill_switch_engaged",
                    lambda app: False,
                )
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
            object.__setattr__(switch, "app", case.app)
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


# -- TASK-33620.15.52: a preview answers from a stamp-proven switch read -------


@pytest.fixture
def run_start_tools(monkeypatch):
    """The proof's module, emptied; store files count as settled at once.

    The fixture writes the switch moments before each test, and a read is kept
    only when the permission file was last changed ``SETTLE_NS`` before it
    (the rule TASK-33620.15.38's anchor follows). The tests that are not about
    that rule set it to 0, as the catalog reuse tests do.
    """
    from tldw_chatbook.Chat import console_run_start_tools as tools

    tools.forget()
    monkeypatch.setattr(tools, "SETTLE_NS", 0)
    yield tools
    tools.forget()


async def _counted_preflight(case, *, fail=False, phase="capacity"):
    """One preflight: its probe of the original permission body, and its tools."""
    probe = _OriginalPermissionRead(case.permissions, fail=fail)
    probe.release.set()
    with probe.installed(), pytest.raises(_Captured):
        await _preflight(case, phase)
    if probe.calls and not fail:
        probe.assert_retired()
    return probe, {row["function"]["name"] for row in case.prepared.pop()[1]}


@pytest.mark.parametrize("phase", ["capacity", "history"])
@pytest.mark.parametrize("engaged", [False, True])
async def test_a_preview_whose_switch_read_is_proven_current_reads_no_store(
    preview_case, run_start_tools, phase, engaged
):
    """The next preview answers from the last read, on the loop, with no hop.

    TASK-33620.15.52 (owner, 2026-11-06: a stamp-proven read, not a worker
    hop). The first preview reads the switch through the preparation worker.
    Nothing it read changes before the second: the permission file keeps its
    stamp, the same store stays selected, no admission state is written. The
    second preview answers the same surface without entering the store's
    permission body at all, so without a hop either (origin/dev read it again
    through the hop for every preview).
    """
    case = preview_case
    case.permissions.set_kill_switch(engaged)
    expected = LIGHT - {"ask_user"} if engaged else LIGHT
    first, surface = await _counted_preflight(case, phase=phase)
    assert first.calls == 1 and first.thread is not threading.current_thread()
    assert surface == expected
    second, surface = await _counted_preflight(case, phase=phase)
    assert second.calls == 0, (
        "the preview read the store again though nothing its last read saw had changed"
    )
    assert surface == expected


def _turn_switch_on(case):
    case.permissions.set_kill_switch(True)


def _put_bytes_back(case):
    """Rewrite the same bytes until the file's ``st_ctime_ns`` has moved.

    The tests set ``SETTLE_NS`` to 0, so on a file system with a coarse clock
    one rewrite could keep the stamp; the case rewrites (bounded) until the
    stamp the proof compares has changed, and skips if it never does.
    """
    path = case.permissions.path
    data = path.read_bytes()
    before = os.stat(path, follow_symlinks=False).st_ctime_ns
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        path.write_bytes(data)
        if os.stat(path, follow_symlinks=False).st_ctime_ns != before:
            return
        time.sleep(0.02)
    pytest.skip("st_ctime_ns did not move within 5 s of rewrites on this file system")


def _publish_config(case):
    """A real config save: the store's installed identity names the config."""
    from tldw_chatbook import config

    before = config.current_config_identity()
    assert config.save_settings_to_cli_config(fresh_profile_gates())
    assert config.current_config_identity() != before


def _write_admission_state(case):
    from tldw_chatbook.Backup_Recovery import bootstrap

    bootstrap.advance_admission_epoch()


def _select_another_store(case):
    selected = case.service._permission_store
    again = MCPPermissionStore(selected.path)
    assert again is not selected and again.path == selected.path
    case.service._permission_store = again


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (_turn_switch_on, LIGHT - {"ask_user"}),
        (_put_bytes_back, LIGHT),
        (_write_admission_state, LIGHT),
        (_select_another_store, LIGHT),
        (_publish_config, LIGHT),
    ],
    ids=[
        "switch-on",
        "bytes-put-back",
        "admission-write",
        "another-store",
        "config-publication",
    ],
)
async def test_a_change_the_proof_covers_sends_the_next_preview_to_a_real_read(
    preview_case, run_start_tools, change, expected
):
    """Each change a kept switch read cannot vouch for is read for real.

    A write to the permission file changes its stamp, the bytes put back
    included (``st_ctime_ns`` moves); an admission or activation write moves
    the admission epoch; another store over the same file is another source;
    a config publication changes the store's installed identity (it names the
    config) without moving the epoch.
    The next preview reads through the hop again and answers what the store
    now says. (Negative controls in the task notes: dropping the stamp, epoch
    or source comparison from the proof turns its case red.)
    """
    case = preview_case
    selected = case.service._permission_store
    try:
        first, surface = await _counted_preflight(case)
        assert first.calls == 1 and surface == LIGHT
        change(case)
        probe = _OriginalPermissionRead(case.service._permission_store)
        probe.release.set()
        with probe.installed(), pytest.raises(_Captured):
            await _preflight(case)
        assert probe.calls == 1, "a kept switch read answered after its input changed"
        assert probe.thread is not threading.current_thread()
        assert {row["function"]["name"] for row in case.prepared.pop()[1]} == expected
    finally:
        case.service._permission_store = selected


async def test_a_switch_read_of_a_just_written_store_is_not_kept(
    preview_case, run_start_tools, monkeypatch
):
    """A file changed less than ``SETTLE_NS`` before the read proves nothing.

    Its stamp could still miss a write inside the clock's granularity, so the
    read is not kept and the next preview reads again (the anchor's rule).
    """
    monkeypatch.setattr(run_start_tools, "SETTLE_NS", 3_600_000_000_000)
    case = preview_case
    first, surface = await _counted_preflight(case)
    assert first.calls == 1 and surface == LIGHT
    second, surface = await _counted_preflight(case)
    assert second.calls == 1, "a read of a just-written store was kept"
    assert surface == LIGHT


async def test_a_switch_read_that_raises_answers_closed_and_proves_nothing(
    preview_case, run_start_tools
):
    """A read that raises still counts as engaged, and nothing is kept from it.

    The preview offers no local tool for it, as before; the next preview
    reads the store again instead of answering from that failure.
    """
    case = preview_case
    first, surface = await _counted_preflight(case, fail=True)
    assert first.calls == 1 and surface == LIGHT - {"ask_user"}
    second, surface = await _counted_preflight(case)
    assert second.calls == 1, "a failed switch read was kept"
    assert surface == LIGHT


@contextlib.contextmanager
def _refusing_state(case, state):
    """Hold one in-process state in which a real switch read is refused."""
    permission = case.service._permission_store
    if state == "producer-paused":
        case.service._producer_lifetime.close()
        try:
            yield
        finally:
            case.service._producer_lifetime.resume()
    elif state == "storage-paused":
        pause = storage_admission._begin_local_pause()
        try:
            yield
        finally:
            pause.resume()
    elif state == "admission-closed":
        participant = raw_participants._raw_participant(permission)
        participant.close_admission()
        try:
            yield
        finally:
            participant.resume()
    else:
        # An earlier write that could not retire its native resources: every
        # later scope over the same selected file is refused until the process
        # retires them (here: the injected record is removed). The write was
        # this store's own, or another object's over the same file.
        from tldw_chatbook.Backup_Recovery import mcp_source_participants

        selected = mcp_source_participants.binding(permission)[1]
        token = object()
        record = raw_participants._State(
            permission if state == "unretired-write" else object(),
            None,
            selected,
            (),
            (),
            True,
            "mcp",
            os.getpid(),
            threading.current_thread(),
            None,
            uncertain=True,
        )
        with storage_admission._lock:
            raw_participants._states[token] = record
        try:
            with pytest.raises(RecoveryRequired, match="raw_resources_not_retired"):
                permission.get_kill_switch()
            yield
        finally:
            with storage_admission._lock:
                raw_participants._states.pop(token, None)


async def _outcome(case):
    """The preflight's tools, or the refusal it raised."""
    try:
        await _preflight(case)
    except _Captured:
        return {row["function"]["name"] for row in case.prepared.pop()[1]}
    except RecoveryRequired as refused:
        return f"refused: {refused}"
    raise AssertionError("the preflight neither planned nor refused")


@pytest.mark.parametrize(
    "state",
    [
        "producer-paused",
        "storage-paused",
        "admission-closed",
        "unretired-write",
        "unretired-write-other-object",
    ],
)
async def test_a_kept_switch_read_never_answers_where_a_real_read_is_refused(
    preview_case, run_start_tools, state
):
    """In-process states that refuse a real read also refuse the kept one.

    Paused MCP producers, a local storage pause, the store's participant
    admission closed, or an unretired uncertain write of the same file: none
    moves the file's stamp or the admission epoch, and a real read is refused
    there (answered as engaged). The kept read must answer exactly what the
    hop route answers in the same state, computed here with nothing kept.
    """
    case = preview_case
    first, surface = await _counted_preflight(case)
    assert first.calls == 1 and surface == LIGHT
    with _refusing_state(case, state):
        kept = await _outcome(case)
        run_start_tools.forget()
        read = await _outcome(case)
    assert read != LIGHT, "precondition: a real read is refused in this state"
    assert kept == read, "a kept switch read answered where a real read is refused"


async def test_a_kept_switch_read_never_waits_for_a_contended_storage_lock(
    preview_case, run_start_tools
):
    """The proof never waits on the storage lock; contended, the hop reads.

    Storage admission holds its lock across file reads and a registry
    ``flock`` (``_acquire_storage``), so a blocking wait for it on the UI loop
    can last as long as another thread's acquisition. With the lock held
    elsewhere for 1 s, the loop-side proof answers None at once and the
    preview takes the hop, which waits on the worker as dev's read does; the
    loop keeps ticking throughout.
    """
    case = preview_case
    first, surface = await _counted_preflight(case)
    assert first.calls == 1 and surface == LIGHT
    held, release = threading.Event(), threading.Event()

    def hold():
        with storage_admission._lock:
            held.set()
            release.wait(1.0)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert held.wait(2)
        started = time.perf_counter()
        answer = run_start_tools.proven_preview_switch(case.service._permission_store)
        waited = time.perf_counter() - started
        assert answer is None, "a kept read answered while the storage lock was held"
        assert waited < 0.2, f"the loop-side proof waited {waited * 1000:.0f} ms"
        gaps, ticking = [], True

        async def tick():
            last = time.perf_counter()
            while ticking:
                await asyncio.sleep(0.01)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        ticker = asyncio.create_task(tick())
        probe = _OriginalPermissionRead(case.permissions)
        probe.release.set()
        with probe.installed(), pytest.raises(_Captured):
            await _preflight(case)
        ticking = False
        await ticker
    finally:
        release.set()
        holder.join(timeout=5)
    assert probe.calls == 1, "the preview did not take the hop"
    assert probe.thread is not threading.current_thread()
    assert max(gaps) < 0.5, f"the UI loop stalled {max(gaps) * 1000:.0f} ms"
    assert {row["function"]["name"] for row in case.prepared.pop()[1]} == LIGHT
