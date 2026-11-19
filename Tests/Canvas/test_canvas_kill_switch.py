"""Cross-seam Canvas kill-switch behavior."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from Tests.Agents.test_canvas_tool_provider import SCOPE, _Coordinator
from tldw_chatbook.Agents.canvas_tool_provider import CanvasToolProvider
from tldw_chatbook.Agents.run_context import use_run_id, use_tool_call_id
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_message_actions import ConsoleMessageActionService
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime


def test_live_disable_removes_cached_canvas_catalog_and_fails_dispatch_closed() -> None:
    enabled = [True]
    coordinator = _Coordinator()
    provider = CanvasToolProvider(
        coordinator,
        scope=SCOPE,
        enabled_reader=lambda: enabled[0],
    )
    registry = ToolCatalogRegistry()
    assert registry.register_canvas_provider(
        provider, provider.issue_registration_authority()
    )
    assert {entry.name for entry in registry.list_catalog()} >= {
        "canvas_list",
        "canvas_read",
        "canvas_create",
        "canvas_update",
    }

    enabled[0] = False

    assert not {entry.name for entry in registry.list_catalog()} & {
        "canvas_list",
        "canvas_read",
        "canvas_create",
        "canvas_update",
    }
    with use_run_id(SCOPE.run_id), use_tool_call_id("disabled-call"):
        result = registry.invoke_by_name("canvas_list", {})
    assert result.ok is False
    assert result.error == "Canvas is disabled. Restart Chatbook after re-enabling it."
    assert coordinator.calls == []

    enabled[0] = True
    assert not {entry.name for entry in registry.list_catalog()} & {
        "canvas_list",
        "canvas_read",
        "canvas_create",
        "canvas_update",
    }


def test_disabled_canvas_hides_and_denies_html_block_actions() -> None:
    enabled = [False]
    service = ConsoleMessageActionService(canvas_enabled_reader=lambda: enabled[0])
    message = ConsoleChatMessage(
        id="assistant-html",
        role=ConsoleMessageRole.ASSISTANT,
        content="```html\n<!doctype html><title>Private</title>\n```",
    )

    assert not {action.action_id for action in service.available_actions(message)} & {
        "canvas-open-0",
        "canvas-open-new-0",
    }
    result = service.dispatch("canvas-open-0", message)
    assert result.status == "blocked"
    assert result.visible_copy == (
        "Canvas is disabled. Restart Chatbook after re-enabling it."
    )
    enabled[0] = True
    assert not {action.action_id for action in service.available_actions(message)} & {
        "canvas-open-0",
        "canvas-open-new-0",
    }


@pytest.mark.asyncio
async def test_native_live_disable_revokes_gateway_and_latches_until_restart() -> None:
    enabled = [True]
    runtime = ConsoleRuntime(object(), canvas_enabled_reader=lambda: enabled[0])

    class Gateway:
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    class Authority:
        disposed = False

        def dispose(self) -> None:
            self.disposed = True

    gateway = Gateway()
    authority = Authority()
    controller = object()
    runtime._canvas_gateway = gateway
    runtime._canvas_native_authority = authority
    runtime._canvas_controller = controller

    enabled[0] = False
    await runtime.apply_canvas_policy()

    assert gateway.closed is True
    assert authority.disposed is True
    assert runtime.canvas_gateway is None
    assert runtime.canvas_controller is controller
    enabled[0] = True
    assert runtime.ensure_canvas_gateway(authority=object()) is None


@pytest.mark.asyncio
async def test_native_policy_watcher_revokes_open_preview_after_external_disable() -> (
    None
):
    enabled = [True]
    runtime = ConsoleRuntime(object(), canvas_enabled_reader=lambda: enabled[0])

    class Gateway:
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    gateway = Gateway()
    runtime._canvas_gateway = gateway
    runtime._start_canvas_policy_watcher()

    enabled[0] = False
    await runtime._canvas_policy_watch_task

    assert gateway.closed is True
    assert runtime.canvas_gateway is None
    assert runtime.canvas_enabled() is False

    enabled[0] = True
    assert runtime.canvas_enabled() is False


def test_native_disabled_startup_does_not_create_canvas_authority_or_gateway() -> None:
    enabled = [False]
    runtime = ConsoleRuntime(object(), canvas_enabled_reader=lambda: enabled[0])

    assert runtime.ensure_canvas_gateway(authority=object()) is None
    assert (
        runtime.ensure_canvas_native_authority(
            scope_resolver=lambda _session_id: SimpleNamespace(),
        )
        is None
    )
    enabled[0] = True
    assert runtime.canvas_enabled() is False


@pytest.mark.asyncio
async def test_native_runtime_observes_disable_before_any_preview_exists() -> None:
    enabled = [True]
    runtime = ConsoleRuntime(object(), canvas_enabled_reader=lambda: enabled[0])
    await asyncio.sleep(0)

    enabled[0] = False
    assert runtime.canvas_enabled() is False
    enabled[0] = True

    assert runtime.canvas_enabled() is False
    assert runtime.ensure_canvas_gateway(authority=object()) is None
    await runtime.dispose()


@pytest.mark.parametrize("enabled_at_start", [True, False])
def test_canvas_guide_disable_reenable_latches_discovery_and_dispatch(enabled_at_start):
    enabled = [enabled_at_start]
    coordinator = _Coordinator()
    provider = CanvasToolProvider(
        coordinator, scope=SCOPE, enabled_reader=lambda: enabled[0]
    )
    registry = ToolCatalogRegistry()
    registered = registry.register_canvas_provider(
        provider, provider.issue_registration_authority()
    )
    assert registered is enabled_at_start
    if enabled_at_start:
        assert registry.resolve_name("canvas_guide") == "canvas:canvas_guide"
        with use_run_id(SCOPE.run_id), use_tool_call_id("guide-call"):
            assert registry.invoke_by_name("canvas_guide", {"topic": "controls"}).ok
    enabled[0] = False
    for live in (False, True):
        enabled[0] = live
        assert "canvas_guide" not in {entry.name for entry in registry.list_catalog()}
        with use_run_id(SCOPE.run_id), use_tool_call_id("guide-call"):
            direct = provider.invoke("canvas:canvas_guide", {"topic": "controls"})
            cached = registry.invoke_by_name("canvas_guide", {"topic": "controls"})
        assert not direct.ok and not cached.ok
        assert json.loads(direct.error)["code"] == "canvas_disabled"
    assert coordinator.calls == []


@pytest.mark.parametrize("source", ["explicit", "patched_current_getter"])
def test_policy_poll_keeps_custom_reader_and_dynamic_getter_seams(monkeypatch, source):
    import threading

    from tldw_chatbook import config
    from tldw_chatbook.Canvas import config_policy

    calls = []

    def custom_reader():
        calls.append(True)
        return True

    if source == "patched_current_getter":
        monkeypatch.setattr(
            config_policy, "get_canvas_execution_enabled", custom_reader
        )
        runtime = ConsoleRuntime(object())
    else:
        runtime = ConsoleRuntime(object(), canvas_enabled_reader=custom_reader)
    calls.clear()
    held, release = threading.Event(), threading.Event()

    def hold_rebuild():
        with config._settings_rebuild_lock():
            held.set()
            assert release.wait(10)

    holder = threading.Thread(target=hold_rebuild)
    holder.start()
    try:
        assert held.wait(5)
        assert (
            config_policy._poll_runtime_canvas_enabled(runtime, ConsoleRuntime) is True
        )
        assert calls == [True], "custom reader was not invoked exactly once"
        assert not release.is_set() and not runtime.canvas_disabled()
    finally:
        release.set()
        holder.join(5)
    assert not holder.is_alive()


@pytest.mark.parametrize("phase", ["lookup", "call"])
@pytest.mark.parametrize(
    "case", ["paused", "other_recovery", "subclass", "exception", "base_exception"]
)
def test_policy_read_preserves_live_lookup_and_error_normalization(phase, case):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    class RecoverySubclass(RecoveryRequired):
        pass

    errors = {
        "paused": RecoveryRequired("storage_locally_paused"),
        "other_recovery": RecoveryRequired("other_refusal"),
        "subclass": RecoverySubclass("storage_locally_paused"),
        "exception": RuntimeError("synthetic_reader_failure"),
        "base_exception": BaseException("synthetic_base_failure"),
    }
    error, calls = errors[case], []

    class Owner:
        @property
        def _canvas_enabled_reader(self):
            calls.append("lookup")
            if phase == "lookup":
                raise error
            return self.read

        def read(self):
            calls.append("call")
            raise error

    if case == "base_exception":
        with pytest.raises(BaseException) as raised:
            ConsoleRuntime._read_canvas_enabled(Owner())
        assert raised.value is error
    else:
        expected = None if case == "paused" else False
        assert ConsoleRuntime._read_canvas_enabled(Owner()) is expected
    assert calls == (["lookup"] if phase == "lookup" else ["lookup", "call"])


@pytest.mark.parametrize("same_request", [False, True])
@pytest.mark.parametrize("accepted", [False, True])
def test_old_read_cleanup_cannot_retire_a_successor_handoff(same_request, accepted):
    from tldw_chatbook.Canvas import config_policy as policy
    from tldw_chatbook.UI.Console_Modules import config_sync

    screen = SimpleNamespace(_console_config_handoffs={})
    projection = SimpleNamespace(_display_proof=None)
    old = config_sync._ConsoleConfigHandoff(screen, "identity", {})
    old.token = policy._begin_canvas_poll_handoff(screen)
    old_token = old.token
    issued = [(), {"identity": (old, old_token)}, None]
    successor = (
        old
        if same_request
        else config_sync._ConsoleConfigHandoff(screen, "identity", {})
    )
    new_token = policy._begin_canvas_poll_handoff(screen)
    successor.token = new_token
    screen._console_config_handoffs["identity"] = successor
    try:
        config_sync._finish_identity_config_read(projection, issued, accepted)
        assert new_token.active, "old read cleanup closed its successor"
        assert screen._console_config_handoffs["identity"] is successor
    finally:
        policy._end_canvas_poll_handoffs(screen)


def test_a_reading_handoff_retires_when_its_source_is_no_longer_current():
    from tldw_chatbook.Canvas import config_policy as policy
    from tldw_chatbook.UI.Console_Modules import config_sync

    screen = SimpleNamespace(_console_config_handoffs={})
    request = config_sync._ConsoleConfigHandoff(screen, "core", {})
    token = request.token = policy._begin_canvas_poll_handoff(screen)
    screen._console_config_handoffs["core"] = request
    try:
        assert request.receipt is None
        assert not request.eligible()
        assert not token.active and screen._console_config_handoffs == {}
    finally:
        policy._end_canvas_poll_handoffs(screen)


@pytest.mark.parametrize("cancelled,finished", [(True, False), (False, True)])
def test_a_cancelled_uninstalled_worker_retires_its_logical_handoff(
    cancelled, finished
):
    from tldw_chatbook.Canvas import config_policy as policy
    from tldw_chatbook.UI.Console_Modules import config_sync

    screen = SimpleNamespace(_console_config_handoffs={})
    request = config_sync._ConsoleConfigHandoff(screen, "identity", {})
    token = request.token = policy._begin_canvas_poll_handoff(screen)
    request.current = lambda: True  # Logical state control; no policy/physical claim.
    request.projection = SimpleNamespace(
        _handoff_read=[
            (),
            {},
            SimpleNamespace(is_cancelled=cancelled, is_finished=finished),
        ]
    )
    screen._console_config_handoffs["identity"] = request
    try:
        assert not request.eligible()
        assert not token.active and screen._console_config_handoffs == {}
    finally:
        policy._end_canvas_poll_handoffs(screen)


def test_stock_refresh_metadata_is_bound_before_worker_enqueue(monkeypatch):
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend

    queued = []
    worker = SimpleNamespace(is_cancelled=False, is_finished=False)
    screen = SimpleNamespace(
        run_worker=lambda coro, **_kwargs: (queued.append(coro), worker)[1]
    )
    projection = spend.ConsoleReadinessConfigProjection(
        screen, read_current=lambda: None
    )
    monkeypatch.setattr(projection, "_key", lambda: ())
    assert projection.run(lambda: None) is False
    try:
        first = projection._handoff_read
        assert queued[0].cr_frame.f_locals["_handoff_read"] is first
        assert first[2] is worker
        projection._handoff_read = [(), {"successor": object()}, worker]
        assert queued[0].cr_frame.f_locals["_handoff_read"] is first
    finally:
        queued[0].close()  # No physical read or task was started by this unit queue.


def test_a_custom_refresh_keeps_its_key_only_call_shape(monkeypatch):
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend

    queued, keys = [], []
    screen = SimpleNamespace(run_worker=lambda coro, **_kwargs: queued.append(coro))
    projection = spend.ConsoleReadinessConfigProjection(
        screen, read_current=lambda: None
    )
    monkeypatch.setattr(projection, "_key", lambda: ())

    async def custom_refresh(key):
        keys.append(key)

    monkeypatch.setattr(projection, "_refresh", custom_refresh)
    assert projection.run(lambda: None) is False
    try:
        assert queued[0].cr_frame.f_locals["key"] == ()
        assert "_handoff_read" not in queued[0].cr_frame.f_locals
        assert keys == []  # It was passed once, not run by this unit queue.
    finally:
        queued[0].close()


@pytest.mark.asyncio
@pytest.mark.parametrize("native_raises", [False, True])
async def test_advisory_handoff_failure_cannot_replace_native_outcome_or_settlement(
    monkeypatch, native_raises
):
    from tldw_chatbook.UI.Console_Modules import config_sync
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend

    error = (
        BaseException("synthetic_native_failure")
        if native_raises
        else RuntimeError("synthetic_checked_read_refusal")
    )
    screen = SimpleNamespace(set_timer=lambda *_args: None)
    key = (None,) * 11

    def read():
        raise error

    projection = spend.ConsoleReadinessConfigProjection(screen, read_current=read)
    monkeypatch.setattr(projection, "_key", lambda: key)
    projection.pending = True
    projection._settled.clear()

    def fail_cleanup(*_args):
        raise BaseException("synthetic_advisory_failure")

    monkeypatch.setattr(config_sync, "_finish_identity_config_read", fail_cleanup)
    if native_raises:
        with pytest.raises(BaseException) as raised:
            await projection._refresh(key)
        assert raised.value is error
    else:
        assert (
            await projection._refresh(key) is None
        )  # Original Exception normalization.
    assert projection._settled.is_set() and not projection.pending
    assert not projection._reads


@pytest.mark.parametrize("retirement", ["replace", "detach", "stale_detach"])
def test_runtime_view_retirement_releases_only_its_opaque_handoff(retirement):
    from tldw_chatbook.Canvas import config_policy as policy

    runtime = ConsoleRuntime(object(), canvas_enabled_reader=lambda: True)
    first, second = SimpleNamespace(), SimpleNamespace()
    generation = runtime.attach_view(first)
    old = policy._begin_canvas_poll_handoff(first)
    new = None
    try:
        if retirement == "detach":
            assert runtime.detach_view(first, generation)
            assert not old.active
        else:
            runtime.attach_view(second)
            assert not old.active
            new = policy._begin_canvas_poll_handoff(second)
            if retirement == "stale_detach":
                assert not runtime.detach_view(first, generation)
                assert new.active
            else:
                assert runtime.detach_view(None)
                assert not new.active
    finally:
        policy._end_canvas_poll_handoffs(first)
        policy._end_canvas_poll_handoffs(second)



def test_idle_coalesced_replay_retires_core_without_stealing_identity_custody():
    from tldw_chatbook.Canvas import config_policy as policy
    from tldw_chatbook.UI.Console_Modules import config_sync

    rendered = []
    screen = SimpleNamespace(
        _console_config_handoffs={},
        _console_sync_in_progress=False,
        _console_control_bar_replay_whole_sync=False,
        _sync_console_control_bar=lambda: rendered.append(True),
    )
    core = config_sync._ConsoleConfigHandoff(screen, "core", {})
    identity = config_sync._ConsoleConfigHandoff(screen, "identity", {})
    core.token = core_token = policy._begin_canvas_poll_handoff(screen)
    identity.token = identity_token = policy._begin_canvas_poll_handoff(screen)
    screen._console_config_handoffs.update(core=core, identity=identity)
    try:
        config_sync.run_coalesced_control_bar_sync(screen, lambda _: False)
        assert not core_token.active
        assert identity_token.active, "idle CORE timer stole the independent identity retry"
        assert screen._console_config_handoffs == {"identity": identity}
        assert rendered == [True]
    finally:
        policy._end_canvas_poll_handoffs(screen)



def test_full_warm_starts_only_the_owed_core_handoff():
    from tldw_chatbook.UI.Console_Modules import config_sync

    began = []
    core_token, identity_token, projection = object(), object(), object()
    core = SimpleNamespace(
        begin=lambda value: (began.append(("core", value)), core_token)[1]
    )
    identity = SimpleNamespace(
        begin=lambda value: (began.append(("identity", value)), identity_token)[1]
    )
    screen = SimpleNamespace(
        _console_config_handoffs={"core": core, "identity": identity}
    )
    selected = config_sync._begin_console_config_handoffs(screen, projection)
    assert began == [("core", projection)], "FULL warm activated unissued identity work"
    assert selected == ((core, core_token),)
def test_unissued_identity_kick_retires_its_logical_lease(monkeypatch):
    """Pure scheduling-branch coverage, not evidence of a real UI cold episode."""
    from tldw_chatbook import config
    from tldw_chatbook.Canvas import config_policy as policy
    from tldw_chatbook.UI.Console_Modules import config_sync
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend

    calls, timers, body = [], [], []
    screen = SimpleNamespace(
        set_timer=lambda delay, callback: timers.append((delay, callback))
    )
    projection = SimpleNamespace(
        pending=False, run=lambda callback: calls.append("unissued_run")
    )

    class LogicalRequest:
        token = None
        receipt = None  # No proof or READY flag is fabricated in this pure unit.

        def eligible(self):
            return False

        def begin(self, _projection):
            calls.append("begin")
            self.token = policy._begin_canvas_poll_handoff(screen)
            return self.token

        def retire(self, **_kwargs):
            calls.append("retire")
            policy._end_canvas_poll_handoffs(screen, self.token)
            self.token = None

    logical = LogicalRequest()
    monkeypatch.setattr(config_sync, "_owe_console_config_sync", lambda *_args: logical)
    monkeypatch.setattr(spend, "defer_cold_live_read", lambda *_args: True)
    monkeypatch.setattr(
        spend.ConsoleReadinessConfigProjection, "for_screen", lambda _screen: projection
    )

    @spend.retry_console_identity_after_config_rebuild
    def identity(_screen):
        body.append(True)

    try:
        with config._settings_rebuild_lock(), config._config_file_lock():
            identity(screen)
        assert calls == ["begin", "unissued_run", "retire"]
        assert logical.token is None and not body
        assert screen._console_config_retries[identity.__name__] == ((), {})
        assert len(timers) == 1 and timers[0][0] == 0.2
    finally:
        policy._end_canvas_poll_handoffs(screen)
