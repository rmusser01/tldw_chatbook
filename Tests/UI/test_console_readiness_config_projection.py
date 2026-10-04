"""Presentation config IO stays owned off-loop; actions still read live."""

import asyncio
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from tldw_chatbook import config
from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend

pytestmark = pytest.mark.bootstrap_profile


def _screen(monkeypatch):
    identity = list(config.current_config_identity())
    monkeypatch.setattr(config, "current_config_identity", lambda: tuple(identity))
    owner = SimpleNamespace(id="first", workspace_id="workspace", revision=1)
    store = SimpleNamespace(
        active_session_id=owner.id,
        sessions=lambda: [owner],
        session_settings_revision=lambda _: owner.revision,
    )
    tasks, published = [], []

    def schedule(operation, **_):
        task = asyncio.create_task(operation)
        tasks.append(task)
        return task

    async def refresh():
        published.append(True)

    screen = SimpleNamespace(
        app_instance=SimpleNamespace(
            app_config={"original": True}, chachanotes_db=object()
        ),
        _console_chat_store=store,
        _console_derivation_memo=None,
        _console_config_snapshot_is_disk_loaded=lambda _: True,
        run_worker=schedule,
        _sync_native_console_chat_ui=refresh,
        _request_console_control_bar_sync=lambda **_: None,
    )

    @contextmanager
    def scope():
        previous = screen._console_derivation_memo
        screen._console_derivation_memo = {} if previous is None else previous
        try:
            yield
        finally:
            screen._console_derivation_memo = previous

    screen._console_derivation_scope = scope
    return screen, owner, identity, tasks, published


def _read_result(value, before=None):
    identity = config.current_config_identity()
    return spend.ConsoleReadinessConfigRead(before or identity, value, identity)


def test_checked_config_source_tags_require_active_matching_operation():
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.config_participants import (
        checked_config_identity,
        operation,
    )

    with operation(config) as active:
        assert (
            checked_config_identity(config, active) == config.current_config_identity()
        )
        with pytest.raises(RecoveryRequired, match="config_operation_source_invalid"):
            checked_config_identity(object(), active)
    with pytest.raises(RecoveryRequired, match="raw_operation_provenance_invalid"):
        checked_config_identity(config, active)


@pytest.mark.asyncio
async def test_cold_and_expired_config_presentation_share_owned_read_and_live_action(
    monkeypatch,
):
    screen, _owner, _identity, tasks, published = _screen(monkeypatch)
    calls, renders = [], []
    value = {"current": 1}

    def read():
        calls.append(threading.get_ident())
        return _read_result(dict(value))

    assert hasattr(spend, "ConsoleReadinessConfigProjection")
    projection = spend.ConsoleReadinessConfigProjection(screen, read_current=read)

    def render():
        renders.append(spend.provider_readiness_app_config(screen, read))

    for _ in range(6):
        assert projection.run(render) is False
    assert not calls and not renders and len(tasks) == 1
    await asyncio.gather(*tasks)
    assert calls == [calls[0]] and calls[0] != threading.get_ident()
    for _ in range(6):
        assert projection.run(render) is True
    assert renders == [{"current": 1}] * 6 and len(calls) == 1
    projection.at -= projection.max_age + 1
    value["current"] = 2
    for _ in range(6):
        assert projection.run(render) is True
    assert renders[-6:] == [{"current": 1}] * 6
    assert len(calls) == 1
    await asyncio.gather(*tasks)
    assert len(calls) == 2
    assert projection.run(render) is True and renders[-1] == {"current": 2}
    assert all(thread != threading.get_ident() for thread in calls)
    assert spend.provider_readiness_app_config(screen, lambda: read().value) == {
        "current": 2
    }
    assert calls[-1] == threading.get_ident(), "explicit getter stopped reading live"
    assert published


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation", ["profile", "session", "workspace", "settings", "database"]
)
async def test_readiness_config_publication_rejects_changed_owner(
    monkeypatch, mutation
):
    screen, owner, identity, tasks, published = _screen(monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def read():
        before = config.current_config_identity()
        entered.set()
        assert release.wait(3)
        return _read_result({"foreign": mutation}, before)

    assert hasattr(spend, "ConsoleReadinessConfigProjection")
    projection = spend.ConsoleReadinessConfigProjection(screen, read_current=read)
    assert projection.run(lambda: pytest.fail("cold projection rendered")) is False
    assert await asyncio.to_thread(entered.wait, 3)
    if mutation == "profile":
        identity[1] = "second-profile"
    elif mutation == "session":
        screen._console_chat_store.active_session_id = "second"
    elif mutation == "workspace":
        owner.workspace_id = "second-workspace"
    elif mutation == "settings":
        owner.revision += 1
    else:
        screen.app_instance.chachanotes_db = object()
    release.set()
    await asyncio.gather(*tasks)
    assert projection.value is None and not published


@pytest.mark.asyncio
async def test_cancelled_config_projection_retires_worker_then_can_retry(monkeypatch):
    screen, _owner, _identity, tasks, _published = _screen(monkeypatch)
    entered, release, retired = threading.Event(), threading.Event(), threading.Event()

    def read():
        try:
            entered.set()
            assert release.wait(3)
            return _read_result({"retired": True})
        finally:
            retired.set()

    assert hasattr(spend, "ConsoleReadinessConfigProjection")
    projection = spend.ConsoleReadinessConfigProjection(screen, read_current=read)
    assert projection.run(lambda: None) is False
    assert await asyncio.to_thread(entered.wait, 3)
    tasks[0].cancel()
    await asyncio.sleep(0)
    assert not tasks[0].done() and not retired.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await tasks[0]
    assert retired.is_set() and not projection.pending
    assert projection.run(lambda: None) is False
    await asyncio.gather(*tasks[1:])
    assert projection.value == {"retired": True}


@pytest.mark.asyncio
async def test_real_screen_presentation_entry_reuses_config_while_actions_read_live(
    monkeypatch,
):
    from tldw_chatbook.UI.Screens import chat_screen as module

    screen, _owner, _identity, tasks, _published = _screen(monkeypatch)
    calls, rendered = [], []

    def load():
        calls.append(threading.get_ident())
        return {"current": True}

    monkeypatch.setattr(module, "load_settings", load)
    screen._provider_readiness_app_config = (
        lambda: module.ChatScreen._provider_readiness_app_config(screen)
    )

    def body():
        rendered.append(screen._provider_readiness_app_config())

    assert module.ChatScreen._run_console_config_sync(screen, body) is False
    assert not calls and not rendered
    await asyncio.gather(*tasks)
    for _ in range(6):
        assert module.ChatScreen._run_console_config_sync(screen, body) is True
    assert rendered == [{"current": True}] * 6
    assert len(calls) == 1 and calls[0] != threading.get_ident()
    assert screen._provider_readiness_app_config() == {"current": True}
    assert calls[-1] == threading.get_ident()


@pytest.mark.asyncio
async def test_actual_mode_bar_defers_cold_config_and_uses_owned_presentation(
    monkeypatch,
):
    from tldw_chatbook.UI.Screens import chat_screen as module

    screen, _owner, _identity, tasks, _published = _screen(monkeypatch)
    reads, updates = [], []

    def load():
        reads.append(threading.get_ident())
        return {"current": True}

    monkeypatch.setattr(module, "load_settings", load)
    screen._provider_readiness_app_config = (
        lambda: module.ChatScreen._provider_readiness_app_config(screen)
    )
    mode_bar = SimpleNamespace(update=updates.append)
    chips = SimpleNamespace(sync_run_chip=lambda *_args: None)
    screen.query_one = (
        lambda selector, *_args: mode_bar if selector == "#console-mode-bar" else chips
    )
    screen._pending_console_launch_context = None
    screen._build_console_control_state = (
        lambda *_args: screen._provider_readiness_app_config()
    )
    screen._console_mode_summary = lambda state: str(state["current"])
    screen._native_run_status_copy = lambda: "Streaming"
    screen._console_active_run_copy = lambda: "Streaming"
    # The same real body without its projection is the positive UI-IO control.
    module.ChatScreen._sync_console_mode_bar.__wrapped__(screen)
    assert reads == [threading.get_ident()] and updates == ["True | Run: Streaming"]
    reads.clear()
    updates.clear()
    assert module.ChatScreen._sync_console_mode_bar(screen) is False
    assert not reads and not updates
    await asyncio.gather(*tasks)
    for _ in range(6):
        assert module.ChatScreen._sync_console_mode_bar(screen) is True
    assert len(reads) == 1 and reads[0] != threading.get_ident()
    assert updates == ["True | Run: Streaming"] * 6


@pytest.mark.asyncio
async def test_checked_config_read_rejects_retarget_to_other_source_and_back(
    monkeypatch,
):
    actual_source = config.current_config_identity()
    screen, _owner, identity, tasks, published = _screen(monkeypatch)
    # Vary the UI's cheap selector observation while the real native operation
    # reads its installed B source. A literal env retarget is separately refused
    # by the installed participant; this must not weaken that source guard.
    from pathlib import Path

    requested = (
        actual_source[0],
        str(Path(actual_source[1]).with_name("requested-A.toml")),
    )
    identity[:] = requested
    entered, begin, loaded, finish = [threading.Event() for _ in range(4)]
    failures = []
    real_to_thread = asyncio.to_thread

    def controlled_read(function):
        entered.set()
        assert begin.wait(3)
        try:
            result = function()
        except Exception as error:
            failures.append(error)
            loaded.set()
            raise
        loaded.set()
        assert finish.wait(3)
        return result

    async def gated_to_thread(function):
        return await real_to_thread(controlled_read, function)

    monkeypatch.setattr(spend.asyncio, "to_thread", gated_to_thread)
    projection = spend.ConsoleReadinessConfigProjection.for_screen(screen)
    try:
        assert projection.run(lambda: pytest.fail("cold owner rendered")) is False
        assert await real_to_thread(entered.wait, 3)
        identity[:] = actual_source
        begin.set()
        assert await real_to_thread(loaded.wait, 3)
        assert not failures, repr(failures)
        identity[:] = requested
        finish.set()
        await asyncio.gather(*tasks)
        assert (
            projection.value is None
        ), "other source was published after selector returned"
        assert not published
    finally:
        begin.set()
        finish.set()
        await asyncio.gather(*tasks, return_exceptions=True)


def _real_session_screen(monkeypatch):
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_session_settings import (
        blank_console_session_settings,
    )
    from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
    from tldw_chatbook.UI.Screens import chat_screen as module

    screen, _owner, _identity, tasks, _published = _screen(monkeypatch)
    screen.app_instance.app_config = config.load_settings().copy()
    screen.app_instance.app_config["chat_defaults"] = {
        "provider": "llama_cpp",
        "model": "established",
    }
    store = ConsoleChatStore()
    screen._console_chat_store = store
    # The production session implementation and store own all convergence and
    # provenance gates; only unrelated constructor dependencies are omitted.
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    controller._screen = screen
    controller.app_instance = screen.app_instance
    controller._chat_store_accessor = lambda: store
    controller._provider_readiness_app_config_fn = (
        lambda: spend.provider_readiness_app_config(screen, module.load_settings)
    )
    controller._pristine_defaults_checked = None
    screen._session = controller
    established = blank_console_session_settings(screen.app_instance.app_config)
    session = store.create_session(
        settings=established, canonical_settings_baseline=established
    )
    return screen, store, session, tasks


@pytest.mark.asyncio
async def test_expired_presentation_never_converges_real_session_but_fresh_handoff_does(
    monkeypatch,
):
    import copy

    from tldw_chatbook.UI.Screens import chat_screen as module

    screen, store, session, tasks = _real_session_screen(monkeypatch)
    stale = copy.deepcopy(screen.app_instance.app_config)
    stale["chat_defaults"]["model"] = "expired"
    fresh = copy.deepcopy(screen.app_instance.app_config)
    fresh["chat_defaults"]["model"] = "fresh"
    entered, release = threading.Event(), threading.Event()

    def read():
        entered.set()
        assert release.wait(3)
        return _read_result(fresh)

    projection = spend.ConsoleReadinessConfigProjection(screen, read_current=read)
    projection.key, projection.value = projection._key(), stale
    revision = store.session_settings_revision(session.id)
    try:
        assert projection.run(screen._session._ensure_active_console_session_settings)
        assert session.settings.model == "established"
        assert store.session_settings_revision(session.id) == revision
    finally:
        release.set()
        await asyncio.gather(*tasks)
    assert session.settings.model == "fresh", "fresh checked result did not converge"
    assert store.session_settings_revision(session.id) == revision + 1
    assert (
        projection.key == projection._key()
    ), "own convergence invalidated the new mapping"
    task_count = len(tasks)
    assert projection.run(screen._session._ensure_active_console_session_settings)
    assert len(tasks) == task_count
    action = copy.deepcopy(fresh)
    action["chat_defaults"]["model"] = "live-action"
    monkeypatch.setattr(module, "load_settings", lambda: action)
    assert (
        screen._session._ensure_active_console_session_settings().model == "live-action"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["session", "workspace"])
async def test_fresh_default_handoff_rejects_actual_session_owner_swap(
    monkeypatch, mutation
):
    import copy

    screen, store, session, tasks = _real_session_screen(monkeypatch)
    fresh = copy.deepcopy(screen.app_instance.app_config)
    fresh["chat_defaults"]["model"] = "foreign"
    entered, release = threading.Event(), threading.Event()

    def read():
        entered.set()
        assert release.wait(3)
        return _read_result(fresh)

    projection = spend.ConsoleReadinessConfigProjection(screen, read_current=read)
    assert projection.run(lambda: None) is False
    assert await asyncio.to_thread(entered.wait, 3)
    if mutation == "session":
        other = store.create_session(
            settings=session.settings, canonical_settings_baseline=session.settings
        )
        assert store.active_session_id == other.id
    else:
        session.workspace_id = "changed-workspace"
    release.set()
    await asyncio.gather(*tasks)
    assert projection.value is None
    assert session.settings.model == "established"
    if mutation == "session":
        assert other.settings.model == "established"


@pytest.mark.asyncio
@pytest.mark.parametrize("edge", ["before", "after"])
async def test_wrong_checked_source_edge_never_converges_actual_session(
    monkeypatch, edge
):
    import copy

    screen, _store, session, tasks = _real_session_screen(monkeypatch)
    fresh = copy.deepcopy(screen.app_instance.app_config)
    fresh["chat_defaults"]["model"] = "wrong-source"
    source = config.current_config_identity()
    wrong = (source[0] + 1, source[1])

    def read():
        return spend.ConsoleReadinessConfigRead(
            wrong if edge == "before" else source,
            fresh,
            wrong if edge == "after" else source,
        )

    projection = spend.ConsoleReadinessConfigProjection(screen, read_current=read)
    assert projection.run(lambda: None) is False
    await asyncio.gather(*tasks)
    assert projection.value is None and session.settings.model == "established"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ineligible", ["user_work", "source_owned", "default_generation"]
)
async def test_fresh_handoff_preserves_actual_session_provenance_gates(
    monkeypatch, ineligible
):
    import copy

    screen, _store, session, tasks = _real_session_screen(monkeypatch)
    if ineligible == "user_work":
        session.has_user_work = True
    elif ineligible == "source_owned":
        session.canonical_settings_baseline = None
    else:
        screen.app_instance.console_new_chat_default_generation = 1
    fresh = copy.deepcopy(screen.app_instance.app_config)
    fresh["chat_defaults"]["model"] = "ineligible"
    projection = spend.ConsoleReadinessConfigProjection(
        screen, read_current=lambda: _read_result(fresh)
    )
    assert projection.run(lambda: None) is False
    await asyncio.gather(*tasks)
    assert projection.value == fresh
    assert session.settings.model == "established"
