"""Checked rendering never substitutes for a real native source lifetime."""

import asyncio
import os
import sys
import threading
from contextlib import contextmanager
from types import MethodType

import pytest

from Tests.UI.test_console_shared_context_policy import _screen as _real_screen
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import config_participants, raw_participants as raw
from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

pytestmark = pytest.mark.bootstrap_profile


def _screen(tmp_path):
    database, store, controller, screen, tasks = _real_screen(tmp_path)
    screen._console_config_snapshot_is_disk_loaded = (
        ChatScreen._console_config_snapshot_is_disk_loaded
    )
    screen._provider_readiness_app_config = MethodType(
        ChatScreen._provider_readiness_app_config, screen
    )
    screen._request_console_control_bar_sync = lambda **_: None
    return database, store, controller, screen, tasks


async def _warm(screen, tasks):
    projection = spend.ConsoleReadinessConfigProjection.for_screen(screen)
    assert projection.run(lambda: None) is False
    await asyncio.gather(*tasks)
    assert projection.value is not None
    assert projection.key == projection._key()
    assert not any(state.source is config for state in raw._states.values())
    return projection


@contextmanager
def _actual_calls():
    """Observe only original source/native code; keep all guards installed."""
    operation_code = config_participants.operation.__wrapped__.__code__
    read_code = config.read_cli_config_serialized.__wrapped__.__code__
    native_open_code = None
    if os.name == "nt":
        from tldw_chatbook.Utils import windows_files

        native_open_code = windows_files._native().open_handle.__func__.__code__
    frames = []
    main = threading.get_ident()
    calls = {"main_scopes": 0, "main_opens": 0, "disk_reads": [], "owners": []}
    old_main, old_thread = sys.getprofile(), threading.getprofile()

    def observe(frame, event, argument):
        if threading.get_ident() != main:
            return
        if event == "call":
            if frame.f_code is operation_code and all(
                frame is not old for old in frames
            ):
                frames.append(frame)
                calls["main_scopes"] += 1
            elif native_open_code is not None and frame.f_code is native_open_code:
                calls["main_opens"] += 1
            elif frame.f_code is read_code:
                calls["disk_reads"].append(getattr(raw._local, "operation", None))
        elif (
            event == "return"
            and frame.f_code is operation_code
            and type(argument) is raw._RawOperation
        ):
            calls["owners"].append(argument)
        elif event == "c_call" and os.name != "nt" and argument is os.open:
            calls["main_opens"] += 1

    threading.setprofile_all_threads(observe)
    try:
        yield calls
    finally:
        threading.setprofile_all_threads(old_thread)
        sys.setprofile(old_main)
        frames.clear()


@pytest.mark.asyncio
async def test_actual_warm_display_opens_no_redundant_native_scope(
    tmp_path, record_property
):
    database, _store, _controller, screen, tasks = _screen(tmp_path)
    rendered, operations = [], []
    try:
        projection = await _warm(screen, tasks)

        def render():
            rendered.append(screen._provider_readiness_app_config())
            operations.append(getattr(raw._local, "operation", None))

        with _actual_calls() as calls:
            for _ in range(6):
                assert ChatScreen._run_console_config_sync(screen, render) is True
        record_property("main_config_entries", calls["main_scopes"])
        record_property("main_native_opens", calls["main_opens"])
        assert len(rendered) == 6
        assert all(value is projection.value for value in rendered)
        assert calls["main_scopes"] == 0, calls
        assert calls["main_opens"] == 0, calls
        assert operations == [None] * 6, "display borrowed a retired raw operation"
        assert not any(state.source is config for state in raw._states.values())
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_default_config_sync_still_enters_real_native_scope(tmp_path):
    database, _store, _controller, screen, tasks = _screen(tmp_path)
    observed = []
    try:
        await _warm(screen, tasks)
        with _actual_calls() as calls:
            assert spend.run_console_config_sync(
                lambda: observed.append(getattr(raw._local, "operation", None)),
                maintenance_paused=False,
                request_retry=lambda: pytest.fail("fresh default entry deferred"),
            )
        assert calls["main_scopes"] == 1
        assert calls["main_opens"] > 0
        assert len(observed) == 1 and observed[0] is not None
        assert observed[0] not in raw._states
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_warm_display_nested_disk_reader_keeps_its_own_fresh_scope(tmp_path):
    database, _store, _controller, screen, tasks = _screen(tmp_path)
    serialized, before_read = [], []
    try:
        await _warm(screen, tasks)

        def read():
            before_read.append(getattr(raw._local, "operation", None))
            serialized.append(config.read_cli_config_serialized())

        with _actual_calls() as calls:
            assert ChatScreen._run_console_config_sync(screen, read) is True
        assert serialized and isinstance(serialized[0], str)
        assert before_read == [None], "nested reader borrowed an enclosing scope"
        # The serialized reader and its two original private helpers retain
        # all three guards while sharing only their own newly issued raw owner.
        assert calls["main_scopes"] == 3, calls
        assert calls["main_opens"] > 0
        assert len(set(calls["owners"])) == 1, calls
        assert all(owner not in raw._states for owner in calls["owners"])
        assert len(calls["disk_reads"]) == 1
        assert calls["disk_reads"][0] is not None
        assert calls["disk_reads"][0] not in raw._states
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_genuine_display_expiry_uses_original_native_entry(tmp_path):
    database, _store, _controller, screen, tasks = _screen(tmp_path)
    observed = []
    try:
        projection = await _warm(screen, tasks)
        assert projection.max_age == 1
        await asyncio.sleep(1.02)
        with _actual_calls() as calls:
            assert ChatScreen._run_console_config_sync(
                screen, lambda: observed.append(getattr(raw._local, "operation", None))
            )
        assert calls["main_scopes"] == 1 and calls["main_opens"] > 0
        assert len(observed) == 1 and observed[0] is not None
        assert observed[0] not in raw._states
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["custom", "copied_proof", "changed_reader", "alias"])
async def test_unqualified_display_uses_fresh_native_lifetime(
    tmp_path, monkeypatch, kind
):
    import copy

    from tldw_chatbook.UI.Screens import chat_screen

    database, _store, _controller, screen, tasks = _screen(tmp_path)
    observed = []
    try:
        if kind == "alias":
            actual = chat_screen.load_settings
            monkeypatch.setattr(chat_screen, "load_settings", lambda: actual())
        projection = await _warm(screen, tasks)
        if kind == "custom":
            custom = spend.ConsoleReadinessConfigProjection(
                screen, read_current=projection.read_current
            )
            custom.key, custom.value, custom.at = (
                projection.key,
                projection.value,
                projection.at,
            )
            screen._console_readiness_config_projection = custom
        elif kind == "copied_proof":
            projection._display_proof = copy.copy(projection._display_proof)
        elif kind == "changed_reader":
            actual = projection.read_current
            projection.read_current = lambda: actual()
        with _actual_calls() as calls:
            assert ChatScreen._run_console_config_sync(
                screen, lambda: observed.append(getattr(raw._local, "operation", None))
            )
        assert calls["main_scopes"] == 1 and calls["main_opens"] > 0
        assert len(observed) == 1 and observed[0] is not None
        assert observed[0] not in raw._states
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_foreign_ui_loop_retains_fresh_native_scope(tmp_path):
    database, _store, _controller, screen, tasks = _screen(tmp_path)
    observed = []
    try:
        await _warm(screen, tasks)

        async def other_loop():
            assert ChatScreen._run_console_config_sync(
                screen, lambda: observed.append(getattr(raw._local, "operation", None))
            )

        await asyncio.to_thread(lambda: asyncio.run(other_loop()))
        assert len(observed) == 1 and observed[0] is not None
        assert observed[0] not in raw._states
        assert not any(state.source is config for state in raw._states.values())
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["closed", "pause", "participant", "module"])
async def test_warm_display_refuses_changed_native_owner_before_body(
    tmp_path, mutation
):
    from dataclasses import replace
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    database, _store, _controller, screen, tasks = _screen(tmp_path)
    retry, rendered = [], []
    screen._request_console_control_bar_sync = lambda **_: retry.append(True)
    participant = state = pause = None
    try:
        await _warm(screen, tasks)
        participant = raw._source_participants[config]
        state = raw._participants[participant]
        if mutation == "closed":
            participant.close_admission()
        elif mutation == "pause":
            pause = storage._begin_local_pause()
        elif mutation == "participant":
            with storage._lock:
                raw._participants[participant] = replace(state)
        else:
            replacement = SimpleNamespace(
                current_config_identity=config.current_config_identity
            )
            sys.modules["tldw_chatbook.config"] = replacement
        with _actual_calls() as calls:
            assert (
                ChatScreen._run_console_config_sync(
                    screen, lambda: rendered.append(True)
                )
                is False
            )
        assert not rendered and retry == [True]
        assert calls["main_scopes"] == calls["main_opens"] == 0
    finally:
        sys.modules["tldw_chatbook.config"] = config
        if mutation == "participant" and participant is not None:
            with storage._lock:
                raw._participants[participant] = state
        if pause is not None:
            pause.resume()
        if mutation == "closed" and participant is not None:
            participant.resume()
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation", ["session", "workspace", "mapping", "source", "closed", "pause"]
)
async def test_warm_display_rejects_post_body_owner_change(
    tmp_path, monkeypatch, mutation
):
    from dataclasses import replace
    from pathlib import Path

    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    database, store, _controller, screen, tasks = _screen(tmp_path)
    retry, rendered, pauses = [], [], []
    screen._request_console_control_bar_sync = lambda **_: retry.append(True)
    participant = None
    original_env = os.environ.get("TLDW_CONFIG_PATH")
    try:
        await _warm(screen, tasks)
        participant = raw._source_participants[config]

        def render():
            rendered.append(True)
            if mutation == "session":
                owner = store._sessions["session-1"]
                store._sessions["session-1"] = replace(owner)
            elif mutation == "workspace":
                store._sessions["session-1"].workspace_id = "changed-display-owner"
            elif mutation == "mapping":
                screen.app_instance.app_config = dict(screen.app_instance.app_config)
            elif mutation == "source":
                selected = Path(config.current_config_identity()[1])
                monkeypatch.setenv(
                    "TLDW_CONFIG_PATH", str(selected.with_name("other-source.toml"))
                )
            elif mutation == "closed":
                participant.close_admission()
            else:
                pauses.append(storage._begin_local_pause())

        with _actual_calls() as calls:
            assert ChatScreen._run_console_config_sync(screen, render) is False
        assert rendered == retry == [True]
        assert calls["main_scopes"] == calls["main_opens"] == 0
        assert not any(state.source is config for state in raw._states.values())
    finally:
        if original_env is None:
            monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
        else:
            monkeypatch.setenv("TLDW_CONFIG_PATH", original_env)
        for pause in pauses:
            pause.resume()
        if mutation == "closed" and participant is not None:
            participant.resume()
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_actual_concurrent_saved_generation_invalidates_display(tmp_path):
    database, _store, _controller, screen, tasks = _screen(tmp_path)
    retry, saved, errors, rendered = [], [], [], []
    finished = threading.Event()
    screen._request_console_control_bar_sync = lambda **_: retry.append(True)
    thread = None
    try:
        await _warm(screen, tasks)
        generation = config.current_config_identity()[0]

        def write():
            try:
                saved.append(
                    config.save_setting_to_cli_config(
                        "general", "users_name", "display-generation-control"
                    )
                )
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        thread = threading.Thread(target=write, name="checked-display-real-save")

        def render():
            rendered.append(getattr(raw._local, "operation", None))
            thread.start()
            assert finished.wait(
                15
            ), "display retained a native scope while the real writer waited"
            assert saved == [True] and not errors
            assert config.current_config_identity()[0] > generation

        assert ChatScreen._run_console_config_sync(screen, render) is False
        assert rendered == [None] and retry == [True]
        thread.join(5)
        assert not thread.is_alive()
        assert not any(state.source is config for state in raw._states.values())
    finally:
        if thread is not None and thread.ident is not None:
            thread.join(20)
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_warm_display_body_error_retains_identity_and_native_failure_state(
    tmp_path,
):
    database, _store, _controller, screen, tasks = _screen(tmp_path)
    failure = RuntimeError("original-display-body-error")
    try:
        await _warm(screen, tasks)
        before = config._CONFIG_PERSISTENCE_ERROR

        def render():
            assert getattr(raw._local, "operation", None) is None
            raise failure

        with pytest.raises(RuntimeError) as caught, _actual_calls() as calls:
            ChatScreen._run_console_config_sync(screen, render)
        assert caught.value is failure
        assert config._CONFIG_PERSISTENCE_ERROR is before
        assert calls["main_scopes"] == calls["main_opens"] == 0
        assert not any(state.source is config for state in raw._states.values())
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_actual_cold_screen_defers_body_until_checked_worker_settles(tmp_path):
    database, _store, _controller, screen, tasks = _screen(tmp_path)
    rendered = []
    try:
        with _actual_calls() as calls:
            assert (
                ChatScreen._run_console_config_sync(
                    screen, lambda: rendered.append(True)
                )
                is False
            )
        assert not rendered and calls["main_scopes"] == calls["main_opens"] == 0
        await asyncio.gather(*tasks)
        with _actual_calls() as calls:
            assert (
                ChatScreen._run_console_config_sync(
                    screen, lambda: rendered.append(True)
                )
                is True
            )
        assert rendered == [True] and calls["main_scopes"] == calls["main_opens"] == 0
        assert not any(state.source is config for state in raw._states.values())
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("edge", ["before", "after"])
async def test_closed_screen_refuses_warm_display_acceptance(tmp_path, edge):
    database, _store, _controller, screen, tasks = _screen(tmp_path)
    retry, rendered = [], []
    screen._request_console_control_bar_sync = lambda **_: retry.append(True)
    try:
        await _warm(screen, tasks)
        screen._closing = edge == "before"

        def render():
            rendered.append(True)
            screen._closing = True

        with _actual_calls() as calls:
            assert ChatScreen._run_console_config_sync(screen, render) is False
        assert rendered == ([] if edge == "before" else [True])
        assert retry == [True] and calls["main_scopes"] == calls["main_opens"] == 0
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_unexpected_post_display_failure_chains_original_body_error(
    tmp_path, monkeypatch
):
    database, store, _controller, screen, tasks = _screen(tmp_path)
    body_error, post_error = (
        RuntimeError("display-body"),
        RuntimeError("display-owner-reader"),
    )
    try:
        await _warm(screen, tasks)

        def failed_owner_read():
            raise post_error

        def render():
            # This is the UI-only owner accessor, not an installed source,
            # storage, guard, or native function.
            monkeypatch.setattr(store, "sessions", failed_owner_read)
            raise body_error

        with pytest.raises(RuntimeError) as caught:
            ChatScreen._run_console_config_sync(screen, render)
        assert caught.value is post_error and caught.value.__cause__ is body_error
        assert not any(state.source is config for state in raw._states.values())
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_permissive_alias_equality_cannot_retain_display_proof(
    tmp_path, monkeypatch
):
    database, _store, _controller, screen, tasks = _screen(tmp_path)
    rendered, retry = [], []
    screen._request_console_control_bar_sync = lambda **_: retry.append(True)
    try:
        await _warm(screen, tasks)
        original = config.current_config_identity

        class EqualAlias:
            def __call__(self):
                return original()

            def __eq__(self, _other):
                return True

        # The lexical UI key is an unguarded reader. Every native source guard
        # and guarded config reader retains its installed original identity.
        monkeypatch.setattr(config, "current_config_identity", EqualAlias())
        with _actual_calls() as calls:
            assert (
                ChatScreen._run_console_config_sync(
                    screen, lambda: rendered.append(True)
                )
                is False
            )
        assert rendered == [] and retry == [True]
        assert calls["main_scopes"] == calls["main_opens"] == 0
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_borrowed_same_class_key_requires_original_native_lifetime(tmp_path):
    database, _store, _controller, screen, tasks = _screen(tmp_path)
    observed = []
    try:
        projection = await _warm(screen, tasks)
        other = spend.ConsoleReadinessConfigProjection(
            screen, read_current=projection.read_current
        )
        projection._key = other._key
        with _actual_calls() as calls:
            assert ChatScreen._run_console_config_sync(
                screen, lambda: observed.append(getattr(raw._local, "operation", None))
            )
        assert len(observed) == 1 and observed[0] is not None
        assert calls["main_scopes"] == 1 and calls["main_opens"] > 0
        assert observed[0] not in raw._states
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["app", "database", "store"])
async def test_permissive_owner_equality_never_accepts_changed_receiver(
    tmp_path, field
):
    database, _store, _controller, screen, tasks = _screen(tmp_path)
    rendered, retry = [], []
    screen._request_console_control_bar_sync = lambda **_: retry.append(True)
    try:
        await _warm(screen, tasks)

        class EqualOwner:
            def __init__(self, original):
                self.original = original

            def __getattr__(self, name):
                return getattr(self.original, name)

            def __eq__(self, _other):
                return True

        if field == "app":
            screen.app_instance = EqualOwner(screen.app_instance)
        elif field == "database":
            screen.app_instance.chachanotes_db = EqualOwner(database)
        else:
            screen._console_chat_store = EqualOwner(screen._console_chat_store)
        with _actual_calls() as calls:
            assert (
                ChatScreen._run_console_config_sync(
                    screen, lambda: rendered.append(True)
                )
                is False
            )
        assert rendered == [] and calls["main_scopes"] == calls["main_opens"] == 0
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unhashable", "unexpected_hash"])
async def test_arbitrary_injected_proof_stays_on_fresh_native_route(tmp_path, kind):
    database, _store, _controller, screen, tasks = _screen(tmp_path)
    observed = []
    try:
        projection = await _warm(screen, tasks)

        class UnexpectedHash:
            def __hash__(self):
                raise RuntimeError(
                    "an arbitrary object must not participate in proof lookup"
                )

        projection._display_proof = [] if kind == "unhashable" else UnexpectedHash()
        with _actual_calls() as calls:
            assert ChatScreen._run_console_config_sync(
                screen, lambda: observed.append(getattr(raw._local, "operation", None))
            )
        assert len(observed) == 1 and observed[0] is not None
        assert calls["main_scopes"] == 1 and calls["main_opens"] > 0
        assert observed[0] not in raw._states
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_borrowed_standard_projection_defers_actual_other_screen(tmp_path):
    from types import SimpleNamespace

    database, _store, _controller, screen, tasks = _screen(tmp_path)
    rendered = []
    try:
        projection = await _warm(screen, tasks)
        other_screen = SimpleNamespace(**vars(screen))
        assert other_screen._console_readiness_config_projection is projection
        with _actual_calls() as calls:
            assert (
                ChatScreen._run_console_config_sync(
                    other_screen, lambda: rendered.append(True)
                )
                is False
            )
        assert rendered == [] and calls["main_scopes"] == calls["main_opens"] == 0
        assert other_screen._console_readiness_config_projection is not projection
        await asyncio.gather(*tasks)
        assert other_screen._console_readiness_config_projection.screen is other_screen
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_retargeted_projection_cannot_reuse_original_screen_proof(tmp_path):
    from types import SimpleNamespace

    database, _store, _controller, screen, tasks = _screen(tmp_path)
    rendered, retry = [], []
    try:
        projection = await _warm(screen, tasks)
        other_screen = SimpleNamespace(**vars(screen))
        other_screen._request_console_control_bar_sync = lambda **_: retry.append(True)
        projection.screen = other_screen
        with _actual_calls() as calls:
            assert (
                ChatScreen._run_console_config_sync(
                    other_screen, lambda: rendered.append(True)
                )
                is False
            )
        assert rendered == [] and retry == [True]
        assert calls["main_scopes"] == calls["main_opens"] == 0
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_spoofed_bound_key_attributes_retain_fresh_native_scope(tmp_path):
    database, _store, _controller, screen, tasks = _screen(tmp_path)
    observed = []
    try:
        projection = await _warm(screen, tasks)
        original = projection._key

        class SpoofedKey:
            def __init__(self):
                self.__self__ = projection
                self.__func__ = original.__func__

            def __call__(self):
                return original()

        projection._key = SpoofedKey()
        with _actual_calls() as calls:
            assert ChatScreen._run_console_config_sync(
                screen, lambda: observed.append(getattr(raw._local, "operation", None))
            )
        assert len(observed) == 1 and observed[0] is not None
        assert calls["main_scopes"] == 1 and calls["main_opens"] > 0
        assert observed[0] not in raw._states
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()
