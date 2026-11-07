"""The deferred first-send fact survives another native settings write."""

import asyncio
import threading
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_reference_custody import (
    reference_host as reference_host,  # noqa: PLC0414
)
from Tests.windows_private_fixture_runner import user_fixture_default_owner
from tldw_chatbook import config
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime


@pytest.mark.bootstrap_profile
def test_first_send_merges_intervening_native_setting():
    async def exercise():
        app = SimpleNamespace(app_config={"console": {"onboarding": {}}})
        runtime = ConsoleRuntime.__new__(ConsoleRuntime)
        runtime._app = app
        app.console_runtime = runtime
        runtime._chat_controller = SimpleNamespace(prompt_queue_registry=None)
        runtime._chat_store = object()
        runtime._execution_capacity_lock = threading.Lock()
        runtime._first_send_sources = 1
        runtime._first_send_publishing = False
        runtime._ordinary_turns = {}
        runtime._turn_custody = {}
        runtime._first_send_task = None
        runtime._first_send_pending_source = None
        runtime._first_send_loop = None
        runtime._first_send_wakeup = None
        source = (
            config.current_config_identity(),
            app.app_config,
            runtime._chat_controller,
            runtime._chat_store,
            app,
            config,
        )
        await runtime._record_successful_first_send(source)
        await asyncio.sleep(0)
        result = config.apply_settings_mutation_to_cli_config(
            {"general": {"users_name": "unrelated-write"}}
        )
        assert result.fully_applied
        runtime._first_send_sources = 0
        runtime._wake_first_send()
        await asyncio.shield(runtime._first_send_task)
        persisted = tomllib.loads(Path(config.current_config_identity()[1]).read_text())
        assert persisted["general"]["users_name"] == "unrelated-write"
        assert persisted["console"]["onboarding"]["first_send_completed"] is True

    with user_fixture_default_owner():
        asyncio.run(exercise())


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("cancel_owner", ["publisher", "intake"])
def test_held_native_writer_drains_before_cancelled_owner_releases(cancel_owner):
    async def exercise():
        app = SimpleNamespace(app_config={"console": {"onboarding": {}}})
        runtime = ConsoleRuntime.__new__(ConsoleRuntime)
        runtime._app = app
        app.console_runtime = runtime
        runtime._chat_controller = SimpleNamespace(prompt_queue_registry=None)
        runtime._chat_store = object()
        runtime._execution_capacity_lock = threading.Lock()
        runtime._first_send_sources = 0
        runtime._first_send_publishing = False
        runtime._ordinary_turns = {}
        runtime._turn_custody = {}
        runtime._first_send_task = None
        runtime._first_send_pending_source = None
        runtime._first_send_loop = None
        runtime._first_send_wakeup = None
        source = (
            config.current_config_identity(),
            app.app_config,
            runtime._chat_controller,
            runtime._chat_store,
            app,
            config,
        )
        entered, release = threading.Event(), threading.Event()
        code = config.apply_literal_settings_transaction_to_cli_config.__code__

        def observe(frame, event, _arg):
            if event == "call" and frame.f_code is code:
                entered.set()
                release.wait(10)

        threading.setprofile_all_threads(observe)
        try:
            await runtime._record_successful_first_send(source)
            assert await asyncio.to_thread(entered.wait, 10)
            publisher = runtime._first_send_task
            acquired = asyncio.Event()

            async def capture():
                async with runtime._first_send_source_scope_async():
                    acquired.set()

            waiter = asyncio.create_task(capture())
            await asyncio.sleep(0.05)
            (publisher if cancel_owner == "publisher" else waiter).cancel()
            await asyncio.sleep(0.05)
            assert runtime._first_send_publishing and runtime._first_send_sources == 0
            assert not acquired.is_set() and not publisher.done()
            release.set()
            if cancel_owner == "publisher":
                with pytest.raises(asyncio.CancelledError):
                    await publisher
                await waiter
                assert acquired.is_set()
            else:
                with pytest.raises(asyncio.CancelledError):
                    await waiter
                await publisher
                assert not acquired.is_set()
            assert runtime._first_send_pending_source is None
            assert not runtime._first_send_publishing
        finally:
            release.set()
            threading.setprofile_all_threads(None)

    with user_fixture_default_owner():
        asyncio.run(exercise())


@pytest.mark.bootstrap_profile
def test_failed_native_write_retries_original_intent_on_later_completion():
    async def exercise():
        app = SimpleNamespace(app_config={"console": {"onboarding": {}}})
        runtime = ConsoleRuntime.__new__(ConsoleRuntime)
        runtime._app = app
        app.console_runtime = runtime
        runtime._chat_controller = SimpleNamespace(prompt_queue_registry=None)
        runtime._chat_store = object()
        runtime._execution_capacity_lock = threading.Lock()
        runtime._first_send_sources = 1
        runtime._first_send_publishing = False
        runtime._ordinary_turns = {}
        runtime._turn_custody = {}
        runtime._first_send_task = None
        runtime._first_send_pending_source = None
        runtime._first_send_loop = None
        runtime._first_send_wakeup = None
        source = (
            config.current_config_identity(),
            app.app_config,
            runtime._chat_controller,
            runtime._chat_store,
            app,
            config,
        )
        path = Path(source[0][1])
        previous = path.read_text()
        await runtime._record_successful_first_send(source)
        path.write_text("[broken\n")
        runtime._first_send_sources = 0
        runtime._wake_first_send()
        await runtime._first_send_task
        first_task = runtime._first_send_task
        assert runtime._first_send_pending_source is source
        path.write_text(previous)
        await runtime._record_successful_first_send(tuple(source))
        assert runtime._first_send_task is not first_task
        await runtime._first_send_task
        assert runtime._first_send_pending_source is None
        assert (
            tomllib.loads(path.read_text())["console"]["onboarding"][
                "first_send_completed"
            ]
            is True
        )

    with user_fixture_default_owner():
        asyncio.run(exercise())


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("cancel_owner", ["publisher", "disposer"])
def test_disposal_waits_for_held_native_writer_under_cancellation(cancel_owner):
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    async def exercise():
        app = SimpleNamespace(app_config={"console": {"onboarding": {}}})
        runtime = ConsoleRuntime(app)
        app.console_runtime = runtime
        store = ConsoleChatStore()
        runtime.set_chat_store(store)
        source = (
            config.current_config_identity(),
            app.app_config,
            runtime._chat_controller,
            store,
            app,
            config,
        )
        entered, release, store_shutdown = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        writer_code = config.apply_literal_settings_transaction_to_cli_config.__code__
        store_code = ConsoleChatStore.end_app_runtime.__code__

        def observe(frame, event, _arg):
            if event == "call" and frame.f_code is writer_code:
                entered.set()
                release.wait(15)
            if event == "call" and frame.f_code is store_code:
                store_shutdown.set()

        threading.setprofile_all_threads(observe)
        try:
            await runtime._record_successful_first_send(source)
            assert await asyncio.to_thread(entered.wait, 10)
            publisher = runtime._first_send_task
            disposer = asyncio.create_task(runtime.dispose())
            await asyncio.sleep(0.1)
            (publisher if cancel_owner == "publisher" else disposer).cancel()
            await asyncio.sleep(0.1)
            assert not store_shutdown.is_set() and not disposer.done()
            assert runtime._first_send_publishing and not publisher.done()
            release.set()
            if cancel_owner == "disposer":
                with pytest.raises(asyncio.CancelledError):
                    await disposer
            else:
                await disposer
            assert store_shutdown.is_set()
            assert runtime._first_send_pending_source is None
            assert not runtime._first_send_publishing
        finally:
            release.set()
            threading.setprofile_all_threads(None)

    with user_fixture_default_owner():
        asyncio.run(exercise())


@pytest.mark.bootstrap_profile
def test_selected_path_cannot_change_during_native_writer(tmp_path):
    import os

    async def exercise():
        app = SimpleNamespace(app_config={"console": {"onboarding": {}}})
        runtime = ConsoleRuntime(app)
        app.console_runtime = runtime
        source = (
            config.current_config_identity(),
            app.app_config,
            runtime._chat_controller,
            runtime._chat_store,
            app,
            config,
        )
        original = Path(source[0][1])
        other = tmp_path / "other.toml"
        other.write_text(original.read_text())
        seen = []
        builders = []
        writer_code = config.apply_literal_settings_transaction_to_cli_config.__code__
        lock_code = config._config_write_lock.__wrapped__.__code__

        def observe(frame, event, _arg):
            if event == "call" and frame.f_code is writer_code:
                os.environ["TLDW_CONFIG_PATH"] = str(other)
            elif event == "call" and frame.f_code is lock_code:
                seen.append(frame.f_locals["config_path"])
                os.environ["TLDW_CONFIG_PATH"] = str(original)
            elif (
                event == "call"
                and frame.f_code.co_name == "build"
                and frame.f_code.co_filename.endswith("/Chat/console_runtime.py")
            ):
                builders.append(frame.f_locals["snapshot"].config_path)

        threading.setprofile_all_threads(observe)
        try:
            await runtime._record_successful_first_send(source)
            await runtime._first_send_task
            assert seen == [other]
            assert not builders  # Native raw ownership refused B before the builder.
            assert runtime._first_send_pending_source is source
            for path in (original, other):
                assert (
                    not tomllib.loads(path.read_text())
                    .get("console", {})
                    .get("onboarding", {})
                    .get("first_send_completed")
                )
        finally:
            os.environ["TLDW_CONFIG_PATH"] = str(original)
            threading.setprofile_all_threads(None)

    with user_fixture_default_owner():
        asyncio.run(exercise())


@pytest.mark.bootstrap_profile
def test_cache_failure_after_replace_retries_as_locked_noop(monkeypatch):
    async def exercise():
        app = SimpleNamespace(app_config={"console": {"onboarding": {}}})
        runtime = ConsoleRuntime(app)
        app.console_runtime = runtime
        source = (
            config.current_config_identity(),
            app.app_config,
            runtime._chat_controller,
            runtime._chat_store,
            app,
            config,
        )
        original_load = config._load_settings_uncached
        writes = []
        write_code = config._write_raw_cli_config_unlocked.__wrapped__.__code__

        def observe(frame, event, _arg):
            if event == "call" and frame.f_code is write_code:
                writes.append(True)

        def fail_reload(*args, **kwargs):
            raise RuntimeError("fixture cache reload failure")

        threading.setprofile_all_threads(observe)
        monkeypatch.setattr(config, "_load_settings_uncached", fail_reload)
        try:
            await runtime._record_successful_first_send(source)
            await runtime._first_send_task
            assert runtime._first_send_pending_source is source
            path = Path(source[0][1])
            assert (
                tomllib.loads(path.read_text())["console"]["onboarding"][
                    "first_send_completed"
                ]
                is True
            )
            monkeypatch.setattr(config, "_load_settings_uncached", original_load)
            await runtime._record_successful_first_send(source)
            await runtime._first_send_task
            assert runtime._first_send_pending_source is None
            assert len(writes) == 1
        finally:
            threading.setprofile_all_threads(None)

    with user_fixture_default_owner():
        asyncio.run(exercise())


@pytest.mark.bootstrap_profile
def test_original_locked_builder_rejects_different_snapshot_path(tmp_path):
    from dataclasses import replace

    async def exercise():
        app = SimpleNamespace(app_config={"console": {"onboarding": {}}})
        runtime = ConsoleRuntime(app)
        app.console_runtime = runtime
        source = (
            config.current_config_identity(),
            app.app_config,
            runtime._chat_controller,
            runtime._chat_store,
            app,
            config,
        )
        other = tmp_path / "other.toml"
        refusals = []

        def observe(frame, event, _arg):
            if (
                event == "call"
                and frame.f_code.co_name == "build"
                and frame.f_code.co_filename.endswith("/Chat/console_runtime.py")
            ):
                snapshot = frame.f_locals["snapshot"]
                assert str(snapshot.config_path) == source[0][1]
                builder = frame.f_back.f_locals["mutation_builder"]
                try:
                    builder(replace(snapshot, config_path=other))
                except RuntimeError:
                    refusals.append(True)
                else:
                    refusals.append(False)

        threading.setprofile_all_threads(observe)
        try:
            await runtime._record_successful_first_send(source)
            await runtime._first_send_task
            assert refusals == [True]
            assert runtime._first_send_pending_source is None
            assert not other.exists()
        finally:
            threading.setprofile_all_threads(None)

    with user_fixture_default_owner():
        asyncio.run(exercise())


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize(
    "replace_owner", ["app", "app_config", "runtime", "controller", "store", "writer"]
)
def test_replaced_owner_refuses_before_native_writer(monkeypatch, replace_owner):
    async def exercise():
        app = SimpleNamespace(app_config={"console": {"onboarding": {}}})
        runtime = ConsoleRuntime(app)
        app.console_runtime = runtime
        runtime._first_send_sources = 1
        source = (
            config.current_config_identity(),
            app.app_config,
            runtime._chat_controller,
            runtime._chat_store,
            app,
            config,
        )
        path = Path(source[0][1])
        before = path.read_bytes()
        invoked = []
        await runtime._record_successful_first_send(source)
        await asyncio.sleep(0)
        if replace_owner == "app":
            runtime._app = SimpleNamespace(app_config=app.app_config)
        elif replace_owner == "app_config":
            app.app_config = {}
        elif replace_owner == "runtime":
            app.console_runtime = object()
        elif replace_owner == "controller":
            runtime._chat_controller = object()
        elif replace_owner == "store":
            runtime._chat_store = object()
        else:
            monkeypatch.setattr(
                config,
                "apply_literal_settings_transaction_to_cli_config",
                lambda *_a, **_k: invoked.append(True),
            )
        runtime._first_send_sources = 0
        runtime._wake_first_send()
        await runtime._first_send_task
        assert runtime._first_send_pending_source is source
        assert not invoked and path.read_bytes() == before

    with user_fixture_default_owner():
        asyncio.run(exercise())


@pytest.mark.bootstrap_profile
def test_writer_slot_replacement_after_initial_check_is_never_called(monkeypatch):
    import sys

    async def exercise():
        app = SimpleNamespace(app_config={"console": {"onboarding": {}}})
        runtime = ConsoleRuntime(app)
        app.console_runtime = runtime
        source = (
            config.current_config_identity(),
            app.app_config,
            runtime._chat_controller,
            runtime._chat_store,
            app,
            config,
        )
        path = Path(source[0][1])
        before = path.read_bytes()
        changed, invoked = [], []

        def foreign(*_args, **_kwargs):
            invoked.append(True)

        def observe(frame, event, value):
            if (
                not changed
                and event == "return"
                and value is True
                and frame.f_code.co_name == "current"
                and frame.f_code.co_filename.endswith("/Chat/console_runtime.py")
            ):
                changed.append(True)
                monkeypatch.setattr(
                    config, "apply_literal_settings_transaction_to_cli_config", foreign
                )

        sys.setprofile(observe)
        try:
            await runtime._record_successful_first_send(source)
            await runtime._first_send_task
            assert changed == [True]
            assert not invoked and path.read_bytes() == before
            assert runtime._first_send_pending_source is source
        finally:
            sys.setprofile(None)

    with user_fixture_default_owner():
        asyncio.run(exercise())


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("repair", [True, False])
def test_disposal_makes_one_final_retry_without_spinning(repair):
    async def exercise():
        app = SimpleNamespace(app_config={"console": {"onboarding": {}}})
        runtime = ConsoleRuntime(app)
        app.console_runtime = runtime
        runtime._first_send_sources = 1
        source = (
            config.current_config_identity(),
            app.app_config,
            runtime._chat_controller,
            runtime._chat_store,
            app,
            config,
        )
        path = Path(source[0][1])
        before = path.read_text()
        attempts = []
        code = config.apply_literal_settings_transaction_to_cli_config.__code__

        def observe(frame, event, _arg):
            if event == "call" and frame.f_code is code:
                attempts.append(True)

        threading.setprofile_all_threads(observe)
        try:
            await runtime._record_successful_first_send(source)
            path.write_text("[broken\n")
            runtime._first_send_sources = 0
            runtime._wake_first_send()
            await runtime._first_send_task
            assert runtime._first_send_pending_source is source
            if repair:
                path.write_text(before)
            await runtime.dispose()
            assert len(attempts) == 2
            if repair:
                assert runtime._first_send_pending_source is None
                assert (
                    tomllib.loads(path.read_text())["console"]["onboarding"][
                        "first_send_completed"
                    ]
                    is True
                )
            else:
                assert runtime._first_send_pending_source is source
        finally:
            threading.setprofile_all_threads(None)
            if not repair:
                path.write_text(before)

    with user_fixture_default_owner():
        asyncio.run(exercise())


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_direct_accepted_turn_uses_receipt_tuple_and_persists(reference_host):
    import sys

    from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus
    from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest

    h = reference_host
    configuration = await h.controller.capture_turn_configuration_snapshot(h.session.id)
    identity = config.current_config_identity()
    expected = (h.app.app_config, h.controller, h.store, h.app, config)
    observed = []
    code = type(h.runtime)._run_custodied_turn.__code__

    def observe(frame, event, _arg):
        if event == "call" and frame.f_code is code and not observed:
            observed.append(frame.f_locals["first_send_source"])

    sys.setprofile(observe)
    try:
        request = ConsoleTurnCustodyRequest(
            turn_id="direct-first-send",
            session_id=h.session.id,
            draft="direct first send",
            configuration=configuration,
        )
        turn = h.runtime.accept_turn(request)
        task = h.runtime._turn_custody[turn].task
        h.tasks.append(task)
        result = await task
        assert result.accepted
        assert (
            h.controller.run_state_for(h.session.id).status
            is ConsoleRunStatus.COMPLETED
        )
        assert len(observed) == 1 and observed[0][0] == identity
        assert all(
            actual is wanted for actual, wanted in zip(observed[0][1:], expected)
        )
        await asyncio.wait_for(asyncio.shield(h.runtime._first_send_task), 10)
        assert h.app.app_config["console"]["onboarding"]["first_send_completed"] is True
        assert (
            tomllib.loads(Path(identity[1]).read_text())["console"]["onboarding"][
                "first_send_completed"
            ]
            is True
        )
    finally:
        sys.setprofile(None)


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_owner", ["publisher", "intake", "both"])
async def test_real_controller_waits_for_cancelled_publication(
    reference_host, cancel_owner
):
    from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus
    from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest

    h = reference_host
    configuration = await h.controller.capture_turn_configuration_snapshot(h.session.id)
    second = h.store.create_session(workspace_id=h.session.workspace_id)
    entered, release = threading.Event(), threading.Event()
    writer_code = config.apply_literal_settings_transaction_to_cli_config.__code__
    lifecycle_code = type(h.controller)._submit_draft_lifecycle.__code__
    lifecycle = []

    def observe(frame, event, _arg):
        if event == "call" and frame.f_code is writer_code:
            entered.set()
            release.wait(15)
        elif event == "call" and frame.f_code is lifecycle_code:
            lifecycle.append((frame.f_locals.get("session_id"), id(frame)))

    threading.setprofile_all_threads(observe)
    try:
        request = ConsoleTurnCustodyRequest(
            turn_id="first-before-waiter",
            session_id=h.session.id,
            draft="first",
            configuration=configuration,
        )
        turn_id = h.runtime.accept_turn(request)
        first = h.runtime._turn_custody[turn_id].task
        h.tasks.append(first)
        result = await first
        assert result.accepted and result.terminal_status is ConsoleRunStatus.COMPLETED
        assert await asyncio.to_thread(entered.wait, 10)
        publisher = h.runtime._first_send_task
        waiter = asyncio.create_task(
            h.controller.submit_draft("second", session_id=second.id)
        )
        await asyncio.sleep(0.1)
        assert not waiter.done() and h.runtime._first_send_sources == 0
        assert not any(session_id == second.id for session_id, _ in lifecycle)
        if cancel_owner in {"publisher", "both"}:
            publisher.cancel()
        if cancel_owner in {"intake", "both"}:
            waiter.cancel()
        await asyncio.sleep(0.05)
        assert h.runtime._first_send_publishing and not publisher.done()
        assert h.runtime._first_send_sources == 0 and not any(
            session_id == second.id for session_id, _ in lifecycle
        )
        release.set()
        if cancel_owner == "publisher":
            with pytest.raises(asyncio.CancelledError):
                await publisher
            second_result = await waiter
            assert (
                second_result.accepted
                and second_result.terminal_status is ConsoleRunStatus.COMPLETED
            )
            assert (
                len(
                    {
                        frame_id
                        for session_id, frame_id in lifecycle
                        if session_id == second.id
                    }
                )
                == 1
            )
        else:
            with pytest.raises(asyncio.CancelledError):
                await waiter
            if cancel_owner == "both":
                with pytest.raises(asyncio.CancelledError):
                    await publisher
            else:
                await publisher
            assert not any(session_id == second.id for session_id, _ in lifecycle)
        assert h.runtime._first_send_pending_source is None
    finally:
        release.set()
        threading.setprofile_all_threads(None)


from Tests.Chat import test_console_received_intent_transitions as received_fixtures

catalog_store = received_fixtures.catalog_store
local_root = received_fixtures.local_root
mcp_sources = received_fixtures.mcp_sources
snapshot_case = received_fixtures.snapshot_case
runtime_case = received_fixtures.runtime_case
configuration_workspace = received_fixtures.configuration_workspace
received_workspace = received_fixtures.received_workspace


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_received_owner_replacement_before_promotion_refuses(
    received_workspace, monkeypatch
):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    case = received_workspace
    runtime, store, session, controller = (
        case.runtime,
        case.store,
        case.session,
        case.controller,
    )
    store.set_session_draft(session.id, "original received draft")
    original_capture = controller.capture_turn_configuration_snapshot
    entered, release = asyncio.Event(), asyncio.Event()

    async def held_capture(*args, **kwargs):
        result = await original_capture(*args, **kwargs)
        entered.set()
        await release.wait()
        return result

    monkeypatch.setattr(controller, "capture_turn_configuration_snapshot", held_capture)
    intent = received_fixtures._stock_intent(case)
    runtime.accept_received_intent(intent)
    record = runtime._turn_custody[intent.turn_id]
    original_app = runtime._app
    try:
        await received_fixtures._wait_for_seam(record, entered)
        assert record.request is None
        runtime._app = SimpleNamespace(app_config=original_app.app_config)
        release.set()
        result = await received_fixtures._outcome(record)
        assert isinstance(result, RecoveryRequired)
        assert record.request is None
        assert store.session_draft(session.id) == intent.inputs.draft
        assert not store.messages_for_session(session.id)
    finally:
        runtime._app = original_app
        release.set()


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("replacement", ["new_source", "selected_path"])
def test_pending_original_is_not_transferred_or_reselected(
    monkeypatch, tmp_path, replacement
):
    async def exercise():
        app = SimpleNamespace(app_config={"console": {"onboarding": {}}})
        runtime = ConsoleRuntime(app)
        app.console_runtime = runtime
        runtime._first_send_sources = 1
        source = (
            config.current_config_identity(),
            app.app_config,
            runtime._chat_controller,
            runtime._chat_store,
            app,
            config,
        )
        original_path = Path(source[0][1])
        before = original_path.read_bytes()
        writes = []
        code = config.apply_literal_settings_transaction_to_cli_config.__code__

        def observe(frame, event, _arg):
            if event == "call" and frame.f_code is code:
                writes.append(True)

        threading.setprofile_all_threads(observe)
        try:
            await runtime._record_successful_first_send(source)
            if replacement == "new_source":
                other_app = SimpleNamespace(app_config={"console": {"onboarding": {}}})
                other_app.console_runtime = runtime
                runtime._app = other_app
            else:
                other = tmp_path / "other.toml"
                monkeypatch.setenv("TLDW_CONFIG_PATH", str(other))
            runtime._first_send_sources = 0
            runtime._wake_first_send()
            await runtime._first_send_task
            assert runtime._first_send_pending_source is source
            if replacement == "new_source":
                other_source = (
                    config.current_config_identity(),
                    other_app.app_config,
                    runtime._chat_controller,
                    runtime._chat_store,
                    other_app,
                    config,
                )
                await runtime._record_successful_first_send(other_source)
                await runtime._first_send_task
                assert runtime._first_send_pending_source is source
                assert not other_app.app_config["console"]["onboarding"]
            else:
                assert not other.exists()
            assert not writes and original_path.read_bytes() == before
        finally:
            monkeypatch.setenv("TLDW_CONFIG_PATH", str(original_path))
            threading.setprofile_all_threads(None)

    with user_fixture_default_owner():
        asyncio.run(exercise())


@pytest.mark.bootstrap_profile
def test_cancelled_publisher_restores_cancellation_after_native_writer_raises(
    monkeypatch,
):
    import contextlib
    import sys

    async def exercise():
        app = SimpleNamespace(app_config={"console": {"onboarding": {}}})
        runtime = ConsoleRuntime(app)
        app.console_runtime = runtime
        source = (
            config.current_config_identity(),
            app.app_config,
            runtime._chat_controller,
            runtime._chat_store,
            app,
            config,
        )
        entered, release, returned = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        writer_code = config.apply_literal_settings_transaction_to_cli_config.__code__
        transaction_code = config._apply_literal_settings_transaction_locked.__code__
        original_exit = contextlib.ExitStack.__exit__
        faults = []

        def failing_exit(stack, *args):
            caller = sys._getframe(1)
            result = original_exit(stack, *args)
            if caller.f_code is transaction_code:
                faults.append(True)
                returned.set()
                raise RuntimeError("fixture failure after native lock exit")
            return result

        def observe(frame, event, _arg):
            if event == "call" and frame.f_code is writer_code:
                entered.set()
                release.wait(10)

        monkeypatch.setattr(contextlib.ExitStack, "__exit__", failing_exit)
        threading.setprofile_all_threads(observe)
        waiter = None
        try:
            await runtime._record_successful_first_send(source)
            assert await asyncio.to_thread(entered.wait, 10)
            publisher = runtime._first_send_task
            acquired = asyncio.Event()

            async def capture():
                async with runtime._first_send_source_scope_async():
                    acquired.set()

            waiter = asyncio.create_task(capture())
            publisher.cancel()
            await asyncio.sleep(0.05)
            assert runtime._first_send_publishing and not publisher.done()
            assert not returned.is_set() and not acquired.is_set()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await publisher
            assert returned.is_set() and faults == [True]
            assert runtime._first_send_pending_source is source
            assert not runtime._first_send_publishing
            await waiter
            assert acquired.is_set()
        finally:
            release.set()
            if waiter is not None:
                await asyncio.gather(waiter, return_exceptions=True)
            monkeypatch.setattr(contextlib.ExitStack, "__exit__", original_exit)
            threading.setprofile_all_threads(None)
            await runtime.dispose()
        assert runtime._first_send_pending_source is None

    with user_fixture_default_owner():
        asyncio.run(exercise())


@pytest.mark.bootstrap_profile
def test_cancelled_publisher_retains_intent_on_native_typed_failure():
    async def exercise():
        app = SimpleNamespace(app_config={"console": {"onboarding": {}}})
        runtime = ConsoleRuntime(app)
        app.console_runtime = runtime
        source = (
            config.current_config_identity(),
            app.app_config,
            runtime._chat_controller,
            runtime._chat_store,
            app,
            config,
        )
        path = Path(source[0][1])
        before = path.read_text()
        entered, release = threading.Event(), threading.Event()
        code = config.apply_literal_settings_transaction_to_cli_config.__code__

        def observe(frame, event, _arg):
            if event == "call" and frame.f_code is code:
                entered.set()
                release.wait(10)

        threading.setprofile_all_threads(observe)
        try:
            await runtime._record_successful_first_send(source)
            assert await asyncio.to_thread(entered.wait, 10)
            publisher = runtime._first_send_task
            path.write_text("[broken\n")
            publisher.cancel()
            assert runtime._first_send_publishing and not publisher.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await publisher
            assert runtime._first_send_pending_source is source
            assert not runtime._first_send_publishing
        finally:
            release.set()
            path.write_text(before)
            threading.setprofile_all_threads(None)
            await runtime.dispose()
        assert runtime._first_send_pending_source is None

    with user_fixture_default_owner():
        asyncio.run(exercise())
