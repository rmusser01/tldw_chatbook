"""Actual Task61 display producers without a view and on their original consumer."""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
import sqlite3
import sys

import pytest

from Tests.Chat.test_console_control_trace import _observe
from Tests.Chat.test_console_selected_mounted_display import selected_mounted_console  # noqa: F401 - Pytest registers this fixture.

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


def _assert_owned_retirement(full, *, success=True):
    assert full.phase_issuance_closed and full.physically_retired
    assert full.logical.retired.done() and full.logical.callback_returned is success
    assert full.effect.completed.done() and full.effect.completed.result() is True
    issued = [phase for phase in full.retirements if phase.native_attempted]
    assert issued
    for phase in issued:
        state = phase.retirement
        assert phase.joined and state.confirmed and state.borrowed is None
        assert state.creator is not None and state.operation is not None and state.holds
    opened = [
        phase.retirement.connection
        for phase in issued
        if phase.retirement.connection is not None
    ]
    assert opened
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
    assert all(task.done() for task in full.tasks.values())


@contextmanager
def _phase_observer(owner, request):
    from tldw_chatbook.Chat.console_display_phases import ConsoleDisplayPhysicalOwner

    rows = []

    def returned(frame):
        full = frame.f_locals["self"]
        if full.owner is not owner:
            return
        phase = frame.f_locals["phase"]
        state = phase.retirement
        rows.append(
            {
                "read": frame.f_locals["read"].__qualname__,
                "result_type": type(frame.f_locals.get("value")).__name__,
                "attempted": phase.native_attempted,
                "joined": phase.joined,
                "connection": state.connection is not None,
                "creator": state.creator is not None,
                "operation": state.operation is not None,
                "holds": len(state.holds),
                "confirmed": state.confirmed,
            }
        )

    with _observe(
        ConsoleDisplayPhysicalOwner.native_phase.__code__,
        sys.monitoring.events.PY_RETURN,
        returned,
    ):
        try:
            yield
        finally:
            request.node.user_properties.append(("display_phases", json.dumps(rows)))


@pytest.mark.parametrize("selected_mounted_console", ["home"], indirect=True)
async def test_viewless_projection_retires_original_nonempty_display_phases(
    selected_mounted_console,  # noqa: F811 - Pytest injects the imported fixture.
    request,
):
    from tldw_chatbook.Chat.console_control_dispatch import (
        ControlIdentity,
        ConsoleLocalControls,
        ReadProjection,
    )
    import pytest_timeout

    app, runtime, _screen, pilot = selected_mounted_console
    controls = runtime._execution_service._local_controls
    store = runtime.ensure_chat_store()
    session = store.create_session(title="Detached display")
    owner = runtime._execution_service._display_owner
    assert runtime.view is None and owner.context.screen is None
    identity = ControlIdentity(
        "resume-none",
        "local",
        1,
        "display-viewless",
        session.id,
        session.incarnation_id,
        controls.source_revision,
    )
    completed = []

    def finish(frame):
        full = frame.f_locals["full"]
        if full.owner is owner:
            completed.append(full)

    with (
        _phase_observer(owner, request),
        _observe(
            ConsoleLocalControls._finish_display_effect.__code__,
            sys.monitoring.events.PY_RETURN,
            finish,
        ),
    ):
        assert (
            not pytest_timeout.is_debugging()
            and float(request.config.getini("timeout")) == 300
        )
        request.node.user_properties.append(
            ("effective_observer_watchdog", "300/debugger=False")
        )
        receipt = controls.dispatch(ReadProjection(identity))
        first_error = None
        try:
            async with asyncio.timeout(10):
                projection = await asyncio.shield(receipt.applied)
            assert projection.session_id == session.id
            assert await receipt.retired is True
            assert len(completed) == 1
            full = completed[0]
            assert (
                full.selection.stamp is None
                and full.selection.command_source is not None
            )
            _assert_owned_retirement(full)
            assert owner._full is None and not controls.admitted_effects
            assert owner.character.state.scope_fingerprint is not None
            assert owner.context.value is not None
        except BaseException as error:
            first_error = error
            raise
        finally:
            try:
                async with asyncio.timeout(10):
                    await asyncio.shield(
                        asyncio.gather(receipt.work, return_exceptions=True)
                    )
            except BaseException:
                if first_error is None:
                    raise


@pytest.mark.parametrize("selected_mounted_console", ["home"], indirect=True)
async def test_preselected_original_navigation_attaches_the_same_display_producers(
    selected_mounted_console,  # noqa: F811 - Pytest injects the imported fixture.
    request,
):
    from tldw_chatbook.Chat.console_control_dispatch import ConsoleLocalControls
    from tldw_chatbook.Constants import TAB_CHAT
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from Tests.UI.test_console_native_chat_flow import _wait_for_selector
    import pytest_timeout

    app, runtime, _screen, pilot = selected_mounted_console
    controls = runtime._execution_service._local_controls  # noqa: F841 - Retain this tested selected-association read.
    owner = runtime._execution_service._display_owner
    character, context, appearance = owner.character, owner.context, owner.appearance
    assert runtime.view is None and runtime._chat_store.active_session_id is None
    completed = []

    def finish(frame):
        full = frame.f_locals["full"]
        if full.owner is owner and full.stamp is not None:
            completed.append(full)

    with (
        _phase_observer(owner, request),
        _observe(
            ConsoleLocalControls._finish_display_effect.__code__,
            sys.monitoring.events.PY_RETURN,
            finish,
        ),
    ):
        assert (
            not pytest_timeout.is_debugging()
            and float(request.config.getini("timeout")) == 300
        )
        request.node.user_properties.append(
            ("effective_observer_watchdog", "300/debugger=False")
        )
        await app.handle_screen_navigation(NavigateToScreen(TAB_CHAT))
        screen = app._navigation_outgoing_screen()
        assert type(screen) is ChatScreen
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        async with asyncio.timeout(10):
            while not any(full.logical.callback_returned for full in completed):
                if app._exception is not None:
                    raise app._exception
                await pilot.pause(0.05)
        full = next(full for full in completed if full.logical.callback_returned)
        _assert_owned_retirement(full)
        assert screen._character_context is character is owner.character
        assert (
            owner.context is context
            and screen._console_context_read_snapshot is context
        )
        assert owner.appearance is appearance
        assert (
            screen._workspace._console_appearance_map_queried_ids
            is appearance.queried_ids
        )
        assert full.selection.parent_entry.view is screen
        assert full.selection.parent_entry.task is full.selection.task
        assert app._exception is None
        request.node.user_properties.append(
            (
                "composition",
                "staged Home-first selected graph; original navigation/FULL; not startup qualification",
            )
        )


@contextmanager
def _appearance_refusal_observer(request):
    from types import CodeType, MethodType
    from tldw_chatbook.Chat.console_domain_display import ConsoleAppearanceDisplay
    from tldw_chatbook.Chat.console_received_dispatch import stock_native_methods
    from tldw_chatbook.UI.Console_Modules.workspace import (
        _APPEARANCE_DISPLAY_READERS,
        _APPEARANCE_DB_READERS,
    )

    code = next(
        value
        for value in ConsoleAppearanceDisplay.warm.__code__.co_consts
        if isinstance(value, CodeType) and value.co_name == "current"
    )
    rows = []

    def refused(frame, args):
        values = frame.f_locals
        state, service, database = values["self"], values["service"], values["database"]
        bound = values["get_appearances"]
        rows.append(
            {
                "error": type(args[-1]).__name__,
                "bound_method": type(bound) is MethodType,
                "bound_receiver": getattr(bound, "__self__", None) is service,
                "domain_local_service": state.owner.domain._conversation_owner[1]
                is service,
                "service_database": service.db is database,
                "memo_service": state.service is service,
                "memo_cache": state.cache is values["captured"],
                "memo_time": state.cached_at == values["cache_at"],
                "refresh_marker": state.refresh_for is values["captured"],
                "service_definition": stock_native_methods(
                    service, _APPEARANCE_DISPLAY_READERS
                ),
                "database_definition": stock_native_methods(
                    database, _APPEARANCE_DB_READERS
                ),
            }
        )

    with _observe(code, sys.monitoring.events.RAISE, refused, values=True):
        import pytest_timeout

        assert (
            not pytest_timeout.is_debugging()
            and float(request.config.getini("timeout")) == 300
        )
        request.node.user_properties.append(
            ("appearance_observer_watchdog", "300/debugger=False")
        )
        try:
            yield
        finally:
            request.node.user_properties.append(
                ("appearance_refusal_predicates", json.dumps(rows))
            )


async def _navigate_selected_chat(case):
    from tldw_chatbook.Constants import TAB_CHAT
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from Tests.UI.test_console_native_chat_flow import _wait_for_selector
    from Tests.Chat.test_console_selected_mounted_display import _whole_core_settled

    app, runtime, _screen, pilot = case
    await app.handle_screen_navigation(NavigateToScreen(TAB_CHAT))
    screen = app._navigation_outgoing_screen()
    await _wait_for_selector(screen, pilot, "#console-native-composer")
    await _whole_core_settled(
        app, screen, runtime._execution_service._local_controls, pilot
    )
    return app, runtime, screen, pilot


@pytest.mark.parametrize("selected_mounted_console", ["home"], indirect=True)
async def test_selected_standalone_settings_warm_joins_original_full(
    selected_mounted_console,  # noqa: F811 - Pytest injects the imported fixture.
    request,
):
    from tldw_chatbook.Chat.console_presentation_port import ConsolePresentationPort
    from tldw_chatbook.Widgets.Console.console_settings_modal import (
        ConsoleSettingsModal,
    )
    import pytest_timeout

    with _appearance_refusal_observer(request):
        app, runtime, screen, pilot = await _navigate_selected_chat(
            selected_mounted_console
        )
    owner = runtime._execution_service._display_owner
    demands = []

    def completed(frame):
        if frame.f_locals["owner"] is owner:
            demands.append(
                (tuple(frame.f_locals["entries"]), frame.f_locals.get("full"))
            )

    work, first_error = None, None
    try:
        # Original one-second presentation expiry, no cache/key/clock mutation.
        await asyncio.sleep(1.05)
        with (
            _phase_observer(owner, request),
            _observe(
                ConsolePresentationPort.warm_display_context.__code__,
                sys.monitoring.events.PY_RETURN,
                completed,
            ),
        ):
            assert (
                not pytest_timeout.is_debugging()
                and float(request.config.getini("timeout")) == 300
            )
            work = asyncio.create_task(screen._open_console_settings())
            async with asyncio.timeout(15):
                opened = await asyncio.shield(work)
            assert opened is True and isinstance(app.screen, ConsoleSettingsModal)
            assert (
                demands
            ), "Original Settings context warm did not reach selected FULL demand"
            fulls = [
                full
                for entries, full in demands
                if len(entries) == 1 and full is not None
            ]
            assert fulls
            for full in fulls:
                _assert_owned_retirement(full)
            assert app._exception is None
    except BaseException as error:
        first_error = error
        raise
    finally:
        if work is not None:
            try:
                async with asyncio.timeout(10):
                    await asyncio.shield(asyncio.gather(work, return_exceptions=True))
            except BaseException:
                if first_error is None:
                    raise
        if isinstance(app.screen, ConsoleSettingsModal):
            await app.pop_screen()


@pytest.mark.parametrize("selected_mounted_console", ["home"], indirect=True)
async def test_display_midpoint_session_switch_refuses_after_actual_retirement(
    selected_mounted_console,  # noqa: F811 - Pytest injects the imported fixture.
    request,
):
    from tldw_chatbook.Chat.console_control_dispatch import (
        ControlIdentity,
        ConsoleLocalControls,
        ReadProjection,
    )
    from tldw_chatbook.Chat.console_domain_display import ConsoleDisplayFull
    from tldw_chatbook.Chat.console_display_phases import ConsoleDisplayPhaseRefused
    import pytest_timeout

    app, runtime, _screen, pilot = selected_mounted_console
    controls = runtime._execution_service._local_controls
    store = runtime.ensure_chat_store()
    session = store.create_session(title="Original display")
    successor = store.create_session(title="Successor", activate=False)
    owner = runtime._execution_service._display_owner
    ages = (
        owner.character._presentation_scope_at,
        owner.context.at,
        owner.appearance.cached_at,
    )
    observed, completed = [], []

    def checkpoint(frame):
        full = frame.f_locals["self"]
        if full.owner is owner and not observed:
            assert full.physically_retired and full.retirements
            assert full.retirements[-1].retirement.connection is not None
            observed.append(full)
            store.switch_session(successor.id)

    def finished(frame):
        if frame.f_locals["full"].owner is owner:
            completed.append(frame.f_locals["full"])

    identity = ControlIdentity(
        "resume-none",
        "local",
        1,
        "display-midpoint",
        session.id,
        session.incarnation_id,
    )
    first_error = None
    with (
        _phase_observer(owner, request),
        _observe(
            ConsoleDisplayFull.character_checkpoint.__code__,
            sys.monitoring.events.PY_START,
            checkpoint,
        ),
        _observe(
            ConsoleLocalControls._finish_display_effect.__code__,
            sys.monitoring.events.PY_RETURN,
            finished,
        ),
    ):
        assert (
            not pytest_timeout.is_debugging()
            and float(request.config.getini("timeout")) == 300
        )
        receipt = controls.dispatch(ReadProjection(identity))
        try:
            with pytest.raises(ConsoleDisplayPhaseRefused):
                async with asyncio.timeout(10):
                    await asyncio.shield(receipt.applied)
            assert len(observed) == len(completed) == 1 and observed[0] is completed[0]
            assert await receipt.retired is True
            _assert_owned_retirement(completed[0], success=False)
            assert store.active_session_id == successor.id
            assert (
                owner.character._presentation_scope_at,
                owner.context.at,
                owner.appearance.cached_at,
            ) == ages
        except BaseException as error:
            first_error = error
            raise
        finally:
            try:
                async with asyncio.timeout(10):
                    await asyncio.shield(
                        asyncio.gather(receipt.work, return_exceptions=True)
                    )
            except BaseException:
                if first_error is None:
                    raise


@pytest.mark.parametrize("selected_mounted_console", ["home"], indirect=True)
async def test_cancelled_display_keeps_source_fence_behind_actual_native_owner(
    selected_mounted_console,  # noqa: F811 - Pytest injects the imported fixture.
    request,
):
    import threading
    from tldw_chatbook.Chat.console_control_dispatch import (
        ControlIdentity,
        ConsoleLocalControls,
        FenceSource,
        ReadProjection,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from Tests.Chat.test_console_control_settings import _entered
    import pytest_timeout

    app, runtime, _screen, pilot = selected_mounted_console
    controls = runtime._execution_service._local_controls
    store = runtime.ensure_chat_store()
    session = store.create_session(title="Held display")
    owner = runtime._execution_service._display_owner
    entered, release, expired = threading.Event(), threading.Event(), threading.Event()
    completed = []

    def metadata(frame):
        if (
            frame.f_locals.get("self") is owner.sources.chat_database
            and not entered.is_set()
        ):
            entered.set()
            if not release.wait(10):
                expired.set()

    def finished(frame):
        if frame.f_locals["full"].owner is owner:
            completed.append(frame.f_locals["full"])

    def identity(sequence):
        return ControlIdentity(
            "resume-none",
            "local",
            sequence,
            f"held-display-{sequence}",
            session.id,
            session.incarnation_id,
        )

    first_error, fence = None, None
    with (
        _phase_observer(owner, request),
        _observe(
            CharactersRAGDB.get_local_authority_id.__code__,
            sys.monitoring.events.PY_RETURN,
            metadata,
        ),
        _observe(
            ConsoleLocalControls._finish_display_effect.__code__,
            sys.monitoring.events.PY_RETURN,
            finished,
        ),
    ):
        assert (
            not pytest_timeout.is_debugging()
            and float(request.config.getini("timeout")) == 300
        )
        receipt = controls.dispatch(ReadProjection(identity(1)))
        try:
            await _entered(entered)
            full = owner._full
            assert (
                full is not None and full.phase_pending and not full.physically_retired
            )
            receipt.work.cancel()
            await asyncio.sleep(0)
            receipt.work.cancel()
            fence = controls.dispatch(FenceSource(identity(2), "configuration"))
            await asyncio.sleep(0)
            assert not fence.applied.done() and not receipt.retired.done()
            assert "display" in controls.admitted_effects
            release.set()
            with pytest.raises(asyncio.CancelledError):
                async with asyncio.timeout(10):
                    await asyncio.shield(receipt.applied)
            async with asyncio.timeout(10):
                await asyncio.shield(fence.applied)
            assert await receipt.retired is True and completed == [full]
            _assert_owned_retirement(full, success=False)
            assert not expired.is_set() and not controls.admitted_effects
        except BaseException as error:
            first_error = error
            raise
        finally:
            release.set()
            try:
                async with asyncio.timeout(10):
                    await asyncio.shield(
                        asyncio.gather(
                            receipt.work,
                            *((fence.work,) if fence is not None else ()),
                            return_exceptions=True,
                        )
                    )
            except BaseException:
                if first_error is None:
                    raise


@pytest.mark.parametrize("selected_mounted_console", ["home"], indirect=True)
async def test_original_full_worker_uses_its_entry_not_inherited_command_source(
    selected_mounted_console,  # noqa: F811 - Pytest injects the imported fixture.
    request,
):
    from tldw_chatbook.Chat.console_control_dispatch import (
        ControlIdentity,
        ConsoleLocalControls,
        ReadProjection,
        _command_source,
    )
    from tldw_chatbook.Chat.console_display_phases import ConsoleDisplayPhaseRefused
    import pytest_timeout

    app, runtime, _screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    authored = store.create_session(title="Inherited display context")
    store.set_session_draft(authored.id, "Retain this authored draft")
    assert authored.has_user_work
    app, runtime, screen, pilot = await _navigate_selected_chat(
        selected_mounted_console
    )
    controls = runtime._execution_service._local_controls
    owner = runtime._execution_service._display_owner
    session = owner.store._sessions[owner.store.active_session_id]
    workers, observations, completed = [], [], []

    async def delayed_original_full(command_task, source):
        await asyncio.shield(command_task)
        task = asyncio.current_task()
        assert _command_source.get() is source
        assert task not in controls._settings_command_tasks
        assert command_task not in controls._settings_command_tasks
        before = tuple(controls._effects)
        with pytest.raises(
            ConsoleDisplayPhaseRefused, match="Original FULL registration changed"
        ):
            await owner.read()
        assert tuple(controls._effects) == before
        assert task not in owner.domain.presentation._display_full_entries
        observations.append((task, source))
        result = await screen._sync_native_console_chat_ui()
        assert result is True

    def command_returned(frame):
        if frame.f_locals["self"] is controls and not workers:
            source = _command_source.get()
            assert source is controls._settings_command_tasks[asyncio.current_task()]
            workers.append(
                screen.run_worker(
                    delayed_original_full(asyncio.current_task(), source),
                    exclusive=False,
                    group="console-context-presentation",
                )
            )

    def finished(frame):
        full = frame.f_locals["full"]
        if full.owner is owner:
            completed.append(full)

    identity = ControlIdentity(
        "resume-none",
        "local",
        1,
        "display-inherited",
        session.id,
        session.incarnation_id,
    )
    first_error = None
    with (
        _phase_observer(owner, request),
        _observe(
            ConsoleLocalControls._read_projection.__code__,
            sys.monitoring.events.PY_RETURN,
            command_returned,
        ),
        _observe(
            ConsoleLocalControls._finish_display_effect.__code__,
            sys.monitoring.events.PY_RETURN,
            finished,
        ),
    ):
        assert (
            not pytest_timeout.is_debugging()
            and float(request.config.getini("timeout")) == 300
        )
        receipt = controls.dispatch(ReadProjection(identity))
        try:
            async with asyncio.timeout(15):
                await asyncio.shield(receipt.applied)
                assert len(workers) == 1
                await workers[0].wait()
            assert len(observations) == 1
            task, inherited = observations[0]
            autonomous = [full for full in completed if full.selection.task is task]
            assert len(autonomous) == 1
            full = autonomous[0]
            assert full.selection.command_source is None
            assert full.selection.parent_entry.task is task
            assert full.selection.parent_entry.view is screen
            assert full.effect.completed.result() is True and full.physically_retired
            assert full.logical.callback_returned and full.logical.retired.done()
            assert inherited is not full.effect.owner
            assert app._exception is None
            request.node.user_properties.append(
                (
                    "inherited_context",
                    "actual command return scheduled original Textual worker; expired context refused without entry; exact original FULL admitted independently",
                )
            )
        except BaseException as error:
            first_error = error
            raise
        finally:
            tasks = [receipt.work, *(worker._task for worker in workers)]
            try:
                async with asyncio.timeout(10):
                    await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))
            except BaseException:
                if first_error is None:
                    raise


@pytest.mark.parametrize("selected_mounted_console", ["home"], indirect=True)
async def test_original_character_mount_and_explicit_force_keep_their_full_owner(
    selected_mounted_console,  # noqa: F811 - Pytest injects the imported fixture.
    request,
):
    from tldw_chatbook.Chat.console_control_dispatch import ConsoleLocalControls
    from tldw_chatbook.Chat.console_presentation_port import ConsolePresentationPort
    from tldw_chatbook.UI.Console_Modules.character_context import (
        ConsoleCharacterContextController,
    )
    import pytest_timeout

    app, runtime, _screen, pilot = selected_mounted_console
    owner = runtime._execution_service._display_owner
    demands, completed, scopes = [], [], []

    def demand(frame):
        if frame.f_locals["owner"] is owner:
            demands.append((frame.f_locals["kind"], asyncio.current_task()))

    def finish(frame):
        full = frame.f_locals["full"]
        if full.owner is owner:
            completed.append(full)

    def scope(frame):
        character = frame.f_locals["self"]
        if character is owner.character:
            full = character._selected_display_full()
            scopes.append((frame.f_locals["force"], full))

    with (
        _phase_observer(owner, request),
        _observe(
            ConsolePresentationPort.refresh_display_character.__code__,
            sys.monitoring.events.PY_START,
            demand,
        ),
        _observe(
            ConsoleLocalControls._finish_display_effect.__code__,
            sys.monitoring.events.PY_RETURN,
            finish,
        ),
        _observe(
            ConsoleCharacterContextController.refresh_if_scope_changed.__code__,
            sys.monitoring.events.PY_START,
            scope,
        ),
    ):
        assert (
            not pytest_timeout.is_debugging()
            and float(request.config.getini("timeout")) == 300
        )
        request.node.user_properties.append(
            ("effective_observer_watchdog", "300/debugger=False")
        )
        app, runtime, screen, pilot = await _navigate_selected_chat(
            selected_mounted_console
        )
        assert any(kind == "scope" for kind, _task in demands)
        assert any(full.character_demand == "scope" for full in completed)
        assert screen._character_context is owner.character
        before = len(completed)
        async with asyncio.timeout(10):
            forced = await owner.character.refresh_if_scope_changed(force=True)
        assert forced is True
        candidates = [
            full
            for full in completed[before:]
            if full.character_demand == "forced-scope"
        ]
        assert len(candidates) == 1
        full = candidates[0]
        assert full._initial_character_scope_only
        assert any(force and captured is full for force, captured in scopes)
        _assert_owned_retirement(full)
        assert (
            owner.character._successful_refresh_generation
            == owner.character._generation
        )
        before = len(completed)
        async with asyncio.timeout(10):
            result = await owner.character.refresh()
        assert result is None
        candidates = [
            full for full in completed[before:] if full.character_demand == "refresh"
        ]
        assert len(candidates) == 1
        assert not candidates[0]._initial_character_scope_only
        _assert_owned_retirement(candidates[0])
        assert app._exception is None
        request.node.user_properties.append(
            ("character_demands", json.dumps([kind for kind, _task in demands]))
        )
        request.node.user_properties.append(
            (
                "forced_scope",
                "original public force=True entrance; pair before late refresh; not an activation outcome",
            )
        )


@pytest.mark.parametrize("selected_mounted_console", ["home"], indirect=True)
async def test_actual_appearance_write_drops_only_the_old_display_section(
    selected_mounted_console,  # noqa: F811 - Pytest injects the imported fixture.
    request,
):
    import threading
    from tldw_chatbook.Chat.console_control_dispatch import (
        ControlIdentity,
        ReadProjection,
    )
    import time
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.Chat.console_appearance import ConsoleConversationAppearance
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.UI.Console_Modules.workspace import (
        CONSOLE_APPEARANCE_MAP_CACHE_TTL_SECONDS,
    )
    from Tests.Chat.test_console_control_settings import _entered
    import pytest_timeout

    with _appearance_refusal_observer(request):
        app, runtime, _screen, pilot = selected_mounted_console
        store = runtime.ensure_chat_store()
        session = store.create_session(title="Appearance invalidation")
        store.set_session_draft(session.id, "Keep the real authored input")
        saved = store.persistence.create_conversation(
            conversation_title="Appearance invalidation"
        )
        store.publish_first_persisted_conversation(session.id, saved)
        app, runtime, screen, pilot = await _navigate_selected_chat(
            selected_mounted_console
        )
        owner = runtime._execution_service._display_owner
        service = app.local_chat_conversation_service
        appearance = owner.appearance
        screen._workspace.console_session_switcher_active_entries()
        assert appearance.service is service and saved in appearance.queried_ids
        assert appearance.cache is not None
        initial_map, initial_age = appearance.cache, appearance.cached_at
        entered, release, expired = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        held, invalidations = [], []
        loop_thread = threading.get_ident()

        def native_return(frame):
            full = owner._full
            if (
                threading.get_ident() == loop_thread
                or entered.is_set()
                or frame.f_locals.get("self") is not service
                or saved not in frame.f_locals.get("conversation_ids", ())
                or full is None
                or "appearance" not in full.sections
                or not full.phase_pending
            ):
                return
            held.append(full)
            entered.set()
            if not release.wait(10):
                expired.set()

        def sync_entry(frame):
            if (
                frame.f_locals.get("self") is screen
                and entered.is_set()
                and not release.is_set()
            ):
                invalidations.append(
                    (
                        appearance.cache,
                        appearance.cached_at,
                        appearance.service,
                        set(appearance.queried_ids),
                    )
                )

        first_error, receipt = None, None
        with (
            _phase_observer(owner, request),
            _observe(
                ChatConversationService.get_conversation_appearances.__code__,
                sys.monitoring.events.PY_RETURN,
                native_return,
            ),
            _observe(
                ChatScreen._sync_console_workspace_context.__code__,
                sys.monitoring.events.PY_START,
                sync_entry,
            ),
        ):
            assert (
                not pytest_timeout.is_debugging()
                and float(request.config.getini("timeout")) == 300
            )
            request.node.user_properties.append(
                ("effective_observer_watchdog", "300/debugger=False")
            )
            try:
                # Real original TTL, no cache or timer mutation. The actual consumer
                # is not called a second time outside FULL (that is a live read).
                # Actual ReadProjection drives the original due display entrance.
                await asyncio.sleep(
                    max(
                        0.0,
                        CONSOLE_APPEARANCE_MAP_CACHE_TTL_SECONDS
                        - (time.monotonic() - initial_age),
                    )
                    + 0.01
                )
                controls = runtime._execution_service._local_controls
                receipt = controls.dispatch(
                    ReadProjection(
                        ControlIdentity(
                            "resume-none",
                            "local",
                            1,
                            "appearance-invalidation",
                            session.id,
                            session.incarnation_id,
                            controls.source_revision,
                        )
                    )
                )
                await _entered(entered)
                full = held[0]
                assert full.phase_pending and not full.physically_retired
                assert full.selection.parent_entry.view is screen
                assert appearance.cache is initial_map
                await screen._workspace._write_console_conversation_appearance_off_loop(
                    service.set_conversation_appearance,
                    saved,
                    ConsoleConversationAppearance(icon="\u2b50", color="#aabbcc"),
                    "Appearance invalidation",
                )
                assert any(
                    cache is None and age == 0.0 and memo is None and not ids
                    for cache, age, memo, ids in invalidations
                )
                assert app.local_chat_conversation_service is service
                assert (
                    owner.domain.conversation_scope_service().local_service is service
                )
                request.node.user_properties.append(
                    (
                        "appearance_invalidation",
                        "original successful write cleared memo/cache/time/coverage; true service/database unchanged",
                    )
                )
                release.set()
                async with asyncio.timeout(10):
                    await asyncio.shield(full.selection.task)
                assert not expired.is_set()
                request.node.user_properties.append(
                    ("retained_full_failure", repr(full._failed))
                )
                appearance_future = full.sections["appearance"][2]
                request.node.user_properties.append(
                    (
                        "appearance_phase_failure",
                        repr(appearance_future.exception())
                        if not appearance_future.cancelled()
                        else "cancelled",
                    )
                )
                _assert_owned_retirement(full)
                assert app._exception is None
                assert appearance.cache is not initial_map
                current = await asyncio.to_thread(
                    service.get_conversation_appearance, saved
                )
                assert current.color == "#aabbcc"
                screen._workspace.console_session_switcher_active_entries()
                assert saved in appearance.queried_ids
                assert appearance.cache[saved][1] == "#aabbcc"
            except BaseException as error:
                first_error = error
                raise
            finally:
                release.set()
                tasks = [held[0].selection.task] if held else []
                if receipt is not None:
                    tasks.extend((receipt.work, receipt.applied))
                if tasks:
                    try:
                        async with asyncio.timeout(10):
                            await asyncio.shield(
                                asyncio.gather(*tasks, return_exceptions=True)
                            )
                    except BaseException:
                        if first_error is None:
                            raise


@contextmanager
def _observe_compose_calls(callbacks):
    """Observe a finite set of original entry/return edges in one non-debug slot."""
    monitoring = sys.monitoring
    slot = next(
        i
        for i in range(6)
        if i != monitoring.DEBUGGER_ID and monitoring.get_tool(i) is None
    )
    monitoring.use_tool_id(slot, "task61-compose-edges")

    def dispatch(event, code):
        callback = callbacks.get((code, event))
        if callback is not None:
            callback(sys._getframe(2))

    def entered(code, *_args):
        dispatch(monitoring.events.PY_START, code)

    def returned(code, *_args):
        dispatch(monitoring.events.PY_RETURN, code)

    monitoring.register_callback(slot, monitoring.events.PY_START, entered)
    monitoring.register_callback(slot, monitoring.events.PY_RETURN, returned)
    codes = {code for code, _event in callbacks}
    try:
        for code in codes:
            mask = 0
            for actual, event in callbacks:
                if actual is code:
                    mask |= event
            monitoring.set_local_events(slot, code, mask)
        yield
    finally:
        for code in codes:
            monitoring.set_local_events(slot, code, 0)
        monitoring.register_callback(slot, monitoring.events.PY_START, None)
        monitoring.register_callback(slot, monitoring.events.PY_RETURN, None)
        monitoring.free_tool_id(slot)


@pytest.mark.parametrize("selected_mounted_console", ["home"], indirect=True)
@pytest.mark.parametrize("compose_mode", ["lazy", "eager", "cancel"])
async def test_pristine_compose_busy_read_retires_before_child_mount(
    selected_mounted_console,  # noqa: F811 - Pytest injects the imported fixture.
    request,
    compose_mode,
):
    import threading
    from tldw_chatbook.Chat.console_control_dispatch import (
        ControlIdentity,
        FenceSource,
        ConsoleLocalControls,
    )
    from textual.widget import Widget
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        ConsoleReadinessConfigProjection,
    )
    from tldw_chatbook.Constants import TAB_CHAT
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from Tests.UI.test_console_native_chat_flow import _wait_for_selector
    import pytest_timeout
    from Tests.Chat.test_console_control_settings import _entered
    from Tests.Chat.test_console_selected_mounted_display import (
        _whole_core_hold_file_lock,
        _enter_selected_eager_tasks,
    )
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.config_participants import (
        checked_config_identity,
    )
    from tldw_chatbook.Chat.console_control_settings import (
        ConsoleDefaultConvergenceRequired,
    )
    from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
    from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel

    compose_purpose = "initial-compose"
    compose_method = ChatScreen.__dict__.get("_compose")
    assert (
        compose_method is not None
    ), "Actual initial Compose closure is not integrated"

    app, runtime, absent, pilot = selected_mounted_console
    controls = runtime._execution_service._local_controls
    store = runtime._chat_store
    assert absent is None and runtime.view is None
    assert store.active_session_id is None and not store.sessions()
    assert not controls._parent_settings_captures

    launch = ConsoleLiveWorkLaunch.from_values(
        source="audit", title="Exact initial Compose claim", payload={}, status="staged"
    )
    channel = HandoffChannel.CONSOLE_LIVE_WORK
    revision = app.pending_handoffs.stage(channel, launch)
    held, compose_rows, busy_rows, native_rows, mount_rows, launch_rows = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    read_entered, read_release = threading.Event(), threading.Event()
    checks_per_scope = {}
    retained = {}
    readiness_demands = []
    queued_captures = []
    errors = []
    navigation = None
    fence = None
    unpublished = []
    release_config = None

    def compose_entry(frame):
        screen = frame.f_locals["self"]
        if screen.app_instance is app:
            assert not compose_rows
            compose_rows.append((screen, asyncio.current_task()))
            unpublished.append(screen._task is None)

    def original_busy(frame, values):
        controller, error = frame.f_locals["self"], values[-1]
        if (
            busy_rows
            or not compose_rows
            or controller._screen is not compose_rows[0][0]
            or not isinstance(error, ConsoleDefaultConvergenceRequired)
            or str(error) != "Console defaults config is busy"
        ):
            return
        assert asyncio.current_task() is compose_rows[0][1]
        assert not busy_rows
        busy_rows.append((controller, error, asyncio.current_task()))
        # The genuine nonblocking acquire has refused. Only now unblock the
        # original reader; no cold/cache/revision/source flags are changed.
        release_config.set()

    def original_native_second_identity(frame, values):
        if not busy_rows or native_rows:
            return
        cursor, projection = frame.f_back, None
        while cursor is not None:
            candidate = cursor.f_locals.get("projection")
            if (
                candidate is not None
                and getattr(candidate, "screen", None) is compose_rows[0][0]
                and getattr(candidate.read_current, "__code__", None) is cursor.f_code
            ):
                projection = candidate
                break
            cursor = cursor.f_back
        if projection is None:
            return
        active = frame.f_locals["active"]
        checks_per_scope[active] = checks_per_scope.get(active, 0) + 1
        if checks_per_scope[active] != 2:
            return
        assert threading.current_thread() is not threading.main_thread()
        state = raw._states[active]
        native_rows.append((projection, active, state, tuple(state.leases)))
        read_entered.set()
        assert read_release.wait(10), "Original checked config return was not released"

    def original_launch_entry(frame):
        if compose_rows and (
            frame.f_locals["self"] is compose_rows[0][0]
            and asyncio.current_task() is compose_rows[0][1]
            and (
                phase := getattr(
                    frame.f_locals["self"], "_console_initial_compose", None
                )
            )
            is not None
            and not phase.closed
        ):
            launch_rows.append(frame.f_locals["self"])

    def readiness_run_return(frame):
        projection = frame.f_locals["self"]
        if not compose_rows or projection.screen is not compose_rows[0][0]:
            return
        handoff = projection._handoff_read
        if handoff is None or handoff[2] is None:
            return
        producer = handoff[2]._task
        if any(row["producer"] is producer for row in readiness_demands):
            return
        issuer = asyncio.current_task()
        caller, callers = frame.f_back, []
        while caller is not None and len(callers) < 12:
            callers.append(
                (caller.f_code.co_filename, caller.f_code.co_name, caller.f_lineno)
            )
            caller = caller.f_back
        phase = getattr(projection.screen, "_console_initial_compose", None)
        entry = controls._domain.presentation._display_full_entries.get(issuer)
        parent = next(
            (
                c
                for c in controls._parent_settings_captures
                if c.operation_task is issuer
            ),
            None,
        )
        readiness_demands.append(
            {
                "producer": producer,
                "issuer": issuer,
                "handoff": handoff,
                "callers": callers,
                "phase_closed": phase is None or phase.closed,
                "entry": entry,
                "parent": parent,
                "tokenizer_call": frame.f_back.f_code
                is ConsoleReadinessConfigProjection.tokenizer_ready.__code__,
            }
        )

    def queue_return(frame):
        if frame.f_locals["self"] is controls and compose_rows:
            queued_captures.append(
                {
                    name: frame.f_locals[name]
                    for name in (
                        "command",
                        "parent",
                        "completion",
                        "previous",
                    )
                }
            )

    def assert_exact_compose_tail_retired(capture):
        phase = capture.checked_config
        completion = phase.completion
        assert (
            completion is not None and completion.done() and not completion.cancelled()
        )
        assert completion not in controls._capture_predecessors
        current = controls._session_capture_tails.get(capture.session_id)
        assert current is not completion
        if current is None:
            return
        chain = []
        cursor = current
        while cursor is not None and cursor is not completion:
            issued = next(row for row in queued_captures if row["completion"] is cursor)
            successor = issued["command"]
            assert issued["parent"] and successor is not capture
            assert successor.session is capture.session
            assert successor in controls._parent_settings_captures
            assert successor.operation_task is not capture.operation_task
            assert (
                controls._parent_settings_captures[successor][-1]
                is successor.operation_task
            )
            assert controls._capture_predecessors[cursor] is issued["previous"]
            chain.append(
                {
                    "purpose": successor.purpose,
                    "task": id(successor.operation_task),
                    "completion": id(cursor),
                    "previous": id(issued["previous"]),
                }
            )
            cursor = issued["previous"]
        assert cursor is completion
        request.node.user_properties.append(
            (
                "compose_successor_tail",
                json.dumps(
                    {
                        "compose_completion": id(completion),
                        "compose_done": True,
                        "chain": chain,
                    }
                ),
            )
        )

    def before_children_mount(frame):
        if not compose_rows or frame.f_locals["self"] is not compose_rows[0][0]:
            return
        screen, task = compose_rows[0]
        assert asyncio.current_task() is task
        assert busy_rows and native_rows and retained
        capture, read = retained["capture"], retained["read"]
        projection, active, state, leases = native_rows[0]
        # Must hold at the original framework mount entry, not after FULL/on_mount.
        assert not state.active
        with storage._lock:
            assert leases and all(lease not in storage._live_leases for lease in leases)
        assert (
            read.callback_returned
            and read.retired.done()
            and not read.retired.cancelled()
        )
        assert read._producer is not None and read._producer.done()
        assert read not in runtime._preparation_reads and read not in projection._reads
        due = retained["due"]
        assert due.source is capture and due.producer is read.task
        assert due.handoff[2]._task is read.task and not due.active
        for other in projection._reads:
            demand = next(
                row for row in readiness_demands if row["producer"] is other.task
            )
            assert other is not read and other.task is not read.task
            assert demand["handoff"][2]._task is other.task
            entry, parent = demand["entry"], demand["parent"]
            tokenizer = demand["tokenizer_call"] and [
                row[1] for row in demand["callers"][:3]
            ] == [
                "tokenizer_ready",
                "console_tokenizer_display_ready",
                "_console_settings_context_estimate_for_session",
            ]
            wait = controls._settings_readiness_waits.get(other.task)
            assert wait is None or wait[0] is not capture
            assert other in runtime._preparation_reads
            assert (
                demand["phase_closed"]
                or tokenizer
                or (entry is not None and entry.task is demand["issuer"])
                or (
                    parent is not None
                    and parent is not capture
                    and parent.operation_task is demand["issuer"]
                )
            )
            retained.setdefault("background_reads", []).append(other)
            request.node.user_properties.append(
                (
                    "independent_before_mount_read",
                    json.dumps(
                        {
                            "held_read": id(read),
                            "other_read": id(other),
                            "producer": id(other.task),
                            "issuer": id(demand["issuer"]),
                            "phase_closed_at_demand": demand["phase_closed"],
                            "full_entry": entry is not None,
                            "parent_purpose": parent.purpose
                            if parent is not None
                            else None,
                            "original_tokenizer_demand": tokenizer,
                            "compose_waiter": False,
                            "callers": demand["callers"],
                        }
                    ),
                )
            )
        assert capture not in controls._parent_settings_captures
        assert (
            capture.effect.completed.done()
            and capture.effect.completed.result() is True
        )
        assert capture.effect.physically_retired
        assert task not in controls._due_settings_reads
        assert getattr(screen, "_console_derivation_memo", None) is None
        assert screen._console_initial_compose.closed
        assert screen._console_initial_compose.content_completed
        assert task not in controls._capture_tasks
        assert_exact_compose_tail_retired(capture)
        # The actual message-pump Task is still running: completion is not the witness.
        assert not task.done()
        assert len(launch_rows) == 1
        assert (
            app.pending_handoffs.exact_revision_status(channel, revision) == "settled"
        )
        assert screen._pending_console_launch_context == launch
        mount_rows.append((screen, task, capture, read))

    async def original_navigation_to_composer():
        await app.handle_screen_navigation(NavigateToScreen(TAB_CHAT))
        screen = app._navigation_outgoing_screen()
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        return screen

    events = sys.monitoring.events
    try:
        release_config = await _whole_core_hold_file_lock(config, held)
        if compose_mode == "eager":
            _enter_selected_eager_tasks(request)
        with (
            _observe_compose_calls(
                {
                    (compose_method.__code__, events.PY_START): compose_entry,
                    (
                        ChatScreen._consume_pending_console_launch.__code__,
                        events.PY_START,
                    ): original_launch_entry,
                    (
                        ConsoleReadinessConfigProjection.run.__code__,
                        events.PY_RETURN,
                    ): readiness_run_return,
                    (
                        ConsoleLocalControls._queue_session_capture.__code__,
                        events.PY_RETURN,
                    ): queue_return,
                }
            ),
            _observe(
                ConsoleSessionController._prepare_stale_default_console_settings.__code__,
                events.RAISE,
                original_busy,
                values=True,
            ),
            _observe(
                checked_config_identity.__code__,
                events.PY_RETURN,
                original_native_second_identity,
                values=True,
            ),
            _observe(
                Widget.mount_composed_widgets.__code__,
                events.PY_START,
                before_children_mount,
            ),
        ):
            # Every observer uses a non-debugger monitoring slot, as _observe specifies.
            assert not pytest_timeout.is_debugging()
            assert float(request.config.getini("timeout")) == 300
            request.node.user_properties.append(
                ("effective_compose_watchdog", "300/debugger=False")
            )
            navigation = asyncio.create_task(original_navigation_to_composer())
            await _entered(read_entered)
            assert len(compose_rows) == len(busy_rows) == len(native_rows) == 1
            screen, task = compose_rows[0]
            projection, active, state, leases = native_rows[0]
            assert (
                runtime.view is screen and controls._domain.presentation._view is screen
            )
            assert leases and state.active
            with storage._lock:
                assert all(lease in storage._live_leases for lease in leases)
            captures = [
                capture
                for capture in controls._parent_settings_captures
                if capture.purpose == compose_purpose and capture.operation_task is task
            ]
            assert len(captures) == 1
            capture = captures[0]
            assert capture.session is store._sessions[store.active_session_id]
            assert capture.source_revision == controls.source_revision
            assert capture.sources is controls._domain.sources
            assert not capture.effect.completed.done()
            due = controls._due_settings_reads[task]
            assert due.source is capture and due.projection is projection
            waits = [
                (producer, entry)
                for producer, entry in controls._settings_readiness_waits.items()
                if entry[0] is capture and entry[1] is task and entry[2] is projection
            ]
            assert len(waits) == 1
            producer, wait = waits[0]
            assert wait[3] is projection._handoff_read
            assert wait[3][2]._task is producer and wait[3][0] is wait[4]
            reads = [
                read
                for read in runtime._preparation_reads
                if read.creator is projection
                and read.callback is projection.read_current
                and read.task is producer
            ]
            assert len(reads) == 1
            read = reads[0]
            assert not read.retired.done() and not read.callback_returned
            retained.update(capture=capture, read=read, due=due)
            assert not mount_rows and not navigation.done()
            assert screen._task is task
            if compose_mode == "eager":
                assert unpublished == [True]
            if compose_mode == "cancel":
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                session = capture.session
                identity = ControlIdentity(
                    "resume-none",
                    "local",
                    1,
                    "compose-cancel-fence",
                    session.id,
                    session.incarnation_id,
                )
                fence = controls.dispatch(FenceSource(identity, "configuration"))
                await asyncio.sleep(0)
                assert not task.done() and not capture.work.done()
                assert not capture.effect.completed.done() and not fence.applied.done()
                assert not read.retired.done() and state.active
                read_release.set()
                with pytest.raises(asyncio.CancelledError):
                    async with asyncio.timeout(10):
                        await asyncio.shield(task)
                async with asyncio.timeout(10):
                    await asyncio.shield(fence.applied)
                assert not state.active and read.retired.done()
                with storage._lock:
                    assert all(lease not in storage._live_leases for lease in leases)
                assert (
                    read not in runtime._preparation_reads
                    and read not in projection._reads
                )
                assert read.callback_returned and read._producer.done()
                assert capture.effect.completed.result() is True
                assert capture not in controls._parent_settings_captures
                assert (
                    capture.retired.result() is True
                    and capture.applied.result() is False
                )
                assert (
                    task not in controls._capture_tasks
                    and task not in controls._due_settings_reads
                )
                assert_exact_compose_tail_retired(capture)
                assert screen._console_derivation_memo is None and not mount_rows
                navigation.cancel()
                async with asyncio.timeout(10):
                    await asyncio.shield(
                        asyncio.gather(navigation, return_exceptions=True)
                    )
                # Original installed-screen navigation avoids Textual's stack-prune
                # gather of the deliberately cancelled pre-Mount message pump.
                # Off-stack shutdown uses _close_messages, which retains its
                # original CancelledError handling; never replace the actual Task.
                from tldw_chatbook.Constants import TAB_HOME

                assert app.is_screen_installed(screen)
                async with asyncio.timeout(10):
                    await app.handle_screen_navigation(NavigateToScreen(TAB_HOME))
                assert type(app._navigation_outgoing_screen()).__name__ == "HomeScreen"
                assert screen._task is task and task.cancelled()
                assert capture.effect.completed.result() is True
                request.node.user_properties.append(
                    (
                        "compose_cancellation_cleanup",
                        "original Home navigation; same cancelled off-stack Task",
                    )
                )
                request.node.user_properties.append(
                    (
                        "compose_cancellation_retirement",
                        json.dumps(
                            {
                                "raw_leases": len(leases),
                                "fence_waited": True,
                                "cancelled": task.cancelled(),
                                "mounted": False,
                            }
                        ),
                    )
                )
                return
            read_release.set()
            async with asyncio.timeout(10):
                await asyncio.shield(navigation)
            assert len(mount_rows) == 1 and app._exception is None
            assert not screen.query("#screen-content-error")
            assert screen.query("#console-native-composer")
            assert not app.pending_handoffs.has_pending(channel)
            assert app.pending_handoffs.claim(channel) is None
            request.node.user_properties.append(
                (
                    "compose_due_retirement",
                    json.dumps(
                        {
                            "busy": len(busy_rows),
                            "same_task": True,
                            "raw_leases": len(leases),
                            "retired_before_mount": True,
                            "initial_launch_calls": len(launch_rows),
                            "mode": compose_mode,
                            "unpublished_entry": unpublished[0],
                        }
                    ),
                )
            )
    except BaseException as error:

        def awaiting(task):
            rows, current = [], task.get_coro()
            while current is not None and len(rows) < 30:
                frame = getattr(current, "cr_frame", None) or getattr(
                    current, "gi_frame", None
                )
                if frame is not None:
                    rows.append(
                        [frame.f_code.co_filename, frame.f_code.co_name, frame.f_lineno]
                    )
                current = getattr(current, "cr_await", None) or getattr(
                    current, "gi_yieldfrom", None
                )
            return rows

        screen = compose_rows[0][0] if compose_rows else None
        phase = getattr(screen, "_console_initial_compose", None) if screen else None
        capture = retained.get("capture")
        if phase is None and capture is not None:
            phase = capture.checked_config
        read = retained.get("read")
        request.node.user_properties.append(
            (
                "compose_first_failure_state",
                json.dumps(
                    {
                        "error": repr(error),
                        "compose": len(compose_rows),
                        "busy": len(busy_rows),
                        "native": len(native_rows),
                        "mount": len(mount_rows),
                        "launch": len(launch_rows),
                        "phase_closed": getattr(phase, "closed", None),
                        "phase_content_completed": getattr(
                            phase, "content_completed", None
                        ),
                        "mounted_event": screen._mounted_event.is_set()
                        if screen
                        else None,
                        "read_retired": read.retired.done() if read else None,
                        "read_callback_returned": read.callback_returned
                        if read
                        else None,
                        "effect_done": capture.effect.completed.done()
                        if capture
                        else None,
                        "captures": [
                            (c.purpose, c.started)
                            for c in controls._parent_settings_captures
                        ],
                        "tails": [
                            (key, value.done())
                            for key, value in controls._session_capture_tails.items()
                        ],
                        "navigation": awaiting(navigation) if navigation else [],
                        "readiness_demands": [
                            {
                                "producer": id(row["producer"]),
                                "issuer": id(row["issuer"]),
                                "phase_closed": row["phase_closed"],
                                "parent": row["parent"].purpose
                                if row["parent"]
                                else None,
                                "full_entry": row["entry"] is not None,
                                "callers": row["callers"],
                            }
                            for row in readiness_demands
                        ],
                        "tasks": [
                            {"name": t.get_name(), "await": awaiting(t)}
                            for t in asyncio.all_tasks()
                            if t is not asyncio.current_task() and not t.done()
                        ],
                    }
                ),
            )
        )
        errors.append(error)
    finally:
        read_release.set()
        for background in retained.get("background_reads", ()):
            try:
                async with asyncio.timeout(10):
                    await asyncio.shield(
                        asyncio.gather(background.task, return_exceptions=True)
                    )
                assert background.retired.done()
                assert background not in runtime._preparation_reads
                assert background not in native_rows[0][0]._reads
            except BaseException as error:
                errors.append(error)
        for thread, release, expired in held:
            release.set()
            try:
                await asyncio.to_thread(thread.join, 5)
                assert not thread.is_alive() and not expired.is_set()
            except BaseException as error:
                errors.append(error)
        if navigation is not None:
            try:
                async with asyncio.timeout(10):
                    await asyncio.shield(
                        asyncio.gather(navigation, return_exceptions=True)
                    )
            except BaseException as error:
                errors.append(error)
    if errors:
        raise BaseExceptionGroup("Initial Compose body/owned cleanup failed", errors)


# Append to Tests/Chat/test_console_detached_display.py.


def _home_display_fault_case(case, request):
    import pytest_timeout

    app, runtime, screen, _pilot = case
    controls = runtime._execution_service._local_controls
    owner = runtime._execution_service._display_owner
    assert screen is None and runtime.view is None and owner.context.screen is None
    assert owner._full is None and owner.context.value is None
    assert owner.character.state.scope_fingerprint is None
    assert (
        not pytest_timeout.is_debugging()
        and float(request.config.getini("timeout")) == 300
    )
    request.node.user_properties.append(
        ("effective_observer_watchdog", "300/debugger=False")
    )
    return app, runtime, controls, owner


def _display_fault_identity(controls, session, sequence):
    from tldw_chatbook.Chat.console_control_dispatch import ControlIdentity

    return ControlIdentity(
        "resume-none",
        "local",
        sequence,
        f"display-fault-{sequence}",
        session.id,
        session.incarnation_id,
        controls.source_revision,
    )


async def _join_display_fault_receipts(receipts):
    futures = [
        future
        for receipt in receipts
        for future in (receipt.applied, receipt.work, receipt.retired)
        if future is not None
    ]
    async with asyncio.timeout(10):
        await asyncio.shield(asyncio.gather(*futures, return_exceptions=True))


@contextmanager
def _observe_display_reader_unwind(code, callback):
    monitoring = sys.monitoring
    event = monitoring.events.PY_UNWIND
    slot = next(
        value
        for value in range(6)
        if value != monitoring.DEBUGGER_ID and monitoring.get_tool(value) is None
    )
    monitoring.use_tool_id(slot, "task61-display-reader-unwind")

    def observe(observed_code, _offset, _exception):
        if observed_code is code:
            callback(sys._getframe(1))

    try:
        monitoring.register_callback(slot, event, observe)
        # Python 3.12 supports PY_UNWIND through the global event mask.
        monitoring.set_events(slot, event)
        yield
    finally:
        monitoring.set_events(slot, 0)
        monitoring.register_callback(slot, event, None)
        monitoring.free_tool_id(slot)


@pytest.mark.parametrize("selected_mounted_console", ["home"], indirect=True)
async def test_selected_character_sql_error_allows_actual_context_publication(
    selected_mounted_console,  # noqa: F811 - Pytest injects the imported fixture.
    request,
):
    import threading

    from tldw_chatbook.Chat.console_context_compaction import EffectiveMemoryKind
    from tldw_chatbook.Chat.console_control_dispatch import ReadProjection
    from tldw_chatbook.UI.Console_Modules.character_context import (
        ConsoleCharacterContextController,
    )

    app, runtime, controls, owner = _home_display_fault_case(
        selected_mounted_console, request
    )
    session = runtime.ensure_chat_store().create_session(
        title="Original display SQL fault"
    )
    database = owner.sources.chat_database
    initial_context_age = owner.context.at
    initial_character_age = owner.character._presentation_scope_at
    reader_code = ConsoleCharacterContextController._load_recent_sync.__code__
    loop_thread = threading.current_thread()
    denied, issued, authorizers = [], [], {}

    def authorize(action, table, column, _db_name, _trigger):
        if action == sqlite3.SQLITE_READ and table == "conversations":
            denied.append((table, column))
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def install(frame):
        full = owner._full
        if frame.f_locals.get("self") is not owner.character or full is None:
            return
        assert frame.f_locals["database"] is database
        assert threading.current_thread() is not loop_thread
        phase = full.retirements[-1]
        assert full.phase_pending and phase.native_attempted and not phase.joined
        connection = database.get_connection()
        authorizers[id(frame)] = connection
        issued.append((full, phase, connection))
        connection.set_authorizer(authorize)

    def remove(frame):
        connection = authorizers.pop(id(frame), None)
        if connection is not None:
            connection.set_authorizer(None)

    events = sys.monitoring.events
    with (
        _observe(reader_code, events.PY_START, install),
        _observe(reader_code, events.PY_RETURN, remove),
        _observe_display_reader_unwind(reader_code, remove),
    ):
        receipt = controls.dispatch(
            ReadProjection(_display_fault_identity(controls, session, 1))
        )
        try:
            async with asyncio.timeout(10):
                projection = await asyncio.shield(receipt.applied)

            assert (
                projection.session_id == session.id and receipt.retired.result() is True
            )
            assert denied
            assert len(issued) == 1 and not authorizers
            full, phase, connection = issued[0]
            assert phase.retirement.connection is connection
            assert phase.retirement.confirmed and full.selection.stamp is None
            _assert_owned_retirement(full)
            result = full.sections["character"][2].result()
            assert result.load_failed is True and result.scope_current is True
            assert (
                owner.character.state.error
                == "Could not load local character chats · Retry"
            )
            assert owner.character.state.scope_fingerprint is None
            assert owner.character._presentation_scope_at == initial_character_age
            assert owner.context.at > initial_context_age
            global_policy, can_hold_memory = full.sections["context"][2].result()
            assert can_hold_memory is False and owner.context.value is not None
            overrides, published_policy, effective = owner.context.value
            assert overrides == session.context_policy_overrides
            assert (
                published_policy is global_policy
                and effective.kind is EffectiveMemoryKind.RAW
            )
            assert owner._full is None and not controls.admitted_effects
            assert app._exception is None
            request.node.user_properties.append(
                ("display_sql_denials", json.dumps(denied))
            )
        finally:
            await _join_display_fault_receipts([receipt])


@pytest.mark.parametrize("selected_mounted_console", ["home"], indirect=True)
async def test_selected_uncertain_acquired_close_blocks_publication_fence_and_drain(
    selected_mounted_console,  # noqa: F811 - Pytest injects the imported fixture.
    request,
    monkeypatch,
):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    from Tests.Chat.test_console_control_settings import _entered
    from Tests.Chat.test_console_selected_mounted_display import (
        _EXPECTED_DISPLAY_RETIREMENT_ERROR,
    )
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Chat.console_context_repository import ConsoleContextRepository
    from tldw_chatbook.Chat.console_control_dispatch import (
        Drain,
        FenceSource,
        ReadProjection,
    )
    from tldw_chatbook.Chat.console_display_phases import (
        ConsoleCharacterDisplayPhases,
        ConsoleDisplayPhaseRefused,
    )
    from tldw_chatbook.Chat.console_native_commit import (
        ConsoleNativeRetirementUncertain,
    )

    app, runtime, controls, owner = _home_display_fault_case(
        selected_mounted_console, request
    )
    store = runtime.ensure_chat_store()
    session = store.create_session(title="Uncertain acquired display close")
    conversation_id = store.persistence.create_conversation(
        conversation_title=session.title
    )
    store.publish_first_persisted_conversation(session.id, conversation_id)
    database = owner.sources.chat_database
    repository = controls._controller._context_repository
    assert type(repository) is ConsoleContextRepository and repository.db is database
    assert not store.messages_for_session(session.id)
    before = (
        owner.character.state,
        owner.character._presentation_scope_at,
        owner.context.at,
    )
    entered, release, expired = threading.Event(), threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    original_submit = loop.run_in_executor
    earlier_results, captured, close_attempts, receipts = [], [], [], []
    close_error = sqlite3.OperationalError("injected native close")
    expected_teardown_error = None

    def character_return(frame, args):
        runner = frame.f_locals["self"]
        if runner.full.owner is owner:
            earlier_results.append((runner.full, args[-1]))

    def context_return(frame, args):
        full = owner._full
        if (
            entered.is_set()
            or frame.f_locals.get("self") is not repository
            or frame.f_locals.get("conversation_id") != conversation_id
            or full is None
            or not full.phase_pending
            or "context" not in full.sections
        ):
            return
        assert args[-1] is False
        assert threading.get_ident() in creator_ids
        connection = database._local.conn
        phase = full.retirements[-1]
        assert connection is not None and phase.native_attempted and not phase.joined
        assert database._connection_quiescence.is_registered(connection)
        participant = phase.retirement.participant
        with storage._lock:
            lease = participant.connections[connection]
        close_type, original_close = type(connection), type(connection).close

        def fail_exact_close(candidate):
            if candidate is connection:
                close_attempts.append(candidate)
                raise close_error
            return original_close(candidate)

        injector.setattr(close_type, "close", fail_exact_close)
        captured.append(
            (full, phase, connection, participant, lease, threading.current_thread())
        )
        entered.set()
        if not release.wait(10):
            expired.set()

    creator_ids = set()

    def record_creator():
        creator_ids.add(threading.get_ident())

    def close_on_creator(connection, creator):
        assert threading.current_thread() is creator
        assert database._local.conn is connection
        database.close_connection()

    events = sys.monitoring.events
    with ThreadPoolExecutor(max_workers=1, initializer=record_creator) as executor:
        display_executor = runtime._display_phase_executor.executor()

        def route_exact_display(selected, callback, *args):
            full = owner._full
            if (
                selected is display_executor
                and full is not None
                and asyncio.current_task() is full.selection.task
            ):
                selected = executor
            return original_submit(selected, callback, *args)

        try:
            with (
                monkeypatch.context() as injector,
                _observe(
                    ConsoleCharacterDisplayPhases.read.__code__,
                    events.PY_RETURN,
                    character_return,
                    values=True,
                ),
                _observe(
                    ConsoleContextRepository.has_active_memory_selection.__code__,
                    events.PY_RETURN,
                    context_return,
                    values=True,
                ),
            ):
                injector.setattr(loop, "run_in_executor", route_exact_display)
                try:
                    read = controls.dispatch(
                        ReadProjection(_display_fault_identity(controls, session, 1))
                    )
                    receipts.append(read)
                    await _entered(entered)
                    assert len(captured) == 1 and len(earlier_results) == 1
                    full, phase, connection, participant, lease, _creator = captured[0]
                    earlier_full, earlier = earlier_results[0]
                    assert (
                        earlier_full is full
                        and earlier.scope_current
                        and not earlier.load_failed
                    )
                    assert all(
                        item.joined and item.retirement.confirmed
                        for item in full.retirements[:-1]
                    )
                    assert not full.sections["character"][2].done()
                    assert not full.sections["context"][2].done()
                    assert not read.retired.done()
                    fence = controls.dispatch(
                        FenceSource(
                            _display_fault_identity(controls, session, 2),
                            "configuration",
                        )
                    )
                    receipts.append(fence)
                    await asyncio.sleep(0)
                    assert not fence.applied.done() and not close_attempts

                    release.set()
                    async with asyncio.timeout(10):
                        with pytest.raises(
                            ConsoleDisplayPhaseRefused,
                            match="native retirement is unproven",
                        ):
                            await asyncio.shield(read.applied)
                        with pytest.raises(
                            RuntimeError, match="native effect retirement is unproven"
                        ):
                            await asyncio.shield(fence.applied)
                    assert not expired.is_set()
                    assert read.retired.result() is False
                    assert (
                        owner._full is full
                        and full.phase_issuance_closed
                        and phase.joined
                    )
                    assert (
                        not full.physically_retired and not phase.retirement.confirmed
                    )
                    assert (
                        phase.retirement.connection is connection
                        and phase.retirement.lease is lease
                    )
                    assert close_attempts == [connection]
                    assert database._connection_quiescence.is_registered(connection)
                    with storage._lock:
                        assert (
                            participant.connections[connection] is lease
                            and lease in storage._live_leases
                        )
                    assert (
                        full.effect.completed.result() is False
                        and full.effect in controls._effects
                    )
                    assert full.logical in runtime._preparation_reads
                    assert full.logical.callback_returned is False
                    uncertainty = full.logical.retired.exception()
                    assert type(uncertainty) is ConsoleNativeRetirementUncertain
                    assert not controls._intake_open and not controls._effects_open
                    assert not controls._fence_drained
                    assert owner.character.state is before[0]
                    assert owner.character._presentation_scope_at == before[1]
                    assert owner.context.at == before[2] and owner.context.value is None
                    assert all(
                        future.cancelled() for _, _, future in full.sections.values()
                    )

                    drain = controls.dispatch(
                        Drain(_display_fault_identity(controls, session, 3))
                    )
                    receipts.append(drain)
                    async with asyncio.timeout(10):
                        with pytest.raises(ConsoleNativeRetirementUncertain) as drained:
                            await asyncio.shield(drain.applied)
                    assert drained.value is uncertainty
                    assert app._exception is None
                    expected_teardown_error = uncertainty
                finally:
                    release.set()
                    await _join_display_fault_receipts(receipts)
        finally:
            # Undo only the injector first, then close on the exact creator.
            if captured:
                _full, _phase, connection, participant, lease, creator = captured[0]
                async with asyncio.timeout(10):
                    await asyncio.shield(
                        original_submit(executor, close_on_creator, connection, creator)
                    )
                assert not database._connection_quiescence.is_registered(connection)
                with storage._lock:
                    assert (
                        connection not in participant.connections
                        and lease not in storage._live_leases
                    )
                with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                    sqlite3.Connection.in_transaction.__get__(connection)
                if expected_teardown_error is not None:
                    assert _full.logical.retired.exception() is expected_teardown_error
                    assert _full.effect.completed.result() is False

    assert expected_teardown_error is not None
    request.node.stash[_EXPECTED_DISPLAY_RETIREMENT_ERROR] = expected_teardown_error
    from Tests.UI.app_factory import EXPECTED_RUNTIME_DISPOSAL_ERRORS

    request.node.stash[EXPECTED_RUNTIME_DISPOSAL_ERRORS] = {
        app: (runtime, expected_teardown_error)
    }
    request.node.user_properties.append(
        (
            "retirement_receipt",
            "same retained uncertainty after original creator cleanup",
        )
    )


@pytest.mark.parametrize("selected_mounted_console", ["home"], indirect=True)
@pytest.mark.parametrize("decline", ["custom", "borrowed"])
async def test_selected_display_declines_before_admission(
    selected_mounted_console,  # noqa: F811 - Pytest injects the imported fixture.
    request,
    monkeypatch,
    decline,
):
    from tldw_chatbook.Chat.console_control_dispatch import (
        ConsoleLocalControls,
        ReadProjection,
    )
    from tldw_chatbook.Chat.console_domain_display import ConsoleDisplayUnsupported
    from tldw_chatbook.Chat.console_display_phases import ConsoleDisplayPhysicalOwner

    app, runtime, controls, owner = _home_display_fault_case(
        selected_mounted_console, request
    )
    session = runtime.ensure_chat_store().create_session(
        title="Original unsupported display"
    )
    database = owner.sources.chat_database
    prior_connection = getattr(database._local, "conn", None)
    connection, receipt = None, None
    admissions, native, custom_calls = [], [], []
    before_effects = set(controls._effects)

    def custom(*args, **kwargs):
        custom_calls.append((args, kwargs))
        raise AssertionError("Selected refusal invoked a custom legacy callback")

    def admitted(frame):
        if frame.f_locals["self"] is controls:
            admissions.append(frame.f_locals["full"])

    def native_entry(frame):
        if frame.f_locals["self"].owner is owner:
            native.append(frame.f_locals["self"])

    with (
        monkeypatch.context() as patch,
        _observe(
            ConsoleLocalControls._begin_display_effect.__code__,
            sys.monitoring.events.PY_START,
            admitted,
        ),
        _observe(
            ConsoleDisplayPhysicalOwner.native_phase.__code__,
            sys.monitoring.events.PY_START,
            native_entry,
        ),
    ):
        try:
            if decline == "custom":
                patch.setattr(
                    owner.controller, "context_control_presentation_inputs", custom
                )
                expected = "custom-or-memory-producer"
            else:
                connection = database.get_connection()
                assert database._connection_quiescence.is_registered(connection)
                assert not connection.in_transaction
                connection.execute("BEGIN")
                assert connection.in_transaction
                expected = "original-caller-native-scope"
            receipt = controls.dispatch(
                ReadProjection(_display_fault_identity(controls, session, 1))
            )
            async with asyncio.timeout(10):
                with pytest.raises(ConsoleDisplayUnsupported) as refused:
                    await asyncio.shield(receipt.applied)
            assert refused.value.reason == expected
            assert admissions == native == custom_calls == []
            assert owner._full is None and controls._effects == before_effects
            assert owner.context.value is None
            if connection is not None:
                assert database._local.conn is connection and connection.in_transaction
                assert connection.execute("SELECT 1").fetchone()[0] == 1
            request.node.user_properties.append(("display_preselection", expected))
        finally:
            if receipt is not None:
                await _join_display_fault_receipts([receipt])
            if connection is not None:
                connection.rollback()
                assert not connection.in_transaction
                if prior_connection is None:
                    database.close_connection()
                else:
                    assert database._local.conn is prior_connection


@pytest.mark.parametrize("selected_mounted_console", ["home"], indirect=True)
async def test_selected_display_defining_reader_drift_retires_without_publication(
    selected_mounted_console,  # noqa: F811 - Pytest injects the imported fixture.
    request,
    monkeypatch,
):
    import threading
    from Tests.Chat.test_console_control_settings import _entered
    from tldw_chatbook.Chat.console_control_dispatch import ReadProjection
    from tldw_chatbook.Chat.console_display_phases import ConsoleDisplayPhaseRefused
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Console_Modules.character_context import (
        _stock_character_scope_pair,
    )

    app, runtime, controls, owner = _home_display_fault_case(
        selected_mounted_console, request
    )
    session = runtime.ensure_chat_store().create_session(
        title="Actual defining reader drift"
    )
    entered, release, expired = threading.Event(), threading.Event(), threading.Event()
    captured, custom_calls = [], []
    before_character_age, before_context_age = (
        owner.character._presentation_scope_at,
        owner.context.at,
    )
    receipt = None

    def original_metadata_return(frame):
        full = owner._full
        if (
            entered.is_set()
            or frame.f_locals.get("self") is not owner.sources.chat_database
            or full is None
            or not full.phase_pending
        ):
            return
        assert threading.current_thread() is not threading.main_thread()
        captured.append(full)
        entered.set()
        if not release.wait(10):
            expired.set()

    def changed_reader(database):
        custom_calls.append(database)
        raise AssertionError("Admitted display invoked the replacement reader")

    with (
        monkeypatch.context() as patch,
        _observe(
            CharactersRAGDB.get_local_authority_id.__code__,
            sys.monitoring.events.PY_RETURN,
            original_metadata_return,
        ),
    ):
        try:
            receipt = controls.dispatch(
                ReadProjection(_display_fault_identity(controls, session, 1))
            )
            await _entered(entered)
            assert len(captured) == 1 and _stock_character_scope_pair(owner.character)
            full = captured[0]
            assert full.selection.command_source is not None and full.session is session
            patch.setattr(
                owner.character, "_read_database_scope_metadata", changed_reader
            )
            assert not _stock_character_scope_pair(owner.character)
            release.set()
            async with asyncio.timeout(10):
                with pytest.raises(ConsoleDisplayPhaseRefused):
                    await asyncio.shield(receipt.applied)
            assert not expired.is_set() and custom_calls == []
            assert receipt.retired.result() is True
            _assert_owned_retirement(full, success=False)
            assert all(future.cancelled() for _, _, future in full.sections.values())
            assert owner.character._presentation_scope_at == before_character_age
            assert (
                owner.context.at == before_context_age and owner.context.value is None
            )
            assert owner._full is None and not controls.admitted_effects
            assert app._exception is None
            request.node.user_properties.append(
                (
                    "display_defining_drift",
                    "original metadata SQL return; exact character reader replaced after admission; sentinel never called; actual owned retirement",
                )
            )
        finally:
            release.set()
            if receipt is not None:
                await _join_display_fault_receipts([receipt])


@pytest.mark.parametrize("selected_mounted_console", ["home"], indirect=True)
async def test_retired_session_refusal_does_not_replay_a_superseded_live_view(
    selected_mounted_console,  # noqa: F811 - Pytest injects the imported fixture.
    request,
):
    import threading
    import pytest_timeout
    from tldw_chatbook.Chat.console_domain_display import ConsoleDomainDisplayOwner
    from tldw_chatbook.Chat.console_display_phases import ConsoleDisplaySessionChanged
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Screens.chat_screen import (
        ChatScreen,
        _console_screen_is_torn_down,
    )
    from tldw_chatbook.Constants import TAB_HOME
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from Tests.Chat.test_console_control_settings import _entered

    app, runtime, _, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    original = store.create_session(title="Original overlap view")
    store.set_session_draft(original.id, "Original authored input")
    app, runtime, screen, pilot = await _navigate_selected_chat(
        selected_mounted_console
    )
    owner = runtime._execution_service._display_owner
    successor = store.create_session(title="Successor input", activate=False)
    store.set_session_draft(successor.id, "Successor authored input")
    # A real unsaved settings mutation invalidates the original display key.
    store.set_session_system_prompt(original.id, "Changed before the original FULL")
    entered, release, expired = threading.Event(), threading.Event(), threading.Event()
    full_task = asyncio.current_task()
    held, replacements, classified, replayed = [], [], [], []
    ages = (
        owner.character._presentation_scope_at,
        owner.context.at,
        owner.appearance.cached_at,
    )
    prior_state = owner.character.state
    first_error = None

    def metadata(frame):
        full = owner._full
        if (
            held
            or full is None
            or full.selection.task is not full_task
            or frame.f_locals.get("self") is not owner.sources.chat_database
        ):
            return
        assert threading.current_thread() is not threading.main_thread()
        held.append(full)
        entered.set()
        if not release.wait(10):
            expired.set()

    def classifier(frame, values):
        if frame.f_locals.get("self") is not owner or not held:
            return
        full = held[0]
        if frame.f_locals.get("error") is not full._failed:
            return
        entry = frame.f_locals["entry"]
        if not replacements:
            assert full.physically_retired and full.phase_issuance_closed
            assert (
                full.logical.retired.done() and full.effect.completed.result() is True
            )
            assert all(task.done() for task in full.tasks.values())
            assert entry.view is screen and entry.task is full_task
            assert runtime.view is screen and entry.port._current_visit(entry.stamp)
            assert not _console_screen_is_torn_down(screen)
            # Original construction/accessor makes the supported fresh claim.
            incoming = ChatScreen(app)
            assert incoming._console_runtime() is runtime
            replacements.append(incoming)
            assert runtime.view is incoming and incoming is not screen
            assert entry.port._require_display_owner(owner) is incoming
            assert not entry.port._current_visit(entry.stamp)
            assert not _console_screen_is_torn_down(screen)
            request.node.user_properties.append(
                (
                    "overlap_classifier_entry",
                    "physically retired; old live; fresh actual owner; old visit false",
                )
            )
        if values:
            classified.append(values[-1])

    def replay(frame):
        if (
            replacements
            and frame.f_locals.get("self") is screen
            and frame.f_locals.get("group") == "console-sync"
        ):
            caller = frame.f_back
            if (
                caller is not None
                and caller.f_code is ChatScreen._sync_native_console_chat_ui.__code__
            ):
                replayed.append(asyncio.current_task())

    async def change_session():
        try:
            await _entered(entered)
            assert screen._console_sync_in_progress and held[0].session is original
            assert await screen._sync_native_console_chat_ui() is False
            assert screen._console_sync_requested
            store.switch_session(successor.id)
            assert store.active_session_id == successor.id
        finally:
            release.set()

    changer = None
    try:
        with (
            _observe(
                CharactersRAGDB.get_local_authority_id.__code__,
                sys.monitoring.events.PY_RETURN,
                metadata,
            ),
            _observe(
                ConsoleDomainDisplayOwner.retired_session_change.__code__,
                sys.monitoring.events.PY_START,
                lambda frame: classifier(frame, ()),
            ),
            _observe(
                ConsoleDomainDisplayOwner.retired_session_change.__code__,
                sys.monitoring.events.PY_RETURN,
                classifier,
                values=True,
            ),
            _observe(
                type(screen).run_worker.__code__, sys.monitoring.events.PY_START, replay
            ),
        ):
            assert (
                not pytest_timeout.is_debugging()
                and float(request.config.getini("timeout")) == 300
            )
            request.node.user_properties.append(
                ("overlap_watchdog", "300/debugger=False")
            )
            changer = asyncio.create_task(change_session())
            with pytest.raises(ConsoleDisplaySessionChanged) as refused:
                async with asyncio.timeout(10):
                    await screen._sync_native_console_chat_ui()
            await changer
            assert held and replacements and refused.value is held[0]._failed
            assert refused.value.full is held[0] and not expired.is_set()
            _assert_owned_retirement(held[0], success=False)
            assert all(future.cancelled() for _, _, future in held[0].sections.values())
            assert owner.character.state is prior_state
            assert (
                owner.character._presentation_scope_at,
                owner.context.at,
                owner.appearance.cached_at,
            ) == ages
            assert not replayed and not screen._console_sync_requested
            assert runtime.view is replacements[0] and not _console_screen_is_torn_down(
                screen
            )
            assert app._exception is None
    except BaseException as error:
        first_error = error
        raise
    finally:
        release.set()
        request.node.user_properties.append(
            ("overlap_classifier_returns", repr(classified))
        )
        request.node.user_properties.append(("overlap_old_replay_calls", len(replayed)))
        if changer is not None:
            try:
                async with asyncio.timeout(10):
                    await asyncio.shield(
                        asyncio.gather(changer, return_exceptions=True)
                    )
            except BaseException:
                if first_error is None:
                    raise
        if replacements:
            incoming = replacements[0]
            assert runtime.detach_view(
                incoming, incoming._console_runtime_attachment_generation
            )
        await app.handle_screen_navigation(NavigateToScreen(TAB_HOME))
