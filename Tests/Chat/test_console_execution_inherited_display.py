"""One actual autonomous FULL retains its owner after a received turn closes."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest
import pytest_timeout

from Tests.Chat.test_console_selected_mounted_display import (
    selected_mounted_console as _selected_mounted_console,
)

from Tests.Chat.test_console_control_trace import _observe

selected_mounted_console = _selected_mounted_console
pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("selected_mounted_console", ["home"], indirect=True)
async def test_original_full_after_received_cancel_keeps_its_independent_native_owner(
    selected_mounted_console,
    request,
    record_property,
):
    from Tests.Chat.test_console_domain_boundary import _selection
    from Tests.Chat.test_console_first_send_atomicity import _until
    from tldw_chatbook.Chat.console_control_dispatch import (
        CancelTurn,
        ConsoleLocalControls,
        ControlIdentity,
        ReceiveTurn,
        PermissionAction,
        UpdateInputs,
        _PendingPlan,
        _command_source,
    )
    from tldw_chatbook.Chat.console_display_phases import ConsoleDisplayPhaseRefused
    from tldw_chatbook.Constants import TAB_CHAT
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from Tests.UI.test_console_native_chat_flow import _wait_for_selector

    app, runtime, _screen, _pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    session = store.create_session(title="Actual inherited received source")
    store.set_session_draft(session.id, "Retain original authored input")
    await app.handle_screen_navigation(NavigateToScreen(TAB_CHAT))
    screen = app._navigation_outgoing_screen()
    await _wait_for_selector(screen, _pilot, "#console-native-composer")
    controls = runtime._execution_service._local_controls
    owner = runtime._execution_service._display_owner
    case = SimpleNamespace(
        controller=controls._controller, store=store, session=session
    )
    workers, inherited_sources, completed = [], [], []
    received = None

    def identity(sequence):
        return ControlIdentity(
            controls.profile_epoch,
            controls.service_epoch,
            sequence,
            f"inherited-received-{sequence}",
            session.id,
            session.incarnation_id,
            controls.source_revision,
        )

    async def delayed_full(command_task, source):
        outcomes = await asyncio.gather(command_task, return_exceptions=True)
        assert isinstance(outcomes[0], asyncio.CancelledError)
        assert await asyncio.shield(source.turn_retirement.retired) is True
        assert source.turn_retirement.source_closed
        assert _command_source.get() is source
        assert asyncio.current_task() is not command_task
        before = tuple(controls._effects)
        with pytest.raises(
            ConsoleDisplayPhaseRefused, match="Original FULL registration changed"
        ):
            await owner.read()
        assert tuple(controls._effects) == before
        # Original explicit Refresh guarantees a real native phase. It routes
        # through the actual FULL entry without inventing an inherited grant.
        await owner.character.refresh()

    def plan_created(frame):
        source = _command_source.get()
        if source is None or source.controls is not controls or workers:
            return
        assert (
            source.turn_retirement
            is controls._turn_retirements["inherited-received-turn"]
        )
        assert source.turn_retirement.task is asyncio.current_task()
        inherited_sources.append(source)
        workers.append(
            screen.run_worker(
                delayed_full(asyncio.current_task(), source),
                exclusive=False,
                group="console-context-presentation",
            )
        )

    def full_finished(frame):
        full = frame.f_locals["full"]
        if (
            full.owner is owner
            and inherited_sources
            and _command_source.get() is inherited_sources[0]
        ):
            completed.append(full)

    from tldw_chatbook.Chat.console_control_permissions import SetKillSwitch

    prepared_permission = controls.dispatch(
        PermissionAction(identity(1), SetKillSwitch(value=False))
    )
    await asyncio.shield(prepared_permission.applied)
    assert await asyncio.shield(prepared_permission.retired) is True
    first_error = None
    with (
        _observe(
            _PendingPlan.__init__.__code__,
            sys.monitoring.events.PY_RETURN,
            plan_created,
        ),
        _observe(
            ConsoleLocalControls._finish_display_effect.__code__,
            sys.monitoring.events.PY_RETURN,
            full_finished,
        ),
    ):
        assert float(request.config.getini("timeout")) == 300
        assert not pytest_timeout.is_debugging()
        controls.dispatch(UpdateInputs(identity(2), "Original pre-ACK receive", (1, 1)))
        received = controls.dispatch(
            ReceiveTurn(
                identity(3),
                "inherited-received-turn",
                session.draft_revision,
                _selection(case),
            )
        )
        try:
            assert await _until(lambda: controls.commit_plans or received.work.done())
            if received.work.done():
                received.work.result()
            assert len(workers) == 1 and len(inherited_sources) == 1
            controls.dispatch(CancelTurn(identity(4), "inherited-received-turn"))
            async with asyncio.timeout(15):
                await workers[0].wait()
            source = inherited_sources[0]
            assert source.turn_retirement.retired.result() is True
            assert (
                source.turn_retirement.reads == []
                and not source.turn_retirement.unqualified_read
            )
            full = next(
                item for item in completed if item.character_demand == "refresh"
            )
            assert full.selection.command_source is None
            assert full.selection.parent_entry.view is screen
            assert full.effect.owner is not source
            assert full.effect.completed.result() is True and full.physically_retired
            assert full.retirements and any(
                item.native_attempted for item in full.retirements
            )
            assert all(
                item.joined and item.retirement.confirmed for item in full.retirements
            )
            assert app._exception is None
            record_property(
                "task62_inherited_received_full",
                "closed actual turn; unregistered read refused; independent native FULL retired",
            )
            record_property("task62_independent_native_phases", len(full.retirements))
        except BaseException as error:
            first_error = error
            record_property(
                "task62_first_body_error", f"{type(error).__name__}: {error}"
            )
            raise
        finally:
            if received is not None and not received.work.done():
                received.work.cancel()
            try:
                async with asyncio.timeout(10):
                    await asyncio.gather(
                        *([received.work] if received is not None else []),
                        *(worker._task for worker in workers),
                        return_exceptions=True,
                    )
            except BaseException as error:
                record_property(
                    "task62_cleanup_error", f"{type(error).__name__}: {error}"
                )
                if first_error is None:
                    raise
