"""App-instance audio admission, external confirmation and close."""

import asyncio

import pytest
from textual.app import App

from Tests.private_profile import private_profile_test
from tldw_chatbook.TTS.audio_output_ownership import OutputClosed
from tldw_chatbook.app_service_wiring import ServiceWiringMixin


class AudioApp(App):
    audio_output = ServiceWiringMixin.audio_output


class Prepared:
    async def retire(self):
        return True


@pytest.mark.asyncio
async def test_lazy_owner_is_private_to_each_app_and_closed_at_exit():
    from tldw_chatbook.app_shutdown import begin_audio_shutdown

    first, second = AudioApp(), AudioApp()
    assert "_audio_output" not in vars(first)
    begin_audio_shutdown(first)
    assert "_audio_output" not in vars(first)
    with pytest.raises(OutputClosed):
        _ = first.audio_output
    assert second.audio_output is second.audio_output
    other = AudioApp()
    assert second.audio_output is not other.audio_output


@pytest.mark.asyncio
@pytest.mark.parametrize("confirm", [False, True])
async def test_external_confirmation_requires_a_positive_button(confirm):
    from tldw_chatbook.UI.audio_output_admission import acquire_audio_output

    app = AudioApp()
    async with app.run_test() as pilot:
        run = Prepared()
        host = app.audio_output
        old = await host.acquire("external", run.retire, external=True)
        old.record_handoff()
        old.release()
        ticket = host.begin_intent()
        task = asyncio.create_task(
            acquire_audio_output(app, ticket, "preview", run, current=ticket.is_current)
        )
        await pilot.pause()
        assert app.screen.title == "External playback"
        assert app.focused.id == "cancel-button"
        if confirm:
            await pilot.click("#confirm-button")
        else:
            await pilot.press("escape")
        lease = await asyncio.wait_for(task, 2)
        assert bool(lease and lease.authorizes(run.retire)) == confirm
        assert (host.handoff_epoch is None) == confirm


@pytest.mark.asyncio
async def test_only_the_current_top_audio_prompt_owns_its_originating_ticket():
    from textual.screen import ModalScreen
    from tldw_chatbook.UI.audio_output_admission import (
        acquire_audio_output,
        is_audio_confirmation_for,
    )

    app = AudioApp()
    async with app.run_test() as pilot:
        run = Prepared()
        host = app.audio_output
        external = await host.acquire("external", run.retire, external=True)
        external.record_handoff()
        external.release()
        old = host.begin_intent()
        ticket = host.begin_intent()
        task = asyncio.create_task(
            acquire_audio_output(app, ticket, "preview", run, current=lambda: True)
        )
        await pilot.pause()
        assert is_audio_confirmation_for(app, ticket)
        assert not is_audio_confirmation_for(app, old)
        unrelated = ModalScreen()
        await app.push_screen(unrelated)
        assert not is_audio_confirmation_for(app, ticket)
        ticket.cancel()
        await pilot.pause()
        assert app.screen is unrelated
        unrelated.dismiss()
        await pilot.pause()
        assert not is_audio_confirmation_for(app, ticket)
        assert await asyncio.wait_for(task, 2) is None
        assert host.handoff_epoch is not None


@pytest.mark.asyncio
async def test_quick_exit_timeout_retains_job_and_direct_unmount_fences_before_await():
    from tldw_chatbook.app_lifecycle import LifecycleMixin

    app = AudioApp()
    other = AudioApp()
    host = app.audio_output
    held, started = asyncio.Event(), asyncio.Event()

    async def retire():
        started.set()
        await held.wait()
        return True

    lease = await host.acquire("held start", retire)

    async def first_unrelated_cleanup():
        assert host.closing and lease.revoked
        raise asyncio.CancelledError()

    app._shutdown_actor_pack_recovery = first_unrelated_cleanup
    with pytest.raises(asyncio.CancelledError):
        await LifecycleMixin.on_unmount(app)
    await started.wait()
    await LifecycleMixin._cleanup_audio_for_quit(app)
    assert host.owner == "held start" and not other.audio_output.closing
    held.set()
    assert await host.begin_shutdown()
    assert host.owner is None


@pytest.mark.asyncio
async def test_quick_exit_reports_negative_native_receipt():
    from tldw_chatbook.app_shutdown import observe_audio_shutdown

    app = AudioApp()
    host = app.audio_output
    await host.acquire("unconfirmed", lambda: asyncio.sleep(0, result=False))
    assert not await observe_audio_shutdown(app, 1)
    assert host.owner == "unconfirmed" and host.blocked


@pytest.mark.asyncio
async def test_retired_external_file_releases_once_on_deadline_or_shutdown(monkeypatch):
    from tldw_chatbook.TTS import audio_player
    from tldw_chatbook.UI.audio_output_admission import retain_retired_handoff_file
    from tldw_chatbook.app_lifecycle import LifecycleMixin

    app, other = AudioApp(), AudioApp()
    delegated = await app.audio_output.acquire(
        "external", Prepared().retire, external=True
    )
    delegated.record_handoff()
    delegated.release()
    epoch = app.audio_output.handoff_epoch
    released = []
    expired = asyncio.Event()
    monkeypatch.setattr(audio_player, "HAND_OFF_RETENTION_SECONDS", 0)
    retain_retired_handoff_file(other, expired.set)
    await asyncio.wait_for(expired.wait(), 1)
    monkeypatch.setattr(audio_player, "HAND_OFF_RETENTION_SECONDS", 60)
    retain_retired_handoff_file(app, lambda: released.append("first"))
    await LifecycleMixin._cleanup_audio_for_quit(app)
    retain_retired_handoff_file(app, lambda: released.append("late"))
    await LifecycleMixin._cleanup_audio_for_quit(app)
    assert released == ["first", "late"] and app.audio_output.handoff_epoch is epoch


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["message", "hands_free", "qualified", "realtime"])
async def test_console_suspend_recognizes_only_its_exact_audio_prompt(kind):
    from types import SimpleNamespace
    from textual.screen import Screen
    from textual.widgets import Button
    from tldw_chatbook.UI.audio_output_admission import acquire_audio_output
    from tldw_chatbook.UI.Console_Modules.wiring import console_audio_confirmation_owner

    app = AudioApp()
    observations = []

    class Origin(Screen):
        def on_screen_suspend(self):
            observations.append(console_audio_confirmation_owner(self))

    async with app.run_test() as pilot:
        origin = Origin()
        origin.app_instance = app
        await app.push_screen(origin)
        run = Prepared()
        host = app.audio_output
        old = await host.acquire("external", run.retire, external=True)
        old.record_handoff()
        old.release()
        ticket = host.begin_intent()
        origin._message = SimpleNamespace(
            audio_intent=ticket if kind == "message" else None
        )
        origin._hands_free = SimpleNamespace(
            _qualified_audio_intent=ticket if kind == "qualified" else None,
            _console_hands_free=SimpleNamespace(
                audio_intent=ticket if kind == "hands_free" else None
            ),
        )
        origin._realtime = SimpleNamespace(
            session=SimpleNamespace(audio_intent=ticket if kind == "realtime" else None)
        )
        task = asyncio.create_task(
            acquire_audio_output(app, ticket, "speech", run, current=lambda: True)
        )
        await pilot.pause()
        assert observations[-1] == ("hands_free" if kind == "qualified" else kind)
        app.screen.query_one("#confirm-button", Button).focus()
        await pilot.press("enter")
        granted = await asyncio.wait_for(task, 1)
        assert granted.authorizes(run.retire)
        await app.push_screen(Screen())
        await pilot.pause()
        assert observations[-1] is None


@pytest.mark.asyncio
@private_profile_test
async def test_mounted_chat_preserves_request_lifetime_only_for_its_own_prompt(request):
    from textual.screen import ModalScreen
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_console_fleet_wake_hidden_screen import _mount_chat
    from tldw_chatbook.UI.audio_output_admission import acquire_audio_output

    app = _build_test_app()
    async with app.run_test(size=(140, 44)) as pilot:
        console = await _mount_chat(app, pilot)
        host = app.audio_output
        run = Prepared()
        external = await host.acquire("external", run.retire, external=True)
        external.record_handoff()
        external.release()
        ticket = console._message.audio_intent = host.begin_intent()
        generation = console._message._console_speech_lifetime_generation
        current = (
            lambda: console._message._console_speech_lifetime_generation == generation
        )
        playing = asyncio.create_task(
            acquire_audio_output(app, ticket, "Console speech", run, current=current)
        )
        await pilot.pause()
        assert current() and app.screen.title == "External playback"
        from textual.widgets import Button

        button = app.screen.query_one("#confirm-button", Button)
        button.focus()
        await pilot.press("enter")
        assert (await asyncio.wait_for(playing, 3)).authorizes(run.retire)
        await app.push_screen(ModalScreen())
        await pilot.pause()
        assert not current()


@pytest.mark.asyncio
@private_profile_test
async def test_unrelated_cover_retires_mounted_handsfree_stream(request):
    from unittest.mock import Mock
    from textual.screen import ModalScreen
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_console_fleet_wake_hidden_screen import _mount_chat
    from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSEventHandler
    from tldw_chatbook.TTS.audio_output_runs import PcmOutputRun
    from tldw_chatbook.UI.Console_Modules.hands_free import ConsoleHandsFreeSession

    app = _build_test_app()
    async with app.run_test(size=(140, 44)) as pilot:
        console = await _mount_chat(app, pilot)
        handler = app._tts_handler or TTSEventHandler()
        handler.app = app
        app._tts_handler = handler
        ticket = app.audio_output.begin_intent()
        stream = Mock()
        run = PcmOutputRun(
            24000, on_event=lambda _: None, stream_factory=lambda **_: stream
        )
        output = handler._prepare_output("mounted reply", None, ticket=ticket)
        output.run = run
        lease = await app.audio_output.acquire(
            "mounted reply", run.retire, intent=ticket
        )
        assert await run.start(lease)
        console._hands_free._console_hands_free = ConsoleHandsFreeSession(
            controller=Mock(state="speaking"), sequencer=Mock(), audio_intent=ticket
        )
        await app.push_screen(ModalScreen())
        await pilot.pause()
        async with asyncio.timeout(3):
            while not run.retirement_confirmed:
                await asyncio.sleep(0.01)
        assert lease.revoked and stream.close.called and app.audio_output.owner is None
        assert console._hands_free._console_hands_free is None

@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["durable", "unmount"])
@private_profile_test
async def test_listener_refusal_preserves_dependencies_and_drains_independent_owners(
    request, monkeypatch, scope
):
    from unittest.mock import AsyncMock, Mock
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook import app_shutdown
    from tldw_chatbook.Utils import egress

    app = TldwCli()
    problem = RuntimeError("player_retirement_unconfirmed")
    monkeypatch.setattr(
        app.article_listening_service, "close", AsyncMock(side_effect=problem)
    )
    if scope == "durable":
        unrelated = AsyncMock()
        monkeypatch.setattr(app, "_shutdown_workflow_session", unrelated)
        dependent = Mock()
        monkeypatch.setattr(egress, "close_pooled_default_sessions", dependent)
        operation = app._shutdown_app_owned_lifecycles()
    else:
        unrelated = AsyncMock()
        monkeypatch.setattr(
            app, "_close_server_context_provider_cached_client", unrelated
        )
        dependent = AsyncMock()
        monkeypatch.setattr(app, "_close_owned_tts_resources", dependent)
        operation = app_shutdown._drain_app_unmount_services(app, None)
    with pytest.raises(RuntimeError) as failure:
        await operation
    assert failure.value is problem
    unrelated.assert_awaited()
    dependent.assert_not_called()
