"""The real pre-ready control-bar route retains first-voice import ownership."""

import asyncio
import sys

import pytest

from Tests.private_profile import private_profile_test

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@private_profile_test
@pytest.mark.parametrize("activate_before_ready", [False, True])
async def test_coalesced_idle_control_bar_preserves_first_voice_import(
    request, monkeypatch, activate_before_ready
):
    from Tests.app_module_patches import set_app_global
    from Tests.UI.test_console_hook_review_send_freeze import _until
    from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
    from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

    name = "tldw_chatbook.Chat.console_voice_input"
    assert name not in sys.modules
    release = asyncio.Event()
    entered = asyncio.Event()
    original = TldwCli._post_mount_setup

    async def held_post_mount(self):
        entered.set()
        await release.wait()
        return await original(self)

    monkeypatch.setattr(TldwCli, "_post_mount_setup", held_post_mount)
    app = _build_test_app(configured_default="chat")
    # Retain the factory's no-profile-DB seam through deferred startup; this
    # test qualifies original voice/paint ownership, not FTS initialization.
    set_app_global(monkeypatch, "get_chachanotes_db_lazy", lambda: None)
    _configure_native_ready_console(app)
    calls = []
    dispatched = ChatScreen._run_coalesced_control_bar_sync.__code__
    action = ConsoleComposerBar.sync_action_state.__code__
    voice = ConsoleComposerBar.sync_dictation_state.__code__
    control = ChatScreen._sync_console_control_bar.__code__
    settled = []
    owned_full_frames = []
    create_worker = ChatScreen.run_worker.__code__
    rail_control = ChatScreen._sync_console_rail_and_controls.__code__
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if frame.f_code in (
            dispatched,
            action,
            voice,
            control,
            rail_control,
            create_worker,
        ):
            ancestor = frame.f_back
            under_dispatch = False
            while ancestor is not None:
                if ancestor.f_code is dispatched or any(
                    ancestor is owned for owned in owned_full_frames
                ):
                    under_dispatch = True
                    break
                ancestor = ancestor.f_back
            if event == "return" and frame.f_code is create_worker and under_dispatch:
                work = getattr(arg, "_work", None)
                if (
                    getattr(work, "cr_code", None)
                    is ChatScreen._sync_native_console_chat_ui.__code__
                ):
                    assert arg.node is frame.f_locals["self"]
                    assert arg.group == "console-sync"
                    owned_full_frames.append(work.cr_frame)
            ready = bool(getattr(app, "_ui_ready", False))
            if event == "call":
                calls.append((frame.f_code, ready, under_dispatch))
            elif (
                event == "return"
                and frame.f_code in (control, rail_control)
                and under_dispatch
            ):
                settled.append((arg, ready))
        if previous is not None:
            previous(frame, event, arg)

    sys.setprofile(observe)
    try:
        async with app.run_test(size=(180, 48)) as pilot:
            try:
                async with asyncio.timeout(5):
                    while not isinstance(app.screen, ChatScreen):
                        await pilot.pause()
                console = app.screen
                await _wait_for_selector(console, pilot, "#console-native-composer")
                assert await _until(entered.is_set, 5)
                assert not app._ui_ready
                # Existing request owner schedules the original callback; no direct
                # projection call, fake readiness flag, timer or sleep substitution.
                console._request_console_control_bar_sync()
                assert await _until(
                    lambda: (
                        any(
                            code is action and not ready and owned
                            for code, ready, owned in calls
                        )
                        and any(
                            result is True and not ready for result, ready in settled
                        )
                    ),
                    5,
                ), {
                    "calls": sorted(
                        {(code.co_name, ready, owned) for code, ready, owned in calls}
                    ),
                    "settled": settled,
                    "owned_full_workers": len(owned_full_frames),
                }
                assert not app._ui_ready
                assert name not in sys.modules, (
                    "idle coalesced projection imported voice before ready"
                )
                if activate_before_ready:
                    # Absence above qualifies first paint. From here, exercise
                    # the real controller/session with only its audio service
                    # and availability boundary replaced by existing fakes.
                    from Tests.UI.test_console_dictation_streaming import (
                        FakeDictationService,
                        _install_streaming_session,
                        _patch_availability,
                    )
                    from tldw_chatbook.Chat import console_voice_input

                    service = FakeDictationService()
                    sessions = _install_streaming_session(monkeypatch, service)
                    _patch_availability(monkeypatch)
                    probe = console_voice_input.probe
                    probes = []

                    def observed_probe(*args, **kwargs):
                        probes.append(bool(app._ui_ready))
                        return probe(*args, **kwargs)

                    monkeypatch.setattr(console_voice_input, "probe", observed_probe)
                    composer = console.query_one(ConsoleComposerBar)
                    composer.load_draft("keep pre-ready draft")
                    assert await pilot.click("#console-dictation")
                    assert await _until(
                        lambda: console._console_dictation_state == "recording", 5
                    )
                    assert not app._ui_ready and service.start_calls == 1 and sessions
                    assert probes and probes[0] is False
                    assert await pilot.click("#console-dictation")
                    assert await _until(
                        lambda: console._console_dictation_state == "idle", 5
                    )
                    assert service.stop_calls == 1 and service.release_calls >= 1
                    assert composer.draft_text() == "keep pre-ready draft"
            finally:
                release.set()
            assert await _until(lambda: app._ui_ready, 5)
            assert await _until(lambda: name in sys.modules, 5)
            await pilot.pause()
    finally:
        release.set()
        sys.setprofile(previous)


@private_profile_test
async def test_custom_idle_composer_without_private_state_keeps_callback(request):
    from types import SimpleNamespace

    from tldw_chatbook.UI.Console_Modules.composer_run_controls import (
        sync_dictation_presentation,
    )

    seen = []
    screen = SimpleNamespace(
        _console_dictation_state="idle", app=SimpleNamespace(_ui_ready=False)
    )
    composer = SimpleNamespace(sync_dictation_state=lambda state: seen.append(state))
    sync_dictation_presentation(screen, composer)
    assert seen == ["idle"]


@private_profile_test
async def test_changed_voice_keyword_default_retains_live_callback(
    request, monkeypatch
):
    from types import SimpleNamespace

    from tldw_chatbook.UI.Console_Modules.composer_run_controls import (
        sync_dictation_presentation,
    )
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

    name = "tldw_chatbook.Chat.console_voice_input"
    assert name not in sys.modules
    composer = ConsoleComposerBar()
    screen = SimpleNamespace(
        _console_dictation_state="idle", app=SimpleNamespace(_ui_ready=False)
    )
    monkeypatch.setitem(
        ConsoleComposerBar.set_voice_status.__kwdefaults__, "message", "custom default"
    )
    sync_dictation_presentation(screen, composer)
    assert name in sys.modules, "custom nested callback was skipped before ready"


@private_profile_test
@pytest.mark.parametrize("change", ["body", "nested-default"])
async def test_preinstalled_voice_producer_remains_live_before_helper_import(
    request, monkeypatch, change
):
    from types import SimpleNamespace

    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

    assert "tldw_chatbook.UI.Console_Modules.composer_run_controls" not in sys.modules
    composer = ConsoleComposerBar()
    screen = SimpleNamespace(
        _console_dictation_state="idle", app=SimpleNamespace(_ui_ready=False)
    )
    if change == "body":

        def custom(self, state):
            self._test_custom_voice_state = state

        monkeypatch.setattr(
            ConsoleComposerBar.sync_dictation_state, "__code__", custom.__code__
        )
    else:
        monkeypatch.setitem(
            ConsoleComposerBar.set_voice_status.__kwdefaults__,
            "message",
            "preinstalled default",
        )
    from tldw_chatbook.UI.Console_Modules.composer_run_controls import (
        sync_dictation_presentation,
    )

    sync_dictation_presentation(screen, composer)
    if change == "body":
        assert getattr(composer, "_test_custom_voice_state", None) == "idle"
    else:
        assert "tldw_chatbook.Chat.console_voice_input" in sys.modules


@private_profile_test
@pytest.mark.parametrize(
    "state,previous,ready",
    [
        ("starting", "idle", False),
        ("recording", "idle", False),
        ("transcribing", "recording", False),
        ("idle", "recording", False),
        ("idle", "idle", True),
    ],
)
async def test_nonidle_transition_and_ready_idle_keep_original_voice_first_use(
    request, state, previous, ready
):
    from types import SimpleNamespace

    from tldw_chatbook.UI.Console_Modules.composer_run_controls import (
        sync_dictation_presentation,
    )
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

    name = "tldw_chatbook.Chat.console_voice_input"
    assert name not in sys.modules
    composer = ConsoleComposerBar()
    composer._dictation_state = previous
    screen = SimpleNamespace(
        _console_dictation_state=state, app=SimpleNamespace(_ui_ready=ready)
    )
    sync_dictation_presentation(screen, composer)
    assert name in sys.modules
    assert composer._dictation_state == state


@private_profile_test
async def test_custom_keyword_value_cannot_claim_stock_voice_defaults(
    request, monkeypatch
):
    from types import SimpleNamespace

    from tldw_chatbook.UI.Console_Modules.composer_run_controls import (
        sync_dictation_presentation,
    )
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

    class CustomMessage(str):
        def __eq__(self, other):
            return True

    name = "tldw_chatbook.Chat.console_voice_input"
    assert name not in sys.modules
    composer = ConsoleComposerBar()
    screen = SimpleNamespace(
        _console_dictation_state="idle", app=SimpleNamespace(_ui_ready=False)
    )
    monkeypatch.setitem(
        ConsoleComposerBar.set_voice_status.__kwdefaults__,
        "message",
        CustomMessage("changed"),
    )
    sync_dictation_presentation(screen, composer)
    assert name in sys.modules, (
        "custom default equality cannot authorize stock deferral"
    )
