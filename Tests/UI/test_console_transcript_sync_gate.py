"""The 0.2s transcript tick skips its message pipeline on an unchanged token."""

from types import SimpleNamespace

import pytest
from textual.css.query import QueryError

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

pytestmark = pytest.mark.bootstrap_profile


def _screen(store):
    class _SyncScreen(ChatScreen):
        def __init__(self) -> None:  # bare shell; gate collaborators stubbed
            pass

        # Bypass the real property, which routes through the console runtime.
        @property
        def _console_chat_store(self):
            return self.__dict__.get("_test_store")

        @_console_chat_store.setter
        def _console_chat_store(self, value):
            self.__dict__["_test_store"] = value

    screen = _SyncScreen()

    def _no_dom(*_args, **_kwargs):
        raise QueryError("bare shell")

    screen.query_one = _no_dom
    fetches = []
    screen._message = SimpleNamespace(
        _native_console_messages=lambda: (fetches.append(1), [])[1]
    )
    screen._change_review_projection = SimpleNamespace(
        project=lambda messages: messages, revision_key=lambda: ("conv", 1)
    )
    screen._console_chat_store = store
    screen._console_transcript_region_or_none = lambda: None
    screen._console_transcript_sync_ticks = 0
    screen._console_transcript_sync_token = None
    return screen, fetches


async def _run(screen):
    try:
        await screen._sync_native_console_transcript()
    except Exception:  # noqa: BLE001 - later collaborators are not stubbed
        pass  # only passage PAST the gate matters here


@pytest.mark.asyncio
async def test_unchanged_token_skips_the_message_pipeline():
    store = SimpleNamespace(
        active_session_id="s1", display_projection_revision=lambda sid: 7
    )
    screen, fetches = _screen(store)

    await _run(screen)
    assert len(fetches) == 1, "first tick must run the pipeline"

    await _run(screen)
    await _run(screen)
    assert len(fetches) == 1, "unchanged token must skip the pipeline"


@pytest.mark.asyncio
async def test_token_bump_reopens_the_pipeline():
    state = {"rev": 7}
    store = SimpleNamespace(
        active_session_id="s1",
        display_projection_revision=lambda sid: state["rev"],
    )
    screen, fetches = _screen(store)

    await _run(screen)
    state["rev"] += 1
    await _run(screen)
    assert len(fetches) == 2, "a store change must re-run the pipeline"


@pytest.mark.asyncio
async def test_forced_pass_runs_every_fifteenth_tick():
    store = SimpleNamespace(
        active_session_id="s1", display_projection_revision=lambda sid: 7
    )
    screen, fetches = _screen(store)

    for _ in range(15):
        await _run(screen)
    assert len(fetches) == 2, "ticks 1 and 15 run; ticks 2-14 skip"


@pytest.mark.asyncio
async def test_session_switch_reopens_the_pipeline():
    state = {"session": "s1"}
    store = SimpleNamespace(
        active_session_id=None,
        display_projection_revision=lambda sid: 7,
    )
    store.active_session_id = state["session"]
    screen, fetches = _screen(store)

    await _run(screen)
    state["session"] = "s2"
    store.active_session_id = "s2"
    await _run(screen)
    assert len(fetches) == 2, "switching sessions must re-run the pipeline"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["trace", "continuation"])
async def test_recovery_changes_are_visible_while_message_pipeline_is_gated(kind):
    from textual.app import ComposeResult
    from textual.widgets import Static

    from Tests.UI.consolidated_css import ConsolidatedCSSApp
    from Tests.UI.test_console_provider_continuation_recovery import _message
    from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
        ProviderContinuationRecoveryCallout,
        ProviderContinuationTranscriptRegion,
        TraceCallRecoveryCallout,
        TraceCallRecoveryState,
    )

    current = {"trace": None, "continuation": None}

    class RecoveryApp(ConsolidatedCSSApp):
        def compose(self) -> ComposeResult:
            yield ProviderContinuationTranscriptRegion(
                session_surface_builder=lambda: Static("transcript"),
                recovery_message_builder=lambda: current["continuation"],
                on_recovery_action=lambda *_: False,
                trace_recovery_state_builder=lambda: current["trace"],
            )

    class PipelineReached(Exception):
        pass

    def stop_pipeline(_messages):
        raise PipelineReached

    app = RecoveryApp()
    async with app.run_test(size=(60, 24)) as pilot:
        await pilot.pause()
        region = app.query_one(ProviderContinuationTranscriptRegion)
        card = region.query_one(
            TraceCallRecoveryCallout
            if kind == "trace"
            else ProviderContinuationRecoveryCallout
        )
        assert region.is_mounted and not card.display
        store = SimpleNamespace(
            active_session_id="s1", display_projection_revision=lambda _sid: 7
        )
        screen, fetches = _screen(store)
        screen._console_transcript_region_or_none = lambda: region
        screen._change_review_projection.project = stop_pipeline
        with pytest.raises(PipelineReached):
            await screen._sync_native_console_transcript()
        assert fetches == [1]

        current[kind] = (
            TraceCallRecoveryState("held-send") if kind == "trace" else _message()
        )
        await screen._sync_native_console_transcript()
        assert card.display
        current[kind] = None
        await screen._sync_native_console_transcript()
        assert not card.display
        assert fetches == [1], "recovery must not reopen the message pipeline"
