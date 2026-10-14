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
