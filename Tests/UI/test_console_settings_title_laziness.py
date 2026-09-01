"""The settings path must not query the Workspace DB for a title it discards.

TASK-26839 (TASK-26834 cause 3). Sampled on the main thread in three separate
in-terminal probe runs (30-60ms buckets):

    registry_service.get_workspace
      <- _console_workspace_session_title
      <- _console_initial_session_title_for_workspace
      <- _ensure_active_console_session_settings
      <- _active_console_provider_model_display_uncached

``store.ensure_session`` uses its ``title`` argument ONLY when creating a
session -- with an active session the argument is discarded -- yet the caller
computed it eagerly, which is a synchronous SQLite ``get_workspace`` per
provider/model display rebuild once the active workspace is non-default.

Shape borrowed from ``test_console_keystroke_workspace_reads.py``
(TASK-21118), which gates the same disease on the keystroke path. The
counter here includes its own control: the creation-path test proves the
counter still sees real ``get_workspace`` traffic, so the zero in the lazy
test can never pass against an unwired spy.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

APP_SIZE = (140, 42)

#: Any non-default workspace id: the default ids short-circuit before the
#: registry, so only a non-default id exercises the lookup this gate is for.
NON_DEFAULT_WORKSPACE_ID = "ws-title-laziness-probe"


class _GetWorkspaceCounter:
    """Count ``LocalWorkspaceRegistryService.get_workspace`` calls."""

    def __init__(self) -> None:
        self.calls = 0
        self._original = LocalWorkspaceRegistryService.get_workspace

    def __enter__(self) -> "_GetWorkspaceCounter":
        counter = self
        original = self._original

        def counting(service, workspace_id):
            counter.calls += 1
            return original(service, workspace_id)

        LocalWorkspaceRegistryService.get_workspace = counting
        return self

    def __exit__(self, *_exc) -> None:
        LocalWorkspaceRegistryService.get_workspace = self._original


async def _console_on_non_default_workspace(host, pilot):
    """Return the mounted Console with its store on a non-default workspace."""
    console = await _mounted_console(host, pilot)
    store = console._session._ensure_console_chat_store()
    store.set_workspace_context(
        replace(
            store.workspace_context,
            active_workspace_id=NON_DEFAULT_WORKSPACE_ID,
        )
    )
    return console, store


@pytest.mark.asyncio
async def test_settings_with_an_active_session_never_queries_the_registry():
    """The discarded-title lookup: zero ``get_workspace`` with a session live."""
    _app, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, store = await _console_on_non_default_workspace(host, pilot)
        # Establish the active session first (this call MAY create).
        console._session._ensure_active_console_session_settings()
        assert store.active_session_id is not None

        with _GetWorkspaceCounter() as counter:
            for _ in range(5):
                console._session._ensure_active_console_session_settings()
        assert counter.calls == 0, (
            f"{counter.calls} get_workspace round-trips for 5 settings reads "
            "with an active session -- ensure_session discards the title, so "
            "computing it is a pure main-thread SQLite tax (TASK-26839)"
        )


@pytest.mark.asyncio
async def test_the_title_seam_still_consults_the_registry():
    """The counter control AND the behaviour pin, at the seam that owns it.

    The full creation path cannot be driven with a fabricated workspace id --
    the store coerces an unknown workspace back to the default and takes the
    numbered default title, bypassing the seam. So the pin sits on
    ``_console_initial_session_title_for_workspace`` itself: for a
    non-default id it must consult the registry (proving the counter is
    wired, so test 1's zero means something) and produce the "<name> Chat"
    shape from whatever the registry answers.
    """
    _app, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, _store = await _console_on_non_default_workspace(host, pilot)

        with _GetWorkspaceCounter() as counter:
            title = console._workspace._console_initial_session_title_for_workspace(
                NON_DEFAULT_WORKSPACE_ID
            )
        assert counter.calls == 1, (
            "the title seam no longer consults the registry -- either the "
            "laziness over-reached into creation, or the counter is unwired"
        )
        assert title.endswith(" Chat")
