"""TASK-16801 arc B, Task 9: the opener's `workspace_roots` wiring.

Without `_open_change_review` passing `workspace_roots` through to the
`ChangeReviewScreen` constructor, `current` mode never appears for a
conversation with no recorded turns yet -- the screen's own candidate set
is otherwise just the distinct roots across `change_snapshots` rows, which
is empty until an agent run has written something. This module pins that
wiring on the REAL opener seam (`ChatScreen._open_change_review`), reusing
the same mounted-Console harness and file-backed `AgentRunsDB` /
`ChangeTurnTracker` fixtures `test_console_changed_files_wiring.py` already
built for this exact opener -- no hand-rolled fake provider shapes here.
"""

from __future__ import annotations

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.test_console_changed_files_wiring import (
    _mount_console_session,
    workspace_fixture,  # noqa: F401 -- imported for pytest fixture discovery
)
from tldw_chatbook.config import save_setting_to_cli_config
from tldw_chatbook.UI.Screens.change_review_screen import ChangeReviewScreen


async def _wait_for_change_review_screen(
    host, pilot, *, attempts: int = 50
) -> ChangeReviewScreen:
    for _ in range(attempts):
        top = host.screen_stack[-1]
        if isinstance(top, ChangeReviewScreen):
            return top
        await pilot.pause(0.02)
    raise AssertionError("the opener never pushed the Review screen")


@pytest.mark.asyncio
async def test_opener_passes_the_controllers_workspace_roots_to_the_screen(
    workspace_fixture,
):
    """The `[console] workspace_root` the real turn-context accessor
    resolves (`resolve_turn_execution_context(session_id).workspace_roots`,
    the same field `console_chat_controller.py` turns into `change_roots`
    for the tracker) must ride the screen constructor unchanged.
    """
    ws = workspace_fixture
    # `get_cli_setting` (what `resolve_turn_execution_context`'s fallback
    # reads) is backed by `load_cli_config_and_ensure_existence`'s OWN
    # cache -- a different cache than `load_settings()`, which is what
    # `_build_test_app(config_overrides=...)` merges into. Writing through
    # `save_setting_to_cli_config` is the real, disk-backed seam (the one
    # the Settings screen itself uses) that both caches end up agreeing on.
    save_setting_to_cli_config("console", "workspace_root", str(ws.root))

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        await _mount_console_session(pilot, console, store, ws)

        controller = console._console_chat_controller
        assert controller is not None
        # A mounted ChatScreen wires `_turn_context_provider` to the
        # session's OWN accessor (`_build_console_turn_execution_context`,
        # `UI/Console_Modules/session.py`), which derives `workspace_roots`
        # from the folder-binding registry rather than the flat
        # `[console] workspace_root` key. Clearing it exercises
        # `resolve_turn_execution_context`'s own fallback implementation
        # (`console_chat_controller.py`) -- the one the brief's accessor
        # description matches -- without touching the pass-through code
        # under test, which reads `.workspace_roots` off whatever
        # `resolve_turn_execution_context` returns either way.
        controller._turn_context_provider = None

        console._open_change_review()
        review = await _wait_for_change_review_screen(host, pilot)

        assert review._workspace_roots == (str(ws.root),)


@pytest.mark.asyncio
async def test_opener_degrades_to_no_roots_when_the_controller_raises(
    workspace_fixture,
):
    """A controller lookup failure must never break the opener -- it opens
    the screen with `workspace_roots=None` (screen-side default: an empty
    tuple), matching `_console_change_review_provider`'s own degrade
    posture just above it in `chat_screen.py`.
    """
    ws = workspace_fixture

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        await _mount_console_session(pilot, console, store, ws)

        controller = console._console_chat_controller
        assert controller is not None

        def _raise(*_args, **_kwargs):
            raise RuntimeError("turn-context resolution exploded")

        controller.resolve_turn_execution_context = _raise

        console._open_change_review()
        review = await _wait_for_change_review_screen(host, pilot)

        assert review._workspace_roots == ()
