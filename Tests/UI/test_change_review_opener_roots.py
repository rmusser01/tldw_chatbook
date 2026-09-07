"""TASK-16801 arc B, Task 9: the opener's `workspace_roots` wiring.

Without `_open_change_review` passing `workspace_roots` through to the
`ChangeReviewScreen` constructor, `current` mode never appears for a
conversation with no recorded turns yet -- the screen's own candidate set
is otherwise just the distinct roots across `change_snapshots` rows, which
is empty until an agent run has written something. This module pins that
wiring on the REAL opener seam (`ChatScreen._open_change_review`), reusing
the same mounted-Console harness and file-backed `AgentRunsDB` /
`ChangeTurnTracker` fixtures built for this exact opener -- no hand-rolled
fake provider shapes here.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.UI.Screens.change_review_screen import ChangeReviewScreen
from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService
from tldw_chatbook.Workspaces.change_turn_tracker import ChangeTurnTracker


class _Workspace:
    def __init__(self, root, service, tracker, db) -> None:
        self.root = root
        self.service = service
        self.tracker = tracker
        self.db = db


@pytest.fixture()
def workspace_fixture(tmp_path) -> _Workspace:
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.py").write_text("line1\nline2\n")
    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    tracker = ChangeTurnTracker(service=service)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    return _Workspace(root, service, tracker, db)


async def _mount_console_session(pilot, console, store, ws: _Workspace):
    """Create a persisted-looking Console session with a real review bridge."""
    await _wait_for_selector(console, pilot, "#console-native-composer")
    session = store.create_session(session_id="conv-opener-roots")
    session.persisted_conversation_id = session.id
    console._ensure_console_chat_controller()
    console._console_agent_bridge = ConsoleAgentBridge(
        agent_runs_db=ws.db,
        store=store,
        provider_gateway=MagicMock(),
        change_tracker=SimpleNamespace(service=ws.service),
    )
    return session


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
    """The mounted turn-context provider passes opted-in roots to Review."""
    ws = workspace_fixture
    app = _build_test_app()
    app.change_review_consent_service = SimpleNamespace(
        admit_turn=lambda _workspace_id: SimpleNamespace(
            ready_roots=(str(ws.root),),
            ready_aliases=(),
            skipped_roots=(),
        )
    )
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        await _mount_console_session(pilot, console, store, ws)

        controller = console._console_chat_controller
        assert controller is not None

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
