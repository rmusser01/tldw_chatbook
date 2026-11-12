"""TASK-33620.15.1.5: one helper owns a session's consent-admitted Change Review roots.

Three readers ask for a session's Change Review roots: the turn
configuration builder, the hook authority check (which compares its answer
with the frozen configuration) and the Environment panel / Change Review
accessor. Each used to repeat the steps -- the session's workspace id, then
the consent admission -- and only two of them coerced the roots to ``str``.
These drive the real runtime-owned controller of a mounted Console and pin
that all three answer from the one read and agree, as strings.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

import tldw_chatbook.Chat.console_turn_context as turn_context
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)

# Drives the real ChatScreen, controller, store and app: keep the
# collection-time profile (the per-test redirect fails config admission).
pytestmark = pytest.mark.bootstrap_profile


@asynccontextmanager
async def _console(consent):
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.change_review_consent_service = consent
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        await pilot.pause(0.2)
        controller = screen._ensure_console_chat_controller()
        assert controller._turn_context_provider is None, (
            "harness precondition: the runtime-owned controller"
        )
        session_id = controller.store.active_session_id
        assert session_id, "harness precondition: an active session"
        yield screen, controller, session_id


def _three_readers(screen, controller, session_id) -> dict[str, tuple]:
    return {
        "snapshot builder": controller.resolve_turn_configuration_snapshot(
            session_id
        ).workspace_roots,
        "hook authority": controller._hook_authority_values(session_id)[
            "workspace_roots"
        ],
        "Environment / Change Review accessor": (
            screen._review_selection._console_change_review_workspace_roots()
        ),
    }


@pytest.mark.asyncio
async def test_the_three_readers_return_the_same_string_roots(tmp_path):
    """AC#2: the same session gets the same ``str`` tuple from all three.

    Base (origin/dev 3be493413a): hook authority returned the admission's
    roots uncoerced, so a ``Path`` root differed from the snapshot's ``str``
    -- and the hook check compares exactly those two values.
    """
    root = PurePosixPath(tmp_path)
    consent = SimpleNamespace(
        admit_turn=lambda _workspace_id: SimpleNamespace(
            ready_roots=(root, root / "second"),
            ready_aliases=("alias-one", "alias-two"),
            skipped_roots=(),
        )
    )
    async with _console(consent) as (screen, controller, session_id):
        readers = _three_readers(screen, controller, session_id)

    expected = (str(root), str(root / "second"))
    assert readers == dict.fromkeys(readers, expected)
    assert all(type(r) is str for roots in readers.values() for r in roots), readers


@pytest.mark.asyncio
async def test_the_three_readers_answer_from_one_helper(monkeypatch, tmp_path):
    """AC#1: replace the one read, and every reader reports the replacement.

    Base: the builder and hook authority each called their own imported
    reference to the admission read, so replacing it in
    ``console_turn_context`` reached only the narrow accessor.
    """
    consent = SimpleNamespace(
        admit_turn=lambda _workspace_id: SimpleNamespace(
            ready_roots=(str(tmp_path),), ready_aliases=("a",), skipped_roots=()
        )
    )
    async with _console(consent) as (screen, controller, session_id):
        own_workspace = controller.store.session_workspace_id(session_id)
        asked: list = []

        def one_read(app, workspace_id):
            asked.append(workspace_id)
            return (f"{tmp_path}/the-one-read",), ("alias",), ()

        monkeypatch.setattr(turn_context, "capture_change_review_admission", one_read)
        readers = _three_readers(screen, controller, session_id)

    assert readers == dict.fromkeys(readers, (f"{tmp_path}/the-one-read",))
    assert asked == [own_workspace] * 3
