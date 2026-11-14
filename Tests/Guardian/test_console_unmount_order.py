"""Guardian's visit summary must not run ahead of Console's ownership release.

``ChatScreen.on_unmount`` releases its settings claim and detaches from the runtime
synchronously, then flushes sidebar state as its first awaited step (task-15470). The
Guardian summary awaits a worker thread, so placed before the detach it reopened the window
in which an unmounting screen is still attached; it also broke
``Tests/UI/test_console_runtime_ownership.py::test_unmount_releases_settings_claim_before_blocked_sidebar_flush``
on the rebased branch (review of #2921).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


class _StopHere(Exception):
    """Ends on_unmount right after the step under test."""


@pytest.mark.asyncio
async def test_guardian_visit_summary_runs_after_release_detach_and_sidebar_flush():
    order: list[str] = []

    async def flush() -> None:
        order.append("sidebar-flush")

    async def finalize() -> None:
        order.append("guardian")
        raise _StopHere

    screen = SimpleNamespace(
        _hooks=SimpleNamespace(cancel_pending=lambda: None),
        _release_claimed_conversation_settings_return=lambda: order.append("claim"),
        _console_runtime=lambda: SimpleNamespace(
            detach_view=lambda *_: order.append("detach")
        ),
        _flush_sidebar_state_now=flush,
        _finalize_guardian_visit=finalize,
    )

    with pytest.raises(_StopHere):
        await ChatScreen.on_unmount(screen)

    assert order == ["claim", "detach", "sidebar-flush", "guardian"]


# --- Staging and consumption of the visit summary (review round 3, item 11) -----
#
# The summary is produced at unmount and shown on the NEXT Console mount; the
# app object carries it between the two screens. Nothing covered either half.

_SUMMARY = {
    "session_id": "session-a",
    "visit_id": "visit-a",
    "per_topic_counts": {"gambling": 2, "alcohol": 1},
    "escalated_rules": ["late-night spending"],
    "trend_notices": [{"label": "gambling rising"}],
}


@pytest.fixture(autouse=True)
def enabled_guardian(monkeypatch):
    from tldw_chatbook.Guardian import settings
    monkeypatch.setattr(settings, "guardian_setting", lambda key, default=None: True if key == "enabled" else default)


def _screen(app, checker=None, session_id="session-a") -> SimpleNamespace:
    return SimpleNamespace(app_instance=app, _guardian_checker=checker,
        _ensure_console_chat_store=lambda: SimpleNamespace(active_session_id=session_id))


@pytest.mark.asyncio
async def test_unmount_stages_the_visit_summary_on_the_app():
    app = SimpleNamespace(_guardian_visit_summary_notice=None)
    async def finalize():
        return dict(_SUMMARY)
    checker = SimpleNamespace(finalize_visit_async=finalize)

    await ChatScreen._finalize_guardian_visit(_screen(app, checker))

    assert app._guardian_visit_summary_notice == {"session-a": _SUMMARY}


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["empty", "raises", "no-checker"])
async def test_unmount_stages_nothing_without_a_summary(outcome):
    """An empty visit, a failed summary and a screen that never built a
    checker all leave the slot as it was -- and never raise into unmount."""
    earlier = {"per_topic_counts": {"kept from an earlier visit": 1}}
    app = SimpleNamespace(_guardian_visit_summary_notice=earlier)

    async def finalize():
        if outcome == "raises":
            raise RuntimeError("guardian store unavailable")
        return None

    checker = (None if outcome == "no-checker"
               else SimpleNamespace(finalize_visit_async=finalize))

    await ChatScreen._finalize_guardian_visit(_screen(app, checker))

    assert app._guardian_visit_summary_notice is earlier


def test_next_mount_shows_the_staged_summary_once_and_clears_it():
    shown: list[tuple[str, str]] = []
    app = SimpleNamespace(
        _guardian_visit_summary_notice={"session-a": dict(_SUMMARY)},
        notify=lambda text, severity: shown.append((text, severity)),
    )
    screen = _screen(app)

    ChatScreen._notify_guardian_visit_summary_if_any(screen)

    assert app._guardian_visit_summary_notice is None, "read AND cleared"
    assert len(shown) == 1
    text, severity = shown[0]
    assert severity == "information"
    assert "alcohol x1, gambling x2" in text
    assert "escalated: late-night spending" in text
    assert "trends: gambling rising" in text

    ChatScreen._notify_guardian_visit_summary_if_any(screen)  # the mount after

    assert len(shown) == 1, "a summary is shown on one mount only"


def test_mount_with_nothing_staged_is_silent():
    shown: list[str] = []
    app = SimpleNamespace(
        _guardian_visit_summary_notice=None,
        notify=lambda text, severity: shown.append(text),
    )

    ChatScreen._notify_guardian_visit_summary_if_any(_screen(app))

    assert shown == []


def test_mount_does_not_consume_another_chats_summary():
    shown = []
    pending = {"session-a": dict(_SUMMARY)}
    app = SimpleNamespace(_guardian_visit_summary_notice=pending,
        notify=lambda text, severity: shown.append(text))
    ChatScreen._notify_guardian_visit_summary_if_any(_screen(app, session_id="session-b"))
    assert app._guardian_visit_summary_notice is pending
    assert shown == []
    ChatScreen._notify_guardian_visit_summary_if_any(_screen(app))
    assert len(shown) == 1
    assert app._guardian_visit_summary_notice is None


def test_disabled_mount_preserves_summary_without_surfacing(monkeypatch):
    from tldw_chatbook.Guardian import settings
    monkeypatch.setattr(settings, "guardian_setting", lambda key, default=None: False)
    shown = []
    pending = {"session-a": dict(_SUMMARY)}
    app = SimpleNamespace(_guardian_visit_summary_notice=pending,
        notify=lambda text, severity: shown.append(text))
    ChatScreen._notify_guardian_visit_summary_if_any(_screen(app))
    assert app._guardian_visit_summary_notice is pending
    assert shown == []
