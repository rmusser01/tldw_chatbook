"""Live-focus ownership regressions for Library list entry continuations."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.widgets import Button

from Tests.app_thread_resource_fixtures import (
    close_owned_console_workers as close_owned_console_workers,
)
from Tests.console_resource_fixtures import (
    close_owned_console_resources as close_owned_console_resources,
    close_owned_console_test_apps as close_owned_console_test_apps,
)
from Tests.UI.test_library_shell import (
    FailingFirstLibraryMediaScopeService,
    LibraryHarness,
    _active_library_screen,
    _build_test_app as _build_library_test_app,
    _seed_conversations,
    _two_media_items,
    _wait_for_condition,
    _wait_for_library_shell,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_MEDIA,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


def _build_test_app():
    """Build the legacy-profile app assumed by Library harness tests."""
    app = _build_library_test_app()
    app.library_new_profile_admission = False
    return app


class _FocusNode:
    """Small focus identity for receipt-free branch controls."""

    def __init__(
        self,
        *classes: str,
        attached: bool = True,
        widget_id: str | None = None,
    ) -> None:
        self._classes = frozenset(classes)
        self.is_attached = attached
        self.id = widget_id

    def has_class(self, class_name: str) -> bool:
        return class_name in self._classes


class _ContinuationHarness:
    """Record the existing continuation's branch outcome without a mounted app."""

    def __init__(
        self,
        *,
        focused: _FocusNode | None,
        anchor: _FocusNode | None = None,
        programmatic_target: _FocusNode | None = None,
        receipt: object | None = None,
        candidate_receipt: object | None = None,
        pending: bool = True,
        generation: int = 7,
    ) -> None:
        self.focused = focused
        self._library_pending_list_entry_focus_anchor = anchor
        self._library_notes_programmatic_focus_target = programmatic_target
        self._library_pending_list_entry_media_return = receipt
        self._library_pending_list_entry_focus = pending
        self._library_list_entry_focus_generation = generation
        self._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
        self._candidate_receipt = candidate_receipt
        self.focus_calls = 0
        self.disarm_calls = 0
        self.armed_receipts = []

    def _library_media_return_candidate(self, receipt: object | None) -> bool:
        return receipt is not None and receipt is self._candidate_receipt

    def _arm_library_media_return_settlement(self, receipt: object) -> None:
        self.armed_receipts.append(receipt)

    def _library_media_settlement_tree(self):
        return None

    def _focus_library_list_entry(self) -> None:
        self.focus_calls += 1

    def _disarm_library_list_entry_focus(self) -> None:
        self.disarm_calls += 1
        self._library_pending_list_entry_focus = False
        self._library_list_entry_focus_generation += 1


@pytest.mark.asyncio
async def test_retry_timer_preserves_newer_live_retry_focus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retry tick must observe live focus before its queued event is handled."""
    app = _build_test_app()
    _seed_conversations(app, [], media=_two_media_items())
    app.media_reading_scope_service = FailingFirstLibraryMediaScopeService(
        _two_media_items()
    )
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media", Button).press()
        controller = screen._library_media_browse_controller
        await _wait_for_condition(
            pilot,
            lambda: (
                controller.state.error_copy
                and len(screen.query("#library-media-retry")) == 1
                and not controller.state.facet_loading
                and not screen.query_one("#library-media-canvas")._recompose_required
                and screen.query_one("#library-media-retry", Button).region.area > 0
            ),
            message="Initial Media error never exposed one laid-out Retry action.",
        )

        screen._disarm_library_list_entry_focus()
        anchor = screen.query_one("#library-media-type-filter", Button)
        screen.set_focus(anchor)
        assert screen.focused is anchor
        assert anchor.is_attached
        screen._arm_library_list_entry_focus()
        assert screen._library_pending_list_entry_focus_anchor is anchor

        scheduled_ticks = []
        scheduled_timers = []
        real_set_timer = screen.set_timer

        def capture_retry_tick(interval, callback, *args, **kwargs):
            timer = real_set_timer(interval, callback, *args, **kwargs)
            scheduled_ticks.append(callback)
            scheduled_timers.append(timer)
            return timer

        monkeypatch.setattr(screen, "set_timer", capture_retry_tick)
        screen._retry_library_list_entry_focus_while_armed()
        assert len(scheduled_ticks) == 1
        assert scheduled_ticks[0].__name__ == "_tick"

        retry = screen.query_one("#library-media-retry", Button)
        assert retry.is_attached
        assert retry.region.area > 0
        screen.set_focus(retry)
        assert screen.focused is retry
        assert screen._library_pending_list_entry_focus

        try:
            scheduled_ticks[0]()
            assert screen.focused is retry
            assert retry.is_attached
            assert not screen._library_pending_list_entry_focus
            assert screen._library_list_entry_focus_retry_timer is None
            assert screen._library_list_entry_focus_timer is None
            assert screen._library_list_entry_focus_deadline is None
            assert screen._library_pending_list_entry_focus_anchor is None
        finally:
            for timer in scheduled_timers:
                timer.stop()


@pytest.mark.parametrize(
    "case",
    (
        "anchor",
        "armed-row",
        "no-focus",
        "detached",
        "adaptive-grip",
        "programmatic-target",
    ),
)
def test_ordinary_entry_continuation_retains_permitted_focus_owners(case: str) -> None:
    """Existing system-owned or unavailable focus states remain eligible."""
    focused = _FocusNode()
    anchor = _FocusNode()
    programmatic_target = None
    if case == "anchor":
        anchor = focused
    elif case == "armed-row":
        focused = _FocusNode("library-media-row")
    elif case == "no-focus":
        focused = None
    elif case == "detached":
        focused = _FocusNode(attached=False)
    elif case == "adaptive-grip":
        focused = _FocusNode("library-adaptive-reader-pane-grip")
    elif case == "programmatic-target":
        programmatic_target = focused
    harness = _ContinuationHarness(
        focused=focused,
        anchor=anchor,
        programmatic_target=programmatic_target,
    )

    LibraryScreen._focus_library_list_entry_if_current(harness, 7)

    assert harness.focus_calls == 1
    assert harness.disarm_calls == 0
    assert harness._library_pending_list_entry_focus


@pytest.mark.parametrize(
    "case",
    (
        "foreign-nonrow",
        "foreign-list-row",
        "stale-programmatic-target",
        "same-id-replacement",
    ),
)
def test_ordinary_entry_continuation_rejects_newer_foreign_focus(case: str) -> None:
    """Only exact current identities and the armed row class retain authority."""
    focused = _FocusNode()
    anchor = _FocusNode()
    programmatic_target = None
    if case == "foreign-list-row":
        focused = _FocusNode("library-notes-row")
    elif case == "stale-programmatic-target":
        programmatic_target = _FocusNode()
    elif case == "same-id-replacement":
        anchor = _FocusNode(widget_id="same-widget")
        focused = _FocusNode(widget_id="same-widget")
    harness = _ContinuationHarness(
        focused=focused,
        anchor=anchor,
        programmatic_target=programmatic_target,
    )

    LibraryScreen._focus_library_list_entry_if_current(harness, 7)

    assert harness.focus_calls == 0
    assert harness.disarm_calls == 1
    assert not harness._library_pending_list_entry_focus


@pytest.mark.parametrize(
    ("pending", "generation"),
    ((False, 7), (True, 8)),
    ids=("disarmed", "superseded"),
)
def test_stale_entry_continuation_cannot_affect_newer_request(
    pending: bool,
    generation: int,
) -> None:
    """An invalidated callback neither moves focus nor disarms current state."""
    harness = _ContinuationHarness(
        focused=_FocusNode(),
        pending=pending,
        generation=generation,
    )

    LibraryScreen._focus_library_list_entry_if_current(harness, 7)

    assert harness.focus_calls == 0
    assert harness.disarm_calls == 0
    assert harness._library_pending_list_entry_focus is pending
    assert harness._library_list_entry_focus_generation == generation


@pytest.mark.parametrize("candidate", (False, True), ids=("noncandidate", "candidate"))
def test_media_receipts_bypass_ordinary_live_focus_guard(candidate: bool) -> None:
    """Every non-null receipt retains its existing continuation route."""
    receipt = SimpleNamespace(final_focus_policy="filter")
    harness = _ContinuationHarness(
        focused=_FocusNode(),
        receipt=receipt,
        candidate_receipt=receipt if candidate else None,
    )

    LibraryScreen._focus_library_list_entry_if_current(harness, 7)

    assert harness.focus_calls == 1
    assert harness.disarm_calls == 0
    assert harness.armed_receipts == ([receipt] if candidate else [])
