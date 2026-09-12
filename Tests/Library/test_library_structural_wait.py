"""Unit tests for the shared Library structural-wait status helper (task-32055)."""

from __future__ import annotations

import pytest

from tldw_chatbook.Library.library_structural_wait import (
    STRUCTURAL_WAIT_PATIENCE_SECONDS,
    StructuralWait,
)


def _wait(**kwargs) -> StructuralWait:
    defaults = {"label": "Changing folder", "started_at": 100.0}
    defaults.update(kwargs)
    return StructuralWait(**defaults)


def test_status_line_before_the_patience_window_is_the_plain_label() -> None:
    wait = _wait(cancel=lambda: None)

    assert wait.status_line(100.0) == "Changing folder…"
    assert wait.status_line(102.9) == "Changing folder…"


def test_status_line_after_the_patience_window_offers_cancel() -> None:
    wait = _wait(cancel=lambda: None)

    assert wait.status_line(103.0) == "Changing folder… · still working · Cancel"
    assert wait.status_line(140.0) == "Changing folder… · still working · Cancel"


def test_status_line_without_a_cancel_never_advertises_one() -> None:
    """A wait nobody can cancel must not print a Cancel affordance."""
    wait = _wait()

    assert wait.status_line(140.0) == "Changing folder… · still working"


def test_patience_window_is_configurable_and_defaults_to_three_seconds() -> None:
    wait = _wait(cancel=lambda: None)

    assert STRUCTURAL_WAIT_PATIENCE_SECONDS == pytest.approx(3.0)
    assert wait.status_line(101.0, patience_seconds=0.5).endswith("· Cancel")
    assert wait.status_line(105.0, patience_seconds=30.0) == "Changing folder…"


def test_is_slow_flips_when_the_patience_window_closes() -> None:
    wait = _wait()

    assert wait.is_slow(102.9) is False
    assert wait.is_slow(103.0) is True
    assert wait.is_slow(101.0, patience_seconds=0.5) is True


def test_patience_suffix_keeps_a_callers_own_progress_line() -> None:
    """A surface with real progress owns the line; the wait only appends."""
    wait = _wait(cancel=lambda: None)
    progress = "Collecting notes…  3/12"

    assert wait.with_patience_suffix(progress, 102.9) == progress
    assert wait.with_patience_suffix(progress, 103.0) == (
        "Collecting notes…  3/12 · still working · Cancel"
    )
    assert _wait().with_patience_suffix(progress, 103.0) == (
        "Collecting notes…  3/12 · still working"
    )


def test_cancel_runs_once_and_reports_whether_it_fired() -> None:
    calls: list[int] = []
    wait = _wait(cancel=lambda: calls.append(1))

    assert wait.request_cancel() is True
    assert wait.request_cancel() is False
    assert calls == [1]
    assert _wait().request_cancel() is False
