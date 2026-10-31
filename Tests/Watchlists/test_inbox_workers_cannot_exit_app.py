"""TASK-21238: the Watchlists inbox workers must not take the app down.

`_mark_notification_read` and `_dismiss_notification` touch the (lazy,
TASK-21105) client notifications store with no internal guard of their own.
A corrupt/unopenable store raising through either worker used to reach
Textual's default ``exit_on_error=True`` and exit the whole application.
These tests pin ``exit_on_error=False`` on both ``run_worker`` call sites so
the regression (a plain call-shape edit) cannot return silently.
"""

from __future__ import annotations

import re
from pathlib import Path

_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "UI"
    / "Screens"
    / "watchlists_collections_screen.py"
).read_text(encoding="utf-8")


def _worker_call_body(group: str) -> str:
    """Return the run_worker(...) call for ``group``, parentheses included."""
    pattern = re.compile(
        r"self\.run_worker\((?:[^()]|\([^()]*\))*?group=\"" + re.escape(group) + r"\"(?:[^()]|\([^()]*\))*?\)",
        re.DOTALL,
    )
    matches = pattern.findall(_SOURCE)
    assert len(matches) == 1, f"expected exactly one run_worker for {group!r}"
    return matches[0]


def test_mark_notification_read_worker_cannot_exit_the_app():
    body = _worker_call_body("wc_mark_notification_read")
    assert "exit_on_error=False" in body, (
        "The mark-read inbox worker must schedule with exit_on_error=False: "
        "a corrupt notifications DB raising through it would otherwise exit "
        "the whole application (TASK-21238)."
    )


def test_dismiss_notification_worker_cannot_exit_the_app():
    body = _worker_call_body("wc_dismiss_notification")
    assert "exit_on_error=False" in body, (
        "The dismiss inbox worker must schedule with exit_on_error=False: "
        "a corrupt notifications DB raising through it would otherwise exit "
        "the whole application (TASK-21238)."
    )
