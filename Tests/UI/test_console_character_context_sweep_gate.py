"""The character-context 2Hz sweep pauses while the Console is suspended.

It also pauses while the Character section is collapsed (TASK-35970): a
count written to a hidden row requests a layout no one can see.
"""

from types import SimpleNamespace

import pytest

from tldw_chatbook.Widgets.Console.console_character_context import (
    ConsoleCharacterContext,
)

pytestmark = pytest.mark.bootstrap_profile


class _SweepWidget(ConsoleCharacterContext):
    """Bare widget shell: the sweep's collaborators are stubbed per test."""

    is_attached = True
    screen = None  # shadow read-only Widget properties for stubbing
    app = None
    shown = True

    def __init__(self) -> None:  # skip the heavy real widget init
        pass

    def _presented(self) -> bool:  # the stub has no DOM ancestors
        return self.shown

    def query(self, selector):  # noqa: D102 - test stub
        return self._buttons


def _widget(*, current: bool):
    screen = SimpleNamespace()
    app = SimpleNamespace(screen=screen if current else SimpleNamespace())
    widget = _SweepWidget()
    widget.screen = screen
    widget.app = app
    widget._buttons = [
        SimpleNamespace(character_row=object(), progress_count=-1, progress_base_label="Chat")
    ]
    calls = []
    widget._controller = SimpleNamespace(
        pending_progress_count=lambda row: (calls.append(row), 0)[1]
    )
    return widget, calls


def test_sweep_runs_while_the_console_is_current():
    widget, calls = _widget(current=True)
    widget._sync_progress_counts()
    assert calls, "current screen: sweep must still run"


def test_sweep_skips_while_the_console_is_suspended():
    widget, calls = _widget(current=False)
    widget._sync_progress_counts()
    widget._sync_progress_counts()
    assert not calls, "another screen current: sweep must not walk buttons"


def test_sweep_skips_when_detached():
    widget, calls = _widget(current=True)
    widget.is_attached = False
    widget._sync_progress_counts()
    assert not calls, "detached widget: sweep must not walk buttons"


def test_sweep_skips_while_the_section_is_collapsed():
    widget, calls = _widget(current=True)
    widget.shown = False
    widget._sync_progress_counts()
    assert not calls, "collapsed section: sweep must not walk its hidden rows"
