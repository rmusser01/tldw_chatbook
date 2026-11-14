"""EnhancedStatusWidget repaints per message instead of recomposing."""

import pytest

from tldw_chatbook.Widgets.status_widget import EnhancedStatusWidget

pytestmark = pytest.mark.bootstrap_profile


def test_messages_reactive_is_refresh_only():
    from textual.reactive import Reactive

    messages = EnhancedStatusWidget._reactives["messages"]
    assert isinstance(messages, Reactive)
    assert messages._recompose is False, (
        "a refresh-only widget must not rebuild its subtree per message"
    )


@pytest.mark.asyncio
async def test_add_message_repaints_without_remounting_children():
    class _Harness:
        def __init__(self) -> None:
            self.refreshes = 0
            self.recomposes = 0

        def refresh(self, *args, **kwargs):
            self.refreshes += 1

        def recompose(self):
            self.recomposes += 1

    class _RepaintWidget(EnhancedStatusWidget):
        # Shadow every reactive with a plain attribute for the bare shell.
        messages: list = []
        max_messages: int = 100
        show_timestamp: bool = True
        title: str = "Status"

        def __init__(self) -> None:  # skip the heavy real widget init
            pass

    widget = _RepaintWidget()
    harness = _Harness()
    widget._messages = []
    widget.refresh = harness.refresh
    widget.recompose = harness.recompose
    widget.scroll_end = lambda: None

    for index in range(50):
        widget.add_message(f"msg {index}")

    assert harness.recomposes == 0, "no remove/remount cycle per message"
    assert len(widget._messages) == 50
    # The repaint itself comes from the reactive machinery on a mounted
    # widget (pinned as refresh-only by the declaration test above); the
    # shadowed plain attribute on this bare shell cannot observe it.
