"""Reusable destination workbench panes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Static


DESTINATION_MODE_STRIP_HEIGHT = 1


@dataclass(frozen=True)
class WorkbenchPane:
    """A titled pane in a destination workbench."""

    title: str
    content: Widget | Iterable[Widget]
    id: str
    classes: str = ""


class DestinationWorkbench(Horizontal):
    """Three-pane terminal-native destination workbench."""

    DEFAULT_CSS = """
    DestinationWorkbench {
        width: 100%;
        height: 1fr;
        min-height: 0;
    }

    .destination-workbench-pane {
        width: 1fr;
        min-width: 0;
        height: 100%;
        min-height: 0;
    }

    .destination-pane-title {
        height: 1;
        min-height: 1;
    }
    """

    def __init__(self, *panes: WorkbenchPane, **kwargs) -> None:
        classes = kwargs.pop("classes", "")
        super().__init__(classes=f"destination-workbench {classes}".strip(), **kwargs)
        self.panes = panes

    def compose(self) -> ComposeResult:
        for pane in self.panes:
            with Vertical(id=pane.id, classes=f"destination-workbench-pane {pane.classes}".strip()):
                yield Static(pane.title, classes="destination-pane-title")
                content = pane.content
                if isinstance(content, Widget):
                    yield content
                else:
                    yield from content


class DestinationModeStrip(Horizontal):
    """Compact one-row mode/action strip for destination screens."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.styles.height = DESTINATION_MODE_STRIP_HEIGHT
        self.styles.min_height = DESTINATION_MODE_STRIP_HEIGHT
