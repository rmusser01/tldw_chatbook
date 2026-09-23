import asyncio
import json
from pathlib import Path

import pytest
from textual.pilot import Pilot
from textual.widgets import Button

OUT = Path("<tmp>/2722-click-probe.jsonl")
PROBE_DELAY_SECONDS = 0.035


def record(value):
    """Append one diagnostic event to the configured JSONL evidence file.

    Args:
        value: JSON-serializable event containing test, mouse or geometry details.
    """
    with OUT.open("a") as out:
        out.write(json.dumps(value) + "\n")


@pytest.fixture(autouse=True)
def probe(monkeypatch, request):
    """Delay pauses by 35ms and record real mouse dispatch for one diagnostic test.

    Args:
        monkeypatch: Pytest fixture restoring patched Pilot/Button methods afterward.
        request: Pytest request providing the test identifier written to each event.
    """
    original = Pilot.click
    pause = Pilot.pause
    pressed = Button._on_click

    async def slow_pause(self, *a, **kw):
        await pause(self, *a, **kw)
        await asyncio.sleep(PROBE_DELAY_SECONDS)

    async def click(self, widget=None, *a, **kw):
        target = (
            self.app.screen.query_one(widget) if isinstance(widget, str) else widget
        )
        before = repr(target.region) if target else None
        hit = await original(self, widget, *a, **kw)
        record(
            {
                "test": request.node.nodeid,
                "target": widget if isinstance(widget, str) else repr(widget),
                "hit": hit,
                "before": before,
                "after": repr(target.region) if target else None,
                "panel": bool(self.app.query("#mcp-inspector-test-panel")),
                "calls": len(self.app.unified_mcp_service.prepared_calls),
            }
        )
        return hit

    async def on_click(self, event):
        record(
            {
                "test": request.node.nodeid,
                "event_target": self.id,
                "disabled": self.disabled,
                "region": repr(self.region),
            }
        )
        return await pressed(self, event)

    monkeypatch.setattr(Pilot, "pause", slow_pause)
    monkeypatch.setattr(Pilot, "click", click)
    monkeypatch.setattr(Button, "_on_click", on_click)
