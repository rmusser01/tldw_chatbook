"""Explicit subprocess-pytest driver for an owned production served parent.

Not normally collected: the controller supplies this file explicitly so the
repository's pre-import isolation applies before any application import.
"""

import asyncio
import json
import os
import shutil
import sqlite3
from pathlib import Path

import pytest

from Tests.Canvas.browser import canvas_live_harness as harness
from Tests.Canvas.browser.canvas_release_policy import release_snapshot
from tldw_chatbook.Canvas.profiles import runtime_snapshot_id


@pytest.mark.loopback_network
async def test_owned_parent_driver(tmp_path, monkeypatch):
    control = Path(os.environ["TLDW_CANVAS_RELEASE_CONTROL"])
    settings = json.loads((control / "settings.json").read_text())
    snapshot = release_snapshot(settings["policy"])
    monkeypatch.setattr(harness.serve, "load_profile_snapshot", lambda: snapshot)
    monkeypatch.setattr(harness, "reserve_loopback_port", lambda: settings["port"])
    monkeypatch.setenv("TLDW_CANVAS_RELEASE_POLICY", settings["policy"])
    data = tmp_path / "test_data"
    data.mkdir(exist_ok=True)
    saved = control / "saved.sqlite"
    if saved.exists():
        shutil.copyfile(saved, data / "canvas-live-chatbook.sqlite")
    stack = await harness.start_live_served_stack(
        tmp_path,
        monkeypatch,
        access_token=settings["token"],
        child_module="Tests.Canvas.browser.canvas_live_chatbook_child",
    )
    events = []
    original_request = stack.server._canvas_control_broker.request

    async def observed_request(child_id, message_type, payload, *, timeout):
        result = "ok"
        code = None
        event_canvases = None
        try:
            response = await original_request(
                child_id, message_type, payload, timeout=timeout
            )
            if message_type == "canvas.events.request":
                event_canvases = [
                    item["canvas_id"] for item in response.payload["events"]
                ]
            return response
        except Exception as error:
            result = type(error).__name__
            code = getattr(error, "code", None)
            raise
        finally:
            if message_type != "scope.snapshot.request" or result != "ok":
                events.append(
                    {
                        "type": message_type,
                        "outcome": result,
                        "code": code,
                        "event_canvases": event_canvases,
                    }
                )
            del events[:-160]
            (control / (settings["policy"] + "-events.json")).write_text(
                json.dumps(events)
            )

    monkeypatch.setattr(
        stack.server._canvas_control_broker, "request", observed_request
    )
    observations = []
    original_events = harness.serve._ServedCanvasAuthorityProxy.read_events

    async def observed_events(proxy, scope, *, after_event_id):
        def epochs():
            return [
                {
                    "epoch": session.selection_epoch,
                    "canvas": session.scope.canvas_id,
                    "revision": session.scope.revision_id,
                }
                for session in stack.server._served_canvas_gateway._sessions.values()
            ]

        before = epochs()
        outcome = "ok"
        try:
            return await original_events(proxy, scope, after_event_id=after_event_id)
        except Exception as error:
            outcome = type(error).__name__
            raise
        finally:
            observations.append(
                {
                    "scope_canvas": scope.canvas_id,
                    "scope_revision": scope.revision_id,
                    "before": before,
                    "after": epochs(),
                    "outcome": outcome,
                }
            )
            (control / (settings["policy"] + "-event-scopes.json")).write_text(
                json.dumps(observations)
            )

    monkeypatch.setattr(
        harness.serve._ServedCanvasAuthorityProxy, "read_events", observed_events
    )
    previous = None
    try:
        while not (control / "stop").exists():
            value = {
                "parent": os.getpid(),
                "children": [
                    service._process.pid
                    for service in stack.services
                    if service._process is not None
                ],
                "origin": stack.origin,
                "data": str(data),
                "snapshot": runtime_snapshot_id(snapshot),
                "pending_bridges": sum(
                    session.pending_bridge is not None
                    for session in stack.server._served_canvas_gateway._sessions.values()
                )
                if stack.server._served_canvas_gateway is not None
                else 0,
            }
            if value != previous:
                pending = control / "state.tmp"
                pending.write_text(json.dumps(value))
                pending.replace(control / "state.json")
                previous = value
            await asyncio.sleep(0.05)
        database = data / "canvas-live-chatbook.sqlite"
        if database.exists():
            with (
                sqlite3.connect(f"file:{database}?mode=ro", uri=True) as source,
                sqlite3.connect(saved) as target,
            ):
                source.backup(target)
    finally:
        await stack.aclose()
    assert all(
        service._process is None or service._process.returncode is not None
        for service in stack.services
    )
    assert all(not path.exists() for path in stack.owned_paths)
