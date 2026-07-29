"""TASK-1240: the persistent log must actually contain records.

`tldw_cli_app.log` was zero bytes on every profile from 1df0c4cb4 onward,
because `PersistentDiagnosticFilter` admits only records marked by
`log_persistent_metadata()` and that function had no production callers. Every
existing test passed throughout: they assert a handler is attached, which was
true the whole time.

This is the guard that would have caught it. It asserts on *named* events
rather than on mere non-emptiness: `persistent_sink_installed` is written the
instant the sink installs (see `Logging_Config._configure_private_file_logging`),
so a bare "file is non-empty" check would still pass even if every other
event in this design were broken.
"""

from __future__ import annotations

import logging

import pytest

pytestmark = pytest.mark.unit


def test_a_booted_app_writes_named_events_to_the_persistent_log(
    tmp_path, monkeypatch
):
    """Install the real sink, run the real emitters, read the real file."""
    from tldw_chatbook.Logging_Config import _configure_private_file_logging
    from tldw_chatbook.Utils.persistent_diagnostics import persist_event

    log_path = tmp_path / "tldw_cli_app.log"
    monkeypatch.setattr(
        "tldw_chatbook.Logging_Config.get_cli_log_file_path", lambda: log_path
    )
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(logging.INFO)
    try:
        assert _configure_private_file_logging(root) is True
        persist_event("app", "app_started")
        for handler in root.handlers:
            handler.flush()
        written = log_path.read_text()
    finally:
        root.setLevel(previous_level)

    assert written, "the persistent log is empty after a real install"

    # Non-empty alone is not enough: `persistent_sink_installed` is written the
    # moment the sink installs, so a bare emptiness check would pass with every
    # other event broken.
    assert "event=persistent_sink_installed" in written
    assert "event=app_started" in written

    events = {
        line.split("event=", 1)[1].split(" ", 1)[0]
        for line in written.splitlines()
        if "event=" in line
    }
    assert events - {"persistent_sink_installed"}, (
        "the only event in the log is the sink's own install line"
    )


def test_no_message_text_reaches_the_persistent_log(tmp_path, monkeypatch):
    """The boundary this task must not widen."""
    from tldw_chatbook.Logging_Config import _configure_private_file_logging
    from tldw_chatbook.Utils.persistent_diagnostics import persist_event

    log_path = tmp_path / "tldw_cli_app.log"
    monkeypatch.setattr(
        "tldw_chatbook.Logging_Config.get_cli_log_file_path", lambda: log_path
    )
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(logging.INFO)
    try:
        assert _configure_private_file_logging(root) is True
        logging.getLogger("tldw_chatbook.someplace").info(
            "a user's prompt: PRIVATE-SENTINEL-VALUE"
        )
        persist_event("app", "app_started")
        for handler in root.handlers:
            handler.flush()
        written = log_path.read_text()
    finally:
        root.setLevel(previous_level)

    assert "PRIVATE-SENTINEL-VALUE" not in written
