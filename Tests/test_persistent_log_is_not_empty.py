"""TASK-1240: the persistent log must actually contain records.

`tldw_cli_app.log` was zero bytes on every profile from 1df0c4cb4 onward,
because `PersistentDiagnosticFilter` admits only records marked by
`log_persistent_metadata()` and that function had no production callers. Every
existing test passed throughout: they assert a handler is attached, which was
true the whole time.

This file proves the machinery works end-to-end -- `persist_event` -> filter
-> handler -> file -- using a synthetic `persist_event` call rather than a
booted application. It asserts on *named* events, not mere non-emptiness:
`persistent_sink_installed` is written the instant the sink installs, so a
bare "file is non-empty" check would still pass even if every other event in
this design were broken.

It does not prove production code *calls* `persist_event`; that half lives in
`Tests/App/test_app_lifecycle_events.py` (app_started/app_stopping) and
`Tests/Scheduling/test_scheduler_observability.py` (scheduler_configured).
"""

from __future__ import annotations

import logging

import pytest

pytestmark = pytest.mark.unit


def _pin_file_log_level(monkeypatch, level_name: str = "INFO") -> None:
    """Pin `[logging] file_log_level` for the duration of a test.

    Without this these tests read the ambient user/CI setting. The shipped
    config comment offers `WARNING, ERROR, CRITICAL`, and under any of those
    the handler's own level drops every INFO record -- including
    `app_started` -- so the assertions below would fail for a reason that has
    nothing to do with what they are testing. `_configure_private_file_logging`
    reads the value through the `get_cli_setting` name bound in
    `Logging_Config`, so that is what is replaced; every other key delegates to
    the real function.
    """
    from tldw_chatbook import config as _config

    real_get_cli_setting = _config.get_cli_setting

    def _pinned(section, key, default=None, *args, **kwargs):
        if section == "logging" and key == "file_log_level":
            return level_name
        return real_get_cli_setting(section, key, default, *args, **kwargs)

    monkeypatch.setattr("tldw_chatbook.Logging_Config.get_cli_setting", _pinned)


def _installed_handler(root: logging.Logger, log_path):
    """Find the handler `_configure_private_file_logging` just installed.

    Mirrors the lookup `_configure_private_file_logging` itself performs
    (matching on `baseFilename`), so the test can remove and close exactly
    the handler it added rather than leaving it -- and its open file
    descriptor -- attached to the root logger for the rest of the pytest
    session. Follows the teardown discipline already used by the sibling
    suite in `Tests/test_persistent_diagnostic_boundary.py`.
    """
    from tldw_chatbook.Logging_Config import PrivateRotatingFileHandler

    return next(
        handler
        for handler in root.handlers
        if isinstance(handler, PrivateRotatingFileHandler)
        and handler.baseFilename == str(log_path)
    )


def test_persist_event_reaches_the_persistent_log_through_the_real_sink(
    tmp_path, monkeypatch
):
    """`persist_event` -> filter -> handler -> file, with a synthetic emit.

    No application is booted here and no production emitter runs;
    `persist_event("app", "app_started")` is called directly, using the same
    component/event name production uses. This proves the machinery reached
    by that call actually persists a record to disk -- it does not prove
    production code calls it (see the module docstring for where that is
    proven).
    """
    from tldw_chatbook.Logging_Config import _configure_private_file_logging
    from tldw_chatbook.Utils.persistent_diagnostics import persist_event

    log_path = tmp_path / "tldw_cli_app.log"
    monkeypatch.setattr(
        "tldw_chatbook.Logging_Config.get_cli_log_file_path", lambda: log_path
    )
    _pin_file_log_level(monkeypatch)
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(logging.INFO)
    handler = None
    try:
        assert _configure_private_file_logging(root) is True
        handler = _installed_handler(root, log_path)
        persist_event("app", "app_started")
        handler.flush()
        written = log_path.read_text()
    finally:
        if handler is not None:
            root.removeHandler(handler)
            handler.close()
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
    """The boundary this task must not widen.

    Carries a positive control: without proof that *something* reached the
    file, `"PRIVATE-SENTINEL-VALUE" not in written` would pass identically
    against an empty file -- exactly the failure shape this whole task exists
    to catch.
    """
    from tldw_chatbook.Logging_Config import _configure_private_file_logging
    from tldw_chatbook.Utils.persistent_diagnostics import persist_event

    log_path = tmp_path / "tldw_cli_app.log"
    monkeypatch.setattr(
        "tldw_chatbook.Logging_Config.get_cli_log_file_path", lambda: log_path
    )
    _pin_file_log_level(monkeypatch)
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(logging.INFO)
    handler = None
    try:
        assert _configure_private_file_logging(root) is True
        handler = _installed_handler(root, log_path)
        logging.getLogger("tldw_chatbook.someplace").info(
            "a user's prompt: PRIVATE-SENTINEL-VALUE"
        )
        persist_event("app", "app_started")
        handler.flush()
        written = log_path.read_text()
    finally:
        if handler is not None:
            root.removeHandler(handler)
            handler.close()
        root.setLevel(previous_level)

    # Positive control: the absence assertion below is only meaningful once
    # this passes -- otherwise an empty file would satisfy it too.
    assert "event=app_started" in written

    assert "PRIVATE-SENTINEL-VALUE" not in written


def test_the_install_event_clears_its_own_gate_at_a_raised_file_log_level(
    tmp_path, monkeypatch
):
    """`persistent_sink_installed` must survive whatever `file_log_level` is.

    The whole point of the event is that an empty persistent log means "the
    sink did not install". Emitted at INFO, it is dropped by its own handler
    whenever the user sets `file_log_level` to any of the values the shipped
    `config.py` comment offers -- `WARNING, ERROR, CRITICAL` -- so a user on
    that config sends a zero-byte log and the maintainer reads it as a failed
    install. Emitting at the handler's own level makes the claim true at every
    setting. This test pins the highest of them.
    """
    from tldw_chatbook.Logging_Config import _configure_private_file_logging

    log_path = tmp_path / "tldw_cli_app.log"
    monkeypatch.setattr(
        "tldw_chatbook.Logging_Config.get_cli_log_file_path", lambda: log_path
    )
    _pin_file_log_level(monkeypatch, "CRITICAL")
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(logging.DEBUG)
    handler = None
    try:
        assert _configure_private_file_logging(root) is True
        handler = _installed_handler(root, log_path)
        assert handler.level == logging.CRITICAL
        handler.flush()
        written = log_path.read_text()
    finally:
        if handler is not None:
            root.removeHandler(handler)
            handler.close()
        root.setLevel(previous_level)

    assert "event=persistent_sink_installed" in written, (
        "the install event was filtered out by the handler it proves installed"
    )
