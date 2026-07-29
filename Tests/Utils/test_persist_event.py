"""TASK-1240: persist_event is the single admitted path to the persistent log."""

from __future__ import annotations

import logging

import pytest

from tldw_chatbook.Utils.persistent_diagnostics import (
    PersistentDiagnosticFilter,
    persist_event,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def sink(tmp_path):
    """A real file handler behind the real filter, as the app installs it."""
    path = tmp_path / "app.log"
    handler = logging.FileHandler(path)
    handler.addFilter(PersistentDiagnosticFilter())
    # The same formatter `_configure_private_file_logging` installs. It matters:
    # `%(name)s` puts the *logger name* on disk, and `persist_event` builds that
    # name from its `component` argument. A default-formatted handler writes
    # only `%(message)s`, so it cannot see what `component` does to the record's
    # second, unvalidated destination -- which is exactly the hole Critical-1
    # closed.
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    root.addHandler(handler)
    previous = root.level
    root.setLevel(logging.INFO)
    yield path, handler
    handler.flush()
    root.removeHandler(handler)
    root.setLevel(previous)
    handler.close()


def test_persist_event_reaches_the_sink(sink):
    path, handler = sink
    persist_event("scheduling", "scheduler_configured", item_count=2, status="ok")
    handler.flush()
    written = path.read_text()
    assert "event=scheduler_configured" in written
    assert "component=scheduling" in written
    assert "item_count=2" in written


def test_ordinary_logging_is_still_rejected(sink):
    """The boundary must not widen: only marked records are admitted."""
    path, handler = sink
    logging.getLogger("tldw_chatbook.diagnostics.scheduling").info(
        "an ordinary line that must not persist"
    )
    handler.flush()
    assert path.read_text() == ""


def test_unknown_fields_are_still_rejected(sink):
    """The schema is the guarantee; persist_event must not bypass it."""
    with pytest.raises(ValueError):
        persist_event("app", "app_started", prompt="secret user text")


@pytest.mark.parametrize(
    "component",
    [
        # Both of these are natural readings of the docstring's "component"
        # and both carry caller-supplied text.
        "tool.read_file(path='/home/u/notes/secret.txt')",
        "provider.acme api-key sk-live-abcdef",
        # Non-token shapes generally.
        "",
        "  ",
        "component with spaces",
        "\n".join(["multi", "line"]),
        None,
        object(),
    ],
)
def test_non_token_component_raises_and_writes_nothing(sink, component):
    """`component` is used raw to build the logger name, so it must be a token.

    The persistent formatter is `... %(name)s:%(lineno)d - %(message)s`, so the
    logger name lands on disk. Validating `component` only as a schema field
    would degrade the *field* to `component=invalid` while the *logger name*
    carried the caller's text verbatim -- every guard reporting success.
    `persist_event` therefore rejects it outright rather than substituting: a
    non-token component means the caller misunderstood the contract, and
    quietly writing `invalid` would hide that.
    """
    path, handler = sink
    with pytest.raises(ValueError, match="bounded metadata token"):
        persist_event(component, "app_started")
    handler.flush()
    assert path.read_text() == "", "a rejected component still reached the file"


def test_a_token_component_is_still_accepted(sink):
    """The guard must reject only non-tokens -- dotted namespaces are fine."""
    path, handler = sink
    persist_event("scheduling.watchlists", "scheduler_configured", status="ok")
    handler.flush()
    written = path.read_text()
    assert "component=scheduling.watchlists" in written
    assert "tldw_chatbook.diagnostics.scheduling.watchlists" in written


def test_forward_loguru_to_standard_drops_the_metadata_marker():
    """`_forward_loguru_to_standard` must rebuild `extra` from scratch.

    This asserts directly on the stdlib `LogRecord` the forwarder builds, by
    attaching a plain (unfiltered) handler to the exact logger the forwarder
    targets -- not by routing a record through `PersistentDiagnosticFilter`.
    The filter's admission rule keys on the record's origin (caller
    module/file under `tldw_chatbook/`), not on this marker, so a record
    built from this test file would be rejected as "not chatbook code"
    whether or not the marker survived -- masking a regression here rather
    than detecting it. Bypassing the filter and inspecting the forwarded
    record directly closes that gap: if `_forward_loguru_to_standard` is
    ever changed to carry Loguru's bound extras through -- including
    `_tldw_metadata_only_record` -- any code could write
    `logger.bind(_tldw_metadata_only_record=True).info(secret)` and reach the
    persistent sink without going through `persist_event`'s schema at all.
    """
    from loguru import logger as loguru_logger

    from tldw_chatbook.Logging_Config import _forward_loguru_to_standard

    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    # `_forward_loguru_to_standard` resolves its target logger from the
    # Loguru record's `name`, which Loguru derives from the calling frame --
    # i.e. this module. Attaching the capture handler to that same logger,
    # with no admission filter in front of it, shows exactly what `extra`
    # the forwarder passed along.
    target_logger = logging.getLogger(__name__)
    previous_level = target_logger.level
    previous_propagate = target_logger.propagate
    handler = _Capture()
    target_logger.addHandler(handler)
    target_logger.setLevel(logging.DEBUG)
    target_logger.propagate = False

    # Deliberately NOT `loguru_logger.remove()` first. That bare call drops
    # *every* sink process-wide, while the `finally` below removes only
    # `sink_id`; conftest's autouse cleanup removes handlers *added* during a
    # test and cannot restore ones a test removed, so Loguru would be left with
    # zero sinks for the remainder of the session -- and `Tests/Utils/` is
    # collected before the root `Tests/test_*.py` files. Adding a sink beside
    # whatever is already installed mutates nothing global; the assertions
    # below are written to tolerate the extra forwarders that implies.
    sink_id = loguru_logger.add(_forward_loguru_to_standard, level="TRACE")
    try:
        loguru_logger.bind(_tldw_metadata_only_record=True).info(
            "event=forged component=attacker"
        )
    finally:
        loguru_logger.remove(sink_id)
        target_logger.removeHandler(handler)
        target_logger.setLevel(previous_level)
        target_logger.propagate = previous_propagate

    # Stronger than `len(captured) == 1`: if a second forwarder is already
    # installed the record arrives twice, and *every* copy must be clean. A
    # count assertion would fail on an irrelevant detail while saying nothing
    # about the security property.
    assert captured, "the forwarder produced no stdlib record"
    assert all(not hasattr(r, "_tldw_metadata_only_record") for r in captured)
