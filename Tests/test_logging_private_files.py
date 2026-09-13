from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

import pytest

from tldw_chatbook import config
from tldw_chatbook.Logging_Config import (
    PrivateRotatingFileHandler,
    RedactingFileFormatter,
    _configure_private_file_logging,
)
from tldw_chatbook.Utils.private_paths import PrivatePathError
from tldw_chatbook.Utils.persistent_diagnostics import (
    _PERSISTENT_METADATA_MARKER,
    PersistentDiagnosticFilter,
)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _select_log_name(
    monkeypatch: pytest.MonkeyPatch,
    user_dir: Path,
    selected_name: str,
) -> None:
    monkeypatch.setattr(config, "get_user_data_dir", lambda: user_dir)
    monkeypatch.setattr(
        config,
        "get_cli_setting",
        lambda section, key, default=None: (
            selected_name
            if section == "logging" and key == "log_filename"
            else default
        ),
    )


@pytest.mark.parametrize(
    "selected_name",
    ["", " ", ".", "..", "../outside/escape.log", "nested/escape.log", r"nested\escape.log"],
)
def test_log_filename_rejects_non_basename_values_without_creating_parents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    selected_name: str,
) -> None:
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    _select_log_name(monkeypatch, user_dir, selected_name)

    with pytest.raises(ValueError, match="basename"):
        config.get_cli_log_file_path()

    assert not (tmp_path / "outside").exists()
    assert not (user_dir / "nested").exists()


def test_log_filename_rejects_absolute_path_without_creating_its_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    outside = tmp_path / "outside"
    _select_log_name(monkeypatch, user_dir, str(outside / "escape.log"))

    with pytest.raises(ValueError, match="basename"):
        config.get_cli_log_file_path()

    assert not outside.exists()


def test_log_filename_returns_direct_child_of_secured_user_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    _select_log_name(monkeypatch, user_dir, "application.log")

    assert config.get_cli_log_file_path() == user_dir / "application.log"


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_private_rotating_handler_hardens_active_and_rotated_generations(
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir(mode=0o755)
    active = log_dir / "application.log"
    rotated = log_dir / "application.log.1"
    active.write_text("old active\n", encoding="utf-8")
    rotated.write_text("old rotated\n", encoding="utf-8")
    active.chmod(0o644)
    rotated.chmod(0o644)

    handler = PrivateRotatingFileHandler(
        active,
        maxBytes=100,
        backupCount=2,
        encoding="utf-8",
    )
    try:
        handler.emit(logging.makeLogRecord({"msg": "new record"}))
        handler.doRollover()
    finally:
        handler.close()

    assert _mode(log_dir) == 0o700
    assert _mode(active) == 0o600
    assert _mode(rotated) == 0o600
    assert _mode(log_dir / "application.log.2") == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink contract")
@pytest.mark.parametrize("generation", ["application.log", "application.log.1"])
def test_private_rotating_handler_rejects_symlinked_generation(
    tmp_path: Path,
    generation: str,
) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    outside = tmp_path / "outside-SENTINEL"
    outside.write_text("preserve", encoding="utf-8")
    (log_dir / generation).symlink_to(outside)

    with pytest.raises(PrivatePathError):
        PrivateRotatingFileHandler(
            log_dir / "application.log",
            maxBytes=100,
            backupCount=2,
            encoding="utf-8",
        )

    assert outside.read_text(encoding="utf-8") == "preserve"


class _CollectingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink contract")
def test_unsafe_file_sink_is_omitted_without_removing_other_handlers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    outside = tmp_path / "outside-SENTINEL"
    outside.write_text("preserve", encoding="utf-8")
    active = log_dir / "application.log"
    active.symlink_to(outside)
    monkeypatch.setattr(
        "tldw_chatbook.Logging_Config.get_cli_log_file_path",
        lambda: active,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Logging_Config.get_cli_setting",
        lambda section, key, default=None: default,
    )
    logger = logging.Logger("private-log-test")
    collecting = _CollectingHandler()
    logger.addHandler(collecting)

    installed = _configure_private_file_logging(logger)

    assert installed is False
    assert collecting in logger.handlers
    assert not any(
        isinstance(handler, PrivateRotatingFileHandler)
        for handler in logger.handlers
    )
    assert outside.read_text(encoding="utf-8") == "preserve"
    assert collecting.messages
    assert all("outside-SENTINEL" not in message for message in collecting.messages)
    assert max(map(len, collecting.messages)) < 200


def test_existing_private_handler_removes_the_legacy_metadata_filter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    active = tmp_path / "application.log"
    monkeypatch.setattr(
        "tldw_chatbook.Logging_Config.get_cli_log_file_path",
        lambda: active,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Logging_Config.get_cli_setting",
        lambda section, key, default=None: default,
    )
    root_logger = logging.Logger("existing-private-handler")
    handler = PrivateRotatingFileHandler(
        active,
        maxBytes=100,
        backupCount=1,
        encoding="utf-8",
    )
    root_logger.addHandler(handler)
    try:
        handler.addFilter(PersistentDiagnosticFilter())

        assert _configure_private_file_logging(root_logger) is True

        assert not any(
            isinstance(item, PersistentDiagnosticFilter) for item in handler.filters
        )
    finally:
        handler.close()


def test_successful_install_writes_its_own_first_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TASK-1240: an empty log must mean "the sink did not install".

    `_configure_private_file_logging` catches Exception, warns, and returns
    False, so a permissions or path problem yields an empty file forever -- the
    same silent-failure class this task exists to fix. Emitting one event the
    moment the sink installs makes the two states distinguishable.

    This must exercise the *real* root logger: `persist_event` logs on its own
    `tldw_chatbook.diagnostics.*` namespace and reaches the private file
    handler only via propagation up to wherever that handler is attached
    (the real root, in production). A handler attached to some other named
    logger -- a sibling of `tldw_chatbook.diagnostics.logging`, not an
    ancestor -- would never see it, and the assertions below would fail for a
    reason unrelated to whether the event is actually emitted.
    """
    log_path = tmp_path / "tldw_cli_app.log"
    monkeypatch.setattr(
        "tldw_chatbook.Logging_Config.get_cli_log_file_path", lambda: log_path
    )
    root_logger = logging.getLogger()
    old_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    installed_handler: PrivateRotatingFileHandler | None = None
    try:
        assert _configure_private_file_logging(root_logger) is True
        installed_handler = next(
            handler
            for handler in root_logger.handlers
            if isinstance(handler, PrivateRotatingFileHandler)
            and handler.baseFilename == str(log_path)
        )
        installed_handler.flush()

        written = log_path.read_text()
        assert "event=persistent_sink_installed" in written
        assert "component=logging" in written
    finally:
        if installed_handler is not None:
            root_logger.removeHandler(installed_handler)
            installed_handler.close()
        root_logger.setLevel(old_level)


# --- TASK-23190: redaction at the private file sink ------------------------
#
# The formatter masks credential/PII values for both ordinary records and
# schema-marked events. Exercise both paths through the installed sink so a
# dropped record cannot make a missing redactor appear to pass.

_METADATA_MARKED = {_PERSISTENT_METADATA_MARKER: True}


@pytest.fixture
def private_sink(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Install the real private file sink on the real root logger.

    Yields the active log path; the sink is configured through
    ``_configure_private_file_logging`` -- the same function
    ``configure_application_logging`` calls -- rather than by building a
    handler by hand, so a formatter that is wired only in a test cannot pass.
    """
    log_path = tmp_path / "tldw_cli_app.log"
    monkeypatch.setattr(
        "tldw_chatbook.Logging_Config.get_cli_log_file_path", lambda: log_path
    )
    monkeypatch.setattr(
        "tldw_chatbook.Logging_Config.get_cli_setting",
        lambda section, key, default=None: default,
    )
    root_logger = logging.getLogger()
    old_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    handler: PrivateRotatingFileHandler | None = None
    try:
        assert _configure_private_file_logging(root_logger) is True
        handler = next(
            item
            for item in root_logger.handlers
            if isinstance(item, PrivateRotatingFileHandler)
            and item.baseFilename == str(log_path)
        )
        yield log_path
    finally:
        if handler is not None:
            handler.flush()
            root_logger.removeHandler(handler)
            handler.close()
        root_logger.setLevel(old_level)


def _read_sink(log_path: Path) -> str:
    for handler in logging.getLogger().handlers:
        handler.flush()
    return log_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("message", "secret"),
    [
        pytest.param(
            "provider rejected Authorization: Bearer sk-live-abc123",
            "sk-live-abc123",
            id="authorization-bearer-header",
        ),
        pytest.param(
            "GET https://svc/customsearch/v1?key=AIzaSyLIVE0123456789012345678901234567 failed",
            "AIzaSyLIVE0123456789012345678901234567",
            id="bare-key-query-parameter",
        ),
        pytest.param(
            "POST https://svc/v1/models?api_key=sk-live-abc123 -> 401",
            "sk-live-abc123",
            id="api-key-query-parameter",
        ),
        pytest.param(
            "https://svc/v1?page=2&token=sk-live-abc123 timed out",
            "sk-live-abc123",
            id="token-query-parameter",
        ),
        pytest.param(
            "OPENAI_API_KEY = sk-abcdefghijklmnopqrstuvwx",
            "sk-abcdefghijklmnopqrstuvwx",
            id="assignment-regression",
        ),
    ],
)
def test_file_sink_redacts_secret_shapes_before_they_reach_disk(
    private_sink: Path,
    message: str,
    secret: str,
) -> None:
    """AC-1/AC-3: each shape is written to disk through the real sink, redacted."""

    logging.getLogger("tldw_chatbook.tests.sink").info(
        message, extra=_METADATA_MARKED
    )

    written = _read_sink(private_sink)

    assert secret not in written
    assert "***REDACTED***" in written


def test_file_sink_redacts_secrets_inside_exception_text(
    private_sink: Path,
) -> None:
    """A credential in ``str(exc)`` is formatted from ``exc_info``, not ``msg``.

    This is why the redaction is a formatter and not a ``logging.Filter``: a
    filter rewriting ``record.msg`` leaves the traceback block -- where
    TASK-23108's provider failures actually carry their URLs -- untouched.
    """
    logger = logging.getLogger("tldw_chatbook.tests.sink")
    try:
        raise ValueError("connect failed for https://svc/v1?key=AIzaSyLIVE0123")
    except ValueError:
        logger.info("provider probe failed", exc_info=True, extra=_METADATA_MARKED)

    written = _read_sink(private_sink)

    assert "AIzaSyLIVE0123" not in written
    assert "ValueError" in written


def test_file_sink_leaves_secret_free_records_intact(private_sink: Path) -> None:
    """Negative control: a redactor that eats everything must not pass.

    Asserts the message text survives verbatim, so an over-broad pattern that
    blanked every line would fail here even though every "secret not in
    written" assertion above would still be green.
    """
    message = "ordinary startup record with no credential in it"
    logging.getLogger("tldw_chatbook.tests.sink").info(
        message, extra=_METADATA_MARKED
    )

    written = _read_sink(private_sink)

    assert message in written
    assert "***REDACTED***" not in written


def test_unmarked_secret_bearing_record_keeps_context_without_the_secret(
    private_sink: Path,
) -> None:
    """Ordinary records receive the same credential/PII redaction as metadata."""
    logging.getLogger("tldw_chatbook.tests.sink").error(
        "Authorization: Bearer sk-live-abc123"
    )
    logging.getLogger("httpx").error("GET https://svc/v1?key=AIzaSyLIVE0123")

    written = _read_sink(private_sink)

    assert "sk-live-abc123" not in written
    assert "AIzaSyLIVE0123" not in written
    assert "Authorization" in written
    assert "GET https://svc/v1?key=***REDACTED***" in written


def test_installed_sink_preserves_diagnostic_text_and_masks_private_values(
    private_sink: Path,
) -> None:
    target = logging.getLogger("tldw_chatbook.Chat.console_trace_service")
    target.error(
        "Trace continuation rejected: unsupported_surface_change "
        "api_key=not-a-real-key email=elise@example.test phase=trace_reservation"
    )
    written = private_sink.read_text()
    assert "Trace continuation rejected: unsupported_surface_change" in written
    assert "phase=trace_reservation" in written
    assert "not-a-real-key" not in written
    assert "elise@example.test" not in written


def test_file_sink_masks_character_user_label(private_sink: Path) -> None:
    logging.getLogger("tldw_chatbook.Character_Chat.Character_Chat_Lib").info(
        "Loading character and image for ID: 42, User: Alice Example"
    )
    written = private_sink.read_text(encoding="utf-8")
    assert "Alice Example" not in written
    assert "Loading character and image for ID: 42, User: ***REDACTED***" in written


def test_installed_sink_uses_the_redacting_formatter(private_sink: Path) -> None:
    """AC-2: redaction is a property of the handler, not of any call site."""

    handler = next(
        item
        for item in logging.getLogger().handlers
        if isinstance(item, PrivateRotatingFileHandler)
        and item.baseFilename == str(private_sink)
    )

    assert isinstance(handler.formatter, RedactingFileFormatter)


def test_preexisting_plain_handler_is_upgraded_to_the_redacting_formatter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A handler installed before this change must not keep writing in clear.

    ``_configure_private_file_logging`` returns early when a matching handler
    is already attached; without reconciliation that branch would leave the
    old plain formatter in place for the rest of the process.
    """
    active = tmp_path / "application.log"
    monkeypatch.setattr(
        "tldw_chatbook.Logging_Config.get_cli_log_file_path", lambda: active
    )
    monkeypatch.setattr(
        "tldw_chatbook.Logging_Config.get_cli_setting",
        lambda section, key, default=None: default,
    )
    root_logger = logging.Logger("preexisting-plain-handler")
    handler = PrivateRotatingFileHandler(
        active, maxBytes=100, backupCount=1, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(handler)
    try:
        assert not isinstance(handler.formatter, RedactingFileFormatter)

        assert _configure_private_file_logging(root_logger) is True

        assert isinstance(handler.formatter, RedactingFileFormatter)
    finally:
        handler.close()
