"""The persistent file sink relinquishes storage while other sinks stay live."""

import io
import logging

import pytest

from Tests.Backup_Recovery.test_participant_lifetimes import local_root
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Logging_Config import PrivateRotatingFileHandler


@pytest.mark.parametrize("delay", [False, True])
def test_logging_pause_retires_stream_and_resumes_private_output(
    tmp_path, local_root, delay
):
    path = tmp_path / "logs" / "application.log"
    handler = PrivateRotatingFileHandler(path, encoding="utf-8", delay=delay)
    other_output = io.StringIO()
    other = logging.StreamHandler(other_output)
    logger = logging.Logger("maintenance-test")
    logger.addHandler(handler)
    logger.addHandler(other)
    try:
        logger.warning("before")
        lease = handler.stream._lease
        assert lease in storage._live_leases
        handler._maintenance_close_admission()
        assert lease not in storage._live_leases
        assert handler.stream is None
        logger.warning("during")
        assert path.read_text() == "before\n"
        assert other_output.getvalue() == "before\nduring\n"
        handler._maintenance_resume()
        logger.warning("after")
        assert path.read_text() == "before\nafter\n"
    finally:
        handler.close()
        other.close()


def test_premature_resume_cannot_reopen_a_paused_storage_source(tmp_path, local_root):
    handler = PrivateRotatingFileHandler(tmp_path / "logs" / "application.log")
    handler._maintenance_close_admission()
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(RuntimeError):
            handler._maintenance_resume()
        handler.handle(logging.makeLogRecord({"msg": "still paused"}))
        assert handler.stream is None
    finally:
        pause.resume()
        handler._maintenance_resume()
        handler.close()


def test_terminal_close_during_pause_cannot_revive_handler(tmp_path, local_root):
    handler = PrivateRotatingFileHandler(tmp_path / "logs" / "application.log")
    handler._maintenance_close_admission()
    handler.close()
    with pytest.raises(RuntimeError, match="logging_handler_closed"):
        handler._maintenance_resume()
    assert handler.stream is None


def test_pause_before_delayed_open_does_not_create_log(tmp_path, local_root):
    path = tmp_path / "logs" / "application.log"
    handler = PrivateRotatingFileHandler(path, delay=True)
    try:
        handler._maintenance_close_admission()
        handler.handle(logging.makeLogRecord({"msg": "paused"}))
        assert not path.exists()
    finally:
        handler.close()


def test_failed_stream_retirement_blocks_resume(tmp_path, local_root, monkeypatch):
    handler = PrivateRotatingFileHandler(tmp_path / "logs" / "application.log")
    stream = handler.stream
    original_close = stream.close

    def failed_close():
        raise OSError("injected stream close failure")

    monkeypatch.setattr(stream, "close", failed_close)
    try:
        with pytest.raises(OSError, match="injected stream close failure"):
            handler._maintenance_close_admission()
        assert stream._lease in storage._live_leases
        with pytest.raises(RuntimeError, match="logging_pause_incomplete"):
            handler._maintenance_resume()
        assert handler.stream is stream
    finally:
        monkeypatch.setattr(stream, "close", original_close)
        handler.close()
