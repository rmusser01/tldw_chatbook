"""A hard crash or hang must leave evidence behind.

TASK-26037. Targeted crash guards exist (text-selection, fd exhaustion) but they
catch known cases. An unexpected segfault, a C-extension crash inside a native
dependency, or a deadlock produced nothing to diagnose from -- a named grep for
`excepthook` and `faulthandler` across the package returned three comments and no
installation.

Dumps are treated as sensitive: they contain stack frames from a running
process, so they live under the private log directory with the same restrictive
mode as every other private log.
"""

from __future__ import annotations

import faulthandler
import os
import stat
import sys

import pytest

import tldw_chatbook.Logging_Config as logging_config


@pytest.fixture
def private_log_dir(tmp_path, monkeypatch):
    log_file = tmp_path / "logs" / "tldw_cli_app.log"
    log_file.parent.mkdir(parents=True)
    # No raising=False: the name exists today, and if a refactor moved the
    # call these tests would otherwise silently write into the user's real
    # private log directory and still pass.
    monkeypatch.setattr(logging_config, "get_cli_log_file_path", lambda: log_file)
    return log_file.parent


@pytest.fixture(autouse=True)
def restore_faulthandler():
    was_enabled = faulthandler.is_enabled()
    # `enable_crash_forensics` is idempotent by design (it runs twice in a
    # normal boot), so the module-level stream has to be cleared between tests
    # or every case after the first would short-circuit.
    previous_stream = logging_config._crash_dump_stream
    logging_config._crash_dump_stream = None
    yield
    logging_config._crash_dump_stream = previous_stream
    if not was_enabled:
        faulthandler.disable()


def test_crash_forensics_installs_a_dump_stream(private_log_dir):
    """Asserting `faulthandler.is_enabled()` proves nothing here.

    pytest's own faulthandler plugin enables it before any test runs, so that
    assertion holds with `enable_crash_forensics()` deleted. Assert on what
    THIS function produces instead.
    """
    logging_config._crash_dump_stream = None

    path = logging_config.enable_crash_forensics()

    assert path is not None
    assert logging_config._crash_dump_stream is not None
    assert faulthandler.is_enabled()


def test_dump_file_lives_in_the_private_log_directory(private_log_dir):
    path = logging_config.enable_crash_forensics()

    assert path is not None
    assert path.parent == private_log_dir


def test_dump_file_is_owner_only(private_log_dir):
    path = logging_config.enable_crash_forensics()

    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode & (stat.S_IRWXG | stat.S_IRWXO) == 0, f"mode was {oct(mode)}"


def test_dump_covers_all_threads(private_log_dir, monkeypatch):
    """A deadlock is only diagnosable if every thread's stack is captured."""
    seen = {}

    def spy_enable(file=None, **kwargs):
        # No default for all_threads: defaulting it True would make the
        # assertion hold even if production stopped passing it.
        seen.update(kwargs)

    monkeypatch.setattr(faulthandler, "enable", spy_enable)
    logging_config.enable_crash_forensics()

    assert seen.get("all_threads") is True


@pytest.mark.skipif(
    not hasattr(__import__("signal"), "SIGUSR2"),
    reason="platform has no SIGUSR2",
)
def test_on_demand_dump_signal_is_registered(private_log_dir, monkeypatch):
    """Lets a hung process be inspected without killing it."""
    import signal

    registered = {}

    def spy_register(signum, file=None, **kwargs):
        registered["signum"] = signum
        registered.update(kwargs)

    monkeypatch.setattr(faulthandler, "register", spy_register)
    logging_config.enable_crash_forensics()

    assert registered.get("signum") == signal.SIGUSR2
    assert registered.get("all_threads") is True


def test_oversized_dump_file_is_reset(private_log_dir):
    """AC#5: repeated dumps must not grow the file without limit."""
    dump = private_log_dir / logging_config.CRASH_DUMP_FILENAME
    dump.write_bytes(b"x" * (logging_config.CRASH_DUMP_MAX_BYTES + 1))

    logging_config.enable_crash_forensics()

    assert dump.stat().st_size == 0


def test_dump_under_the_cap_is_preserved(private_log_dir):
    """The most useful dump is the one from the crash that just happened."""
    dump = private_log_dir / logging_config.CRASH_DUMP_FILENAME
    dump.write_bytes(b"previous crash evidence")

    logging_config.enable_crash_forensics()

    assert b"previous crash evidence" in dump.read_bytes()


def test_failure_to_enable_never_breaks_startup(private_log_dir, monkeypatch):
    """Forensics is a diagnostic aid; it must not become a boot failure."""

    def explode(*_args, **_kwargs):
        raise OSError("SENTINEL-FORENSICS-FAILURE")

    monkeypatch.setattr(faulthandler, "enable", explode)

    assert logging_config.enable_crash_forensics() is None


def test_installing_twice_is_a_no_op(private_log_dir):
    """`configure_application_logging` runs twice in a normal boot."""
    first = logging_config.enable_crash_forensics()
    stream_after_first = logging_config._crash_dump_stream

    second = logging_config.enable_crash_forensics()

    assert first is not None
    assert second is None, "the second install should short-circuit"
    assert logging_config._crash_dump_stream is stream_after_first
