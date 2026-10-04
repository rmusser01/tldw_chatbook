"""Opt-in read-only macOS pytest census; never closes handles or collects GC.

Load explicitly with this directory on PYTHONPATH and ``-p descriptor_census_probe``.
Use a fresh explicit --basetemp. TLDW_TEST_REQUIRE_FILE_RETIREMENT=1 makes retained
database files fail the process even if the test bodies pass. This is not native
or Windows qualification, and must not run alongside latency measurements.
"""

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

_DARWIN_F_GETPATH = 50
_observations = []


def pytest_configure(config):
    """Refuse platforms without this native observer and require an owned root."""
    if sys.platform != "darwin":
        raise pytest.UsageError("Descriptor census supports macOS only; not qualified")
    if config.getoption("basetemp") is None:
        raise pytest.UsageError(
            "Descriptor census requires a fresh explicit --basetemp"
        )


def pytest_sessionfinish(session):
    """Fail the opt-in retirement gate without changing observed resource state."""
    if os.environ.get("TLDW_TEST_REQUIRE_FILE_RETIREMENT") != "1":
        return
    if any(
        any(
            re.search(r"\.(?:sqlite3?|db)(?:-|$)", name)
            for name in row["private_files"]
        )
        for row in _observations
    ):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_teardown(item):
    """Count only this process's private profile and explicit temporary-root files."""
    import fcntl

    yield
    files = Counter()
    basetemp = Path(item.config.getoption("basetemp")).resolve()
    for name in os.listdir("/dev/fd"):
        try:
            target = fcntl.fcntl(int(name), _DARWIN_F_GETPATH, bytes(1024))
            path = target.split(b"\0", 1)[0].decode()
        except (OSError, ValueError, UnicodeError):
            continue
        if (
            "/tldw-chatbook-test-" in path
            or "/tldw_test_config_" in path
            or Path(path).is_relative_to(basetemp)
        ):
            files[Path(path).name] += 1
    _observations.append({"test": item.name, "private_files": dict(files)})


def pytest_terminal_summary(terminalreporter):
    """Retain all observations and state whether retirement was required."""
    terminalreporter.write_line("PRIVATE-FD-CENSUS " + json.dumps(_observations))
    terminalreporter.write_line(
        "FILE-RETIREMENT-REQUIRED "
        + str(os.environ.get("TLDW_TEST_REQUIRE_FILE_RETIREMENT") == "1")
    )
