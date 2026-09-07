"""Opt-in ownership evidence for the character Keyword release test subset.

Load with ``-p Tests.fixtures.character_keyword_resources``. Each fixture tracks
the real database owners constructed by that test and drains their existing
quiescence registry after the mounted app/worker fixtures have settled.
"""

from __future__ import annotations

import os
import sys

import pytest

_START_FDS = 0
_OWNERS = 0
_HANDLES_DRAINED = 0
_REMAINING = 0


def _fd_count() -> int:
    return len(os.listdir("/dev/fd" if sys.platform == "darwin" else "/proc/self/fd"))


def pytest_sessionstart(session: pytest.Session) -> None:
    global _START_FDS
    _START_FDS = _fd_count()


@pytest.fixture(autouse=True)
def keyword_database_owners(monkeypatch: pytest.MonkeyPatch):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    owners = []
    original_init = CharactersRAGDB.__init__

    def tracked_init(owner, *args, **kwargs):
        original_init(owner, *args, **kwargs)
        owners.append(owner)

    monkeypatch.setattr(CharactersRAGDB, "__init__", tracked_init)
    yield
    global _OWNERS, _HANDLES_DRAINED, _REMAINING
    for owner in owners:
        _OWNERS += 1
        _HANDLES_DRAINED += owner.registered_connection_count()
        with owner.quiesce_connections(timeout_seconds=2.0):
            pass
        remaining = owner.registered_connection_count()
        _REMAINING += remaining
        assert remaining == 0, "Keyword fixture left registered SQLite handles"


def pytest_terminal_summary(terminalreporter) -> None:
    end = _fd_count()
    terminalreporter.write_line(
        f"Keyword ownership: owners={_OWNERS}, handles_drained={_HANDLES_DRAINED}, "
        f"remaining={_REMAINING}; process FDs start={_START_FDS}, end={end}, "
        f"delta={end - _START_FDS}. Subset evidence only."
    )
