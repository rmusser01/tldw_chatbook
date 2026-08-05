"""Tests for mapping a server ingest job's status onto a local job state.

Pure mapping, so the polling worker's decisions can be pinned without a server
(task-684.2). The vocabulary is taken from the client's own tests
(``Tests/tldw_api/test_media_ingest_jobs_client.py`` uses ``queued``,
``completed`` and ``cancelled``) and from the sibling job APIs in
``tldw_api/`` which declare ``queued|running|completed|failed|cancelled``.
``MediaIngestJobStatus.status`` is an unpinned ``str``, so the unknown case
matters as much as the known ones.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Library.library_ingest_jobs import IngestJobState
from tldw_chatbook.Library.server_ingest_status import (
    is_terminal_server_status,
    local_state_for_server_status,
)


class TestLocalStateForServerStatus:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            ("queued", IngestJobState.QUEUED),
            ("pending", IngestJobState.QUEUED),
            ("running", IngestJobState.PARSING),
            ("processing", IngestJobState.PARSING),
            ("in_progress", IngestJobState.PARSING),
            ("completed", IngestJobState.DONE),
            ("complete", IngestJobState.DONE),
            ("success", IngestJobState.DONE),
            ("succeeded", IngestJobState.DONE),
            ("failed", IngestJobState.FAILED),
            ("error", IngestJobState.FAILED),
            ("cancelled", IngestJobState.CANCELLED),
            ("canceled", IngestJobState.CANCELLED),
        ],
    )
    def test_maps_the_known_vocabulary(self, status: str, expected) -> None:
        assert local_state_for_server_status(status) is expected

    def test_is_case_and_whitespace_insensitive(self) -> None:
        assert local_state_for_server_status("  COMPLETED ") is IngestJobState.DONE

    def test_unknown_status_is_not_treated_as_finished(self) -> None:
        """An unrecognised status must never silently become DONE.

        Marking a job done on a status we do not understand is the same class
        of bug as reporting a successful ingest that stored nothing (task-677):
        the user gets a success they cannot act on. Returning ``None`` keeps the
        job visibly unfinished, which is recoverable.
        """
        assert local_state_for_server_status("reticulating_splines") is None
        assert local_state_for_server_status("") is None
        assert local_state_for_server_status(None) is None

    def test_no_known_status_maps_to_the_local_writing_stage(self) -> None:
        """WRITING is the local single-writer stage; a server job never enters it."""
        mapped = {
            local_state_for_server_status(s)
            for s in ("queued", "running", "completed", "failed", "cancelled")
        }
        assert IngestJobState.WRITING not in mapped


class TestIsTerminalServerStatus:
    @pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "error"])
    def test_finished_statuses_are_terminal(self, status: str) -> None:
        assert is_terminal_server_status(status) is True

    @pytest.mark.parametrize("status", ["queued", "running", "processing"])
    def test_in_flight_statuses_are_not_terminal(self, status: str) -> None:
        assert is_terminal_server_status(status) is False

    def test_unknown_status_is_not_terminal_so_polling_continues(self) -> None:
        """Polling must not stop on a status it does not understand.

        Treating unknown as terminal would abandon a job that is still running;
        treating it as in-flight keeps watching, which is the recoverable
        direction.
        """
        assert is_terminal_server_status("something_new") is False
