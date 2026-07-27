"""Pure reader over the app's five local-server process handles.

Ollama is deliberately excluded -- see the docstring on
``lab_server_status.LAB_SERVER_SOURCES`` -- so it is not exercised here.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.UI.Lab_Modules.lab_server_status import (
    LAB_SERVER_SOURCES,
    LabServerRow,
    read_server_rows,
    servers_chip_text,
)


class _FakeProc:
    """Stands in for subprocess.Popen; poll() is None while alive."""

    def __init__(self, alive: bool) -> None:
        self._alive = alive

    def poll(self):
        return None if self._alive else 0


class _FakeApp:
    def __init__(self, **procs) -> None:
        for attribute, _name in LAB_SERVER_SOURCES:
            setattr(self, attribute, None)
        for attribute, proc in procs.items():
            setattr(self, attribute, proc)


def test_all_six_servers_are_reported_even_when_none_run():
    rows = read_server_rows(_FakeApp())
    assert len(rows) == len(LAB_SERVER_SOURCES) == len(LAB_SERVER_SOURCES)
    assert all(row.running is False for row in rows)


def test_a_live_process_reads_as_running():
    rows = read_server_rows(_FakeApp(llamacpp_server_process=_FakeProc(True)))
    by_name = {row.name: row.running for row in rows}
    assert by_name["llama.cpp"] is True
    assert by_name["vLLM"] is False


def test_an_exited_process_reads_as_stopped():
    """poll() returning an exit code means the server died."""
    rows = read_server_rows(_FakeApp(llamacpp_server_process=_FakeProc(False)))
    assert {row.name: row.running for row in rows}["llama.cpp"] is False


def test_a_missing_attribute_reads_as_stopped():
    """The app may not have set every handle yet; that is not an error."""

    class _Bare:
        pass

    rows = read_server_rows(_Bare())
    assert len(rows) == len(LAB_SERVER_SOURCES)
    assert all(row.running is False for row in rows)


def test_a_process_whose_poll_raises_reads_as_stopped():
    class _Exploding:
        def poll(self):
            raise OSError("process gone")

    rows = read_server_rows(_FakeApp(vllm_server_process=_Exploding()))
    assert {row.name: row.running for row in rows}["vLLM"] is False


def test_row_order_is_stable_and_matches_the_source_order():
    rows = read_server_rows(_FakeApp())
    assert [row.name for row in rows] == [name for _attr, name in LAB_SERVER_SOURCES]


def test_chip_text_counts_running_servers():
    rows = (
        LabServerRow(name="llama.cpp", running=True),
        LabServerRow(name="Ollama", running=True),
        LabServerRow(name="vLLM", running=False),
    )
    assert servers_chip_text(rows) == "Servers: 2 running"


def test_chip_text_when_one_is_running_is_singular():
    rows = (LabServerRow(name="llama.cpp", running=True),)
    assert servers_chip_text(rows) == "Servers: 1 running"


def test_chip_text_when_none_are_running():
    rows = (LabServerRow(name="llama.cpp", running=False),)
    assert servers_chip_text(rows) == "Servers: none running"


@pytest.mark.unit
def test_every_lifecycle_tracked_server_is_reported_by_the_lab_status_list():
    """The Lab status list must not silently omit a tracked provider.

    Ollama was excluded from `LAB_SERVER_SOURCES` because, when this module
    was written, nothing ever assigned `app.ollama_server_process` -- a row
    would have read "stopped" forever. Dev later routed Ollama through the
    shared lifecycle owner, and the exclusion stayed: the Models screen
    under-reported, showing a server count that ignored a running Ollama.

    Asserting the two sets agree means the next provider added to
    `SERVER_PROCESS_ATTRS` cannot go unreported the same way. Compared by
    attribute name rather than display name, since only the attribute is
    what liveness is actually read from.
    """
    from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
        SERVER_PROCESS_ATTRS,
    )

    reported = {attribute for attribute, _ in LAB_SERVER_SOURCES}
    tracked = set(SERVER_PROCESS_ATTRS.values())

    assert tracked - reported == set(), (
        "lifecycle tracks these providers but the Lab status list omits them: "
        f"{sorted(tracked - reported)}"
    )
    assert reported - tracked == set(), (
        "the Lab status list reports attributes nothing publishes: "
        f"{sorted(reported - tracked)}"
    )
