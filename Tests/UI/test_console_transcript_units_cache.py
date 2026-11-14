"""Transcript grouping runs once per ingested message list, shared by consumers."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.Widgets.Console import console_transcript as transcript_module
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

pytestmark = pytest.mark.bootstrap_profile


class _StandaloneUnit:
    def __init__(self, message_id: str) -> None:
        self.standalone = SimpleNamespace(id=message_id)
        self.assistant_turn = None


def _bare_transcript() -> ConsoleTranscript:
    widget = ConsoleTranscript.__new__(ConsoleTranscript)
    widget._messages = []
    widget._units_cache_source = None
    widget._units_cache = ()
    widget._ownership_cache_source = None
    widget._ownership_cache = None
    return widget


def _patch_grouping(monkeypatch, units_by_len):
    calls = []

    def counting(messages):
        calls.append(tuple(m.id for m in messages))
        return tuple(_StandaloneUnit(m.id) for m in messages)

    monkeypatch.setattr(
        transcript_module, "group_console_transcript_messages", counting
    )
    return calls


def test_grouping_runs_once_per_ingest_list_across_consumers(monkeypatch):
    widget = _bare_transcript()
    calls = _patch_grouping(monkeypatch, None)

    widget._messages = [SimpleNamespace(id="a"), SimpleNamespace(id="b")]
    widget._units_for_current_messages()
    widget._ownership_by_message_id()
    widget._ownership_by_message_id()
    assert len(calls) == 1, "units and ownership share ONE walk per list"

    widget._messages = [SimpleNamespace(id="a"), SimpleNamespace(id="b")]
    widget._units_for_current_messages()  # new list object: regroup
    assert len(calls) == 2


def test_ownership_maps_every_id_from_the_shared_units(monkeypatch):
    widget = _bare_transcript()
    _patch_grouping(monkeypatch, None)

    widget._messages = [SimpleNamespace(id="a"), SimpleNamespace(id="b")]
    ownership = widget._ownership_by_message_id()
    assert ownership == {
        "a": ("a", ("a",)),
        "b": ("b", ("b",)),
    }


def test_explicit_message_lists_stay_uncached(monkeypatch):
    widget = _bare_transcript()
    calls = _patch_grouping(monkeypatch, None)

    external = [SimpleNamespace(id="x")]
    widget._ownership_by_message_id(external)
    widget._ownership_by_message_id(external)
    assert len(calls) == 2, "explicit lists are grouped per call, never cached"
