# Tests/Dreams/test_story_service.py
"""Story generation: honest metadata extraction, row-shaped outcomes, chat seam.

Controller ruling (Task 3 review): the resolver must return a closure with
``api_endpoint``/``api_key``/``model`` pre-bound -- a bare ``chat_api_call``
reference would TypeError into the fallback forever. The kwargs contract is
asserted explicitly here.
"""
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from tldw_chatbook.Dreams import story_service
from tldw_chatbook.Dreams.discovery import Candidate
from tldw_chatbook.Dreams.story_service import generate_story


def _chat_ok(body):
    def chat(**kwargs):
        return {"choices": [{"message": {"content": body}}]}
    return chat


def test_generate_story_parses_explicit_event_date_and_kind():
    cand = Candidate("Wednesday 13 at The Crocodile, Nov 3 2026",
                     "https://x/1", "Tickets $40 — Seattle show", "web")
    res = generate_story(_chat_ok("A story about the show."),
                         candidate=cand, snapshot={"topics": [], "region": "Seattle"})
    assert res.status == "complete"
    assert res.kind == "event"
    assert res.event_date is not None and res.event_date.startswith("2026-11-03")
    assert res.location == "Seattle"


def test_generate_story_never_invents_metadata_absent_from_source():
    cand = Candidate("An article about rust", "https://x/2", "No dates or places here.", "web")
    res = generate_story(_chat_ok("A story about rust."),
                         candidate=cand, snapshot={"topics": [], "region": ""})
    assert res.event_date is None and res.kind == "content"


def test_generate_story_empty_and_failed_outcomes_are_rows_not_exceptions():
    empty = generate_story(_chat_ok("   "), candidate=Candidate("t", "https://x/3", "s", "web"),
                           snapshot={"topics": [], "region": ""})
    assert empty.status == "empty"

    def boom(**kwargs):
        raise RuntimeError("provider 500")
    failed = generate_story(boom, candidate=Candidate("t", "https://x/4", "s", "web"),
                            snapshot={"topics": [], "region": ""})
    assert failed.status == "failed" and "provider 500" in failed.error


def test_extract_metadata_bumps_bare_month_day_past_eleven_months():
    # "Jan 5" read in late December is ~11.5 months in the past: the next
    # occurrence is next year. (Year-explicit dates never bump.)
    meta = story_service._extract_metadata(
        "Doors Jan 5", "", now=datetime(2026, 12, 20, tzinfo=UTC))
    event_date, kind = meta
    assert event_date == "2027-01-05" and kind == "event"
    # A near-term bare date keeps this year: "Nov 3" read in September is
    # upcoming, not past.
    event_date, kind = story_service._extract_metadata(
        "Show Nov 3", "", now=datetime(2026, 9, 22, tzinfo=UTC))
    assert event_date == "2026-11-03"


def test_extract_metadata_never_invents_a_day_from_bare_month_year():
    # Review finding 1: "Month YYYY" text carries no day, so the day group
    # must not eat the year's first two digits ("Sept 2026" -> NOT Sept 20).
    event_date, _ = story_service._extract_metadata(
        "Guide updated Sept 2026", "", now=datetime(2026, 9, 22, tzinfo=UTC))
    assert event_date is None
    event_date, _ = story_service._extract_metadata(
        "Class of May 2026 grads", "", now=datetime(2026, 9, 22, tzinfo=UTC))
    assert event_date is None
    # Day-present forms still parse, including dotted abbreviations.
    event_date, _ = story_service._extract_metadata(
        "Doors Sept. 22", "", now=datetime(2026, 9, 22, tzinfo=UTC))
    assert event_date == "2026-09-22"


def test_resolve_dreams_chat_prebinds_endpoint_key_and_model(monkeypatch):
    captured = {}

    def fake_chat_api_call(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(story_service, "chat_api_call", fake_chat_api_call)
    monkeypatch.setattr("tldw_chatbook.Dreams.settings.get_cli_setting",
                        lambda section, key, default=None: default)
    monkeypatch.setattr(
        story_service, "load_cli_config_and_ensure_existence",
        lambda **kw: {"chat_defaults": {"provider": "OpenAI", "model": "gpt-5"}})
    monkeypatch.setattr(
        story_service, "get_provider_readiness",
        lambda provider, config, **kw: SimpleNamespace(api_key="sk-dreams"))

    chat = story_service.resolve_dreams_chat()
    out = chat(messages_payload=[{"role": "user", "content": "hi"}],
               system_message="s", streaming=False, max_tokens=64, temp=0.2)
    assert out["choices"][0]["message"]["content"] == "ok"
    # The closure pre-binds the three wiring kwargs; the caller supplies only
    # payload/sampling -- exactly what query_synthesis/_invoke sends.
    assert captured["api_endpoint"] == "openai"
    assert captured["api_key"] == "sk-dreams"
    assert captured["model"] == "gpt-5"
    assert captured["messages_payload"] == [{"role": "user", "content": "hi"}]
    assert captured["streaming"] is False


def test_resolve_dreams_chat_prefers_dreams_provider_settings(monkeypatch):
    captured = {}

    def fake_chat_api_call(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(story_service, "chat_api_call", fake_chat_api_call)
    monkeypatch.setattr(
        "tldw_chatbook.Dreams.settings.get_cli_setting",
        lambda section, key, default=None: ("zai" if key == "provider"
                                            else "glm-5" if key == "model" else default))
    monkeypatch.setattr(
        story_service, "get_provider_readiness",
        lambda provider, config, **kw: SimpleNamespace(api_key=None))

    chat = story_service.resolve_dreams_chat()
    chat(messages_payload=[{"role": "user", "content": "hi"}],
         system_message="s", streaming=False, max_tokens=64, temp=0.2)
    assert captured["api_endpoint"] == "zai"
    assert captured["model"] == "glm-5"
    assert captured["api_key"] is None


def test_resolve_dreams_chat_raises_when_nothing_resolves(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Dreams.settings.get_cli_setting",
        lambda section, key, default=None: default)
    monkeypatch.setattr(story_service, "load_cli_config_and_ensure_existence",
                        lambda **kw: {})
    with pytest.raises(RuntimeError, match="Dreams provider/model unavailable"):
        story_service.resolve_dreams_chat()
