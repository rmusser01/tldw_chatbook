"""RAG Answer names the configured provider AND model before Run, and bills
that pair (TASK-34000.21, review finding L-06).

Real app, production stylesheet, one boot per test. Exactly one seam is
faked: `app.library_rag_answer_chat`, a recording callable standing in for
`chat_api_call` -- the kwargs it receives are what the real dispatcher would
have been handed, which is how `model=` is pinned without spending. Its
payload carries the provider's `model` and `usage` keys the real normalizers
emit, so the footer is the production footer. The persisted config is the
sandbox's own `load_settings()` with `[chat_defaults]` and one credential
layered on, as the harness `launch.sh` writes them.

The slower long-model arm lives in the `_extended` sibling, outside the lane.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from textual.widgets import Input, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_rag_result_focus import _assert_painted
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _seed_conversations,
    _StaticLibraryRagSearchService,
    _wait_for_library_rag_query_ready,
    _wait_for_library_shell,
    _wait_for_selector,
)
from Tests.UI.test_product_maturity_gate16_library_search_rag import (
    _rag_result_fixture,
    _switch_to_rag_mode,
    _wait_until,
)
from tldw_chatbook import config as app_config

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


def _persist_provider(monkeypatch, *, provider: str, model: str, key_field: str):
    """Layer `[chat_defaults]` and one resolvable credential over the REAL
    loaded settings (never a hand-built stand-in, so app boot still sees
    the genuine config shape) -- the same technique as gate16's autouse
    fixture, parameterised so an Anthropic arm can be persisted too."""
    monkeypatch.setattr(app_config, "default_api_endpoint", "openai", raising=False)
    real_load_settings = app_config.load_settings

    def _load_settings_with_persisted_pair(*args, **kwargs):
        settings = dict(real_load_settings(*args, **kwargs))
        chat_defaults = dict(settings.get("chat_defaults") or {})
        chat_defaults["provider"] = provider
        chat_defaults["model"] = model
        settings["chat_defaults"] = chat_defaults
        api_settings = dict(settings.get("api_settings") or {})
        provider_settings = dict(api_settings.get(key_field) or {})
        provider_settings["api_key"] = f"sk-test-{key_field}-ready-key"
        api_settings[key_field] = provider_settings
        settings["api_settings"] = api_settings
        return settings

    monkeypatch.setattr(app_config, "load_settings", _load_settings_with_persisted_pair)


def _payload(model: str) -> dict:
    """A non-streaming provider payload in the shape `chat_api_call` returns:
    the handler's own `model` and `usage` keys beside the completion."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "An expired credential caused the incident [S1].",
                }
            }
        ],
        "model": model,
        "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
    }


def _painted(screen) -> str:
    return " ".join(strip.text for strip in screen._compositor.render_strips())


async def _boot_rag_mode(app, host, pilot, question: str):
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-browse-search").press()
    await _switch_to_rag_mode(screen, pilot)
    field = screen.query_one("#library-rag-query-input", Input)
    field.value = question
    await _wait_for_library_rag_query_ready(screen, pilot, question)
    field.focus()
    await pilot.pause()
    return screen, field


@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
async def test_quiet_line_and_request_name_the_configured_model(monkeypatch, size):
    """AC#1, AC#3, AC#5: before Run the painted quiet line names
    openai · gpt-4.1-mini; Enter sends exactly that pair; the footer
    reports the same model back."""
    _persist_provider(
        monkeypatch, provider="OpenAI", model="gpt-4.1-mini", key_field="openai"
    )
    app = _build_test_app()
    _seed_conversations(app, [], notes=[{"id": "note-42", "title": "Incident"}])
    app.library_rag_search_service = _StaticLibraryRagSearchService(
        _rag_result_fixture()
    )
    calls: list[dict] = []

    def answer(**kwargs):
        calls.append(kwargs)
        return _payload("gpt-4.1-mini")

    app.library_rag_answer_chat = answer
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=size) as pilot:
        screen, field = await _boot_rag_mode(
            app, host, pilot, "Why did the incident happen?"
        )
        quiet = screen.query_one("#library-rag-query-quiet-line", Static)
        _assert_painted(screen, quiet)
        assert quiet.region.height == 1
        assert "To openai · gpt-4.1-mini: question + evidence" in _painted(screen)

        await pilot.press("enter")
        await _wait_until(pilot, lambda: len(calls) == 1, "Answer never reached provider")
        await _wait_for_selector(screen, pilot, "#library-rag-answer-provenance")
        await asyncio.wait_for(screen.workers.wait_for_complete(), timeout=10)
        await pilot.wait_for_scheduled_animations()
        await pilot.pause()

        assert calls[0]["api_endpoint"] == "openai"
        assert calls[0]["model"] == "gpt-4.1-mini"
        footer = screen.query_one("#library-rag-answer-provenance", Static)
        assert str(footer.renderable).startswith("openai · gpt-4.1-mini")
        if size == (170, 48):
            _assert_painted(screen, footer)
            assert "openai · gpt-4.1-mini" in _painted(screen)
        # The line named before Run is the line the footer reports.
        assert "To openai · gpt-4.1-mini: question + evidence" in _painted(screen)


async def test_anthropic_chat_defaults_bill_anthropic(monkeypatch):
    """AC#2: `[chat_defaults]` Anthropic / claude-haiku-4-5 bills Anthropic
    with that model -- not the `openai` endpoint fallback -- and the three
    provider-naming lines (before, during, after) agree."""
    _persist_provider(
        monkeypatch,
        provider="Anthropic",
        model="claude-haiku-4-5",
        key_field="anthropic",
    )
    app = _build_test_app()
    _seed_conversations(app, [], notes=[{"id": "note-42", "title": "Incident"}])
    app.library_rag_search_service = _StaticLibraryRagSearchService(
        _rag_result_fixture()
    )
    calls: list[dict] = []
    release = threading.Event()

    def answer(**kwargs):
        calls.append(kwargs)
        assert release.wait(10), "Answer was never released"
        return _payload("claude-haiku-4-5")

    app.library_rag_answer_chat = answer
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(170, 48)) as pilot:
        screen, _field = await _boot_rag_mode(
            app, host, pilot, "Why did the incident happen?"
        )
        assert "To anthropic · claude-haiku-4-5: question + evidence" in _painted(screen)

        try:
            await pilot.press("enter")
            await _wait_until(pilot, lambda: len(calls) == 1, "Answer never reached provider")
            await _wait_for_selector(screen, pilot, "#library-rag-answer-status")
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()
            assert "Asking anthropic…" in _painted(screen)
        finally:
            release.set()
        await _wait_for_selector(screen, pilot, "#library-rag-answer-provenance")
        await asyncio.wait_for(screen.workers.wait_for_complete(), timeout=10)
        await pilot.wait_for_scheduled_animations()
        await pilot.pause()

        assert calls[0]["api_endpoint"] == "anthropic"
        assert calls[0]["model"] == "claude-haiku-4-5"
        footer = screen.query_one("#library-rag-answer-provenance", Static)
        _assert_painted(screen, footer)
        assert "anthropic · claude-haiku-4-5" in _painted(screen)
