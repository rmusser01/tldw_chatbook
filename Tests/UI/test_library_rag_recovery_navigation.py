"""Recovery actions navigate without submitting or losing the Search/RAG draft."""

from unittest.mock import Mock

import pytest
from textual.screen import Screen
from textual.widgets import Button, Input

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_rag_history_keyboard import _open, _settle, _submit, _tab_to
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _seed_conversations,
    _StaticLibraryRagSearchService,
    _two_media_items,
    _wait_for_selector,
)
from Tests.UI.test_product_maturity_gate16_library_search_rag import _rag_result_fixture
from tldw_chatbook.Library.library_rag_answer_service import LibraryRagProviderGate
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId

QUERY = "draft [/archive] résumé"


def _host(theme, *, populated):
    app = _build_test_app()
    _seed_conversations(
        app,
        [],
        notes=[{"id": "note-1", "title": "Research"}] if populated else [],
        media=_two_media_items() if populated else [],
    )
    service = _StaticLibraryRagSearchService({"results": []})
    app.library_rag_search_service = service
    app.library_rag_answer_chat = Mock()
    app.submit_library_ingest_job = Mock()
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    return host, app, service


def _assert_draft(screen, mode, deselected):
    assert screen.query_one("#library-rag-query-input", Input).value == QUERY
    assert screen.query_one("#library-search-input", Input).value == QUERY
    assert screen._rag_search_state.mode == mode
    assert screen._rag_search_state.scope_deselected == deselected


def _assert_no_submissions(app, service):
    assert service.calls == []
    app.library_rag_answer_chat.assert_not_called()
    app.submit_library_ingest_job.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("mode", ["search", "rag"])
async def test_empty_library_recovery_opens_import_and_retains_draft(theme, size, mode):
    host, app, service = _host(theme, populated=False)
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _open(screen, pilot)
        if mode == "rag":
            screen.query_one("#library-rag-mode-toggle", Button).focus()
            await pilot.press("enter")
            await _settle(screen, pilot)
        field = screen.query_one("#library-rag-query-input", Input)
        field.value = QUERY
        field.focus()
        await _settle(screen, pilot)
        assert screen.query_one("#library-rag-run-query", Button).disabled
        await _tab_to(
            screen,
            pilot,
            lambda widget: (
                getattr(widget, "id", None) == "library-rag-open-import-export"
            ),
        )
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-ingest-canvas")
        assert screen._library_selected_row_id == "ingest-import-media"
        assert not host.seen_routes
        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")
        await _settle(screen, pilot)
        _assert_draft(screen, mode, set())
        assert screen.query("#library-rag-open-import-export")
        _assert_no_submissions(app, service)


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("credential_missing", [False, True])
async def test_provider_recovery_routes_and_restores_draft_with_current_readiness(
    theme, size, credential_missing, monkeypatch
):
    gate = LibraryRagProviderGate(
        provider=None,
        credential_recovery="Set OPENAI_API_KEY." if credential_missing else "",
    )
    monkeypatch.setattr(
        library_screen_module, "library_rag_answer_provider_gate", lambda: gate
    )
    host, app, service = _host(theme, populated=True)
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _open(screen, pilot)
        for selector in ("#library-rag-mode-toggle", "#library-rag-scope-toggle-notes"):
            screen.query_one(selector, Button).focus()
            await pilot.press("enter")
            await _settle(screen, pilot)
        screen.query_one("#library-rag-query-input", Input).value = QUERY
        await _settle(screen, pilot)
        # First return leaves the gate blocked; the second observes a newly
        # ready provider. Native evidence covers the actual Settings destination.
        for repaired in (False, True):
            field = screen.query_one("#library-rag-query-input", Input)
            field.focus()
            await _settle(screen, pilot)
            assert screen.query_one("#library-rag-run-query", Button).disabled
            await _tab_to(
                screen,
                pilot,
                lambda widget: (
                    getattr(widget, "id", None) == "library-rag-open-provider-settings"
                ),
            )
            await pilot.press("enter")
            await pilot.pause()
            assert host.seen_routes[-1] == "settings"
            assert host.seen_contexts[-1] == {
                "category": SettingsCategoryId.PROVIDERS_MODELS
            }
            snapshot = screen.save_state()
            await host.switch_screen(Screen())
            if repaired:
                gate = LibraryRagProviderGate(provider="openai")
            screen = LibraryScreen(app)
            screen.restore_state(snapshot)
            await host.switch_screen(screen)
            await _open(screen, pilot)
            await _settle(screen, pilot)
            _assert_draft(screen, "rag", {"notes"})
            assert screen.query_one("#library-rag-run-query", Button).disabled is (
                not repaired
            )
            assert bool(screen.query("#library-rag-open-provider-settings")) is (
                not repaired
            )
            assert screen._rag_search_state.history == ()
            _assert_no_submissions(app, service)


@pytest.mark.asyncio
async def test_deselecting_existing_sources_does_not_offer_import_recovery():
    host, app, service = _host("textual-dark", populated=True)
    async with host.run_test(size=(80, 24)) as pilot:
        screen = _active_library_screen(host)
        await _open(screen, pilot)
        for source in ("notes", "media"):
            screen.query_one(f"#library-rag-scope-toggle-{source}", Button).focus()
            await pilot.press("enter")
            await _settle(screen, pilot)
        assert screen.query_one("#library-rag-run-query", Button).disabled
        assert not screen.query("#library-rag-open-import-export")
        _assert_no_submissions(app, service)


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
async def test_retained_search_refreshes_provider_gate_on_resume_without_rerunning(
    theme, size, monkeypatch
):
    gate = LibraryRagProviderGate(provider=None)
    monkeypatch.setattr(
        library_screen_module, "library_rag_answer_provider_gate", lambda: gate
    )
    host, app, service = _host(theme, populated=True)
    service.result = _rag_result_fixture()
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _open(screen, pilot)
        await _submit(screen, pilot, QUERY)
        screen.query_one("#library-rag-mode-toggle", Button).press()
        await _settle(screen, pilot)
        results = screen.query_one("#library-rag-results")
        history = screen.query_one("#library-rag-history")
        results_children = tuple(results.children)
        state = screen._rag_search_state
        previous_history = state.history
        for provider in ("openai", None):
            await host.push_screen(Screen())
            gate = LibraryRagProviderGate(provider=provider)
            await host.pop_screen()
            await _settle(screen, pilot)
            assert host.screen is screen
            assert screen.query_one("#library-rag-run-query", Button).disabled is (
                provider is None
            )
            assert bool(screen.query("#library-rag-open-provider-settings")) is (
                provider is None
            )
            _assert_draft(screen, "rag", set())
            assert state.history == previous_history
            assert screen.query_one("#library-rag-history") is history
            assert screen.query_one("#library-rag-results") is results
            assert tuple(results.children) == results_children
            assert len(service.calls) == 1
            app.library_rag_answer_chat.assert_not_called()
            app.submit_library_ingest_job.assert_not_called()
