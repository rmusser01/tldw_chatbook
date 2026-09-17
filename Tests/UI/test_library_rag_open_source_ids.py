"""Search result IDs must resolve to the intended local reader or explain why not."""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest
from textual.widgets import Button, Input

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _run_library_search_and_wait_for_open_result,
    _seed_conversations,
    _StaticLibraryRagSearchService,
    _two_conversations,
    _wait_for_library_shell,
)
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    LocalPromptService,
    PromptScopeService,
)


async def _until(pilot, predicate):
    deadline = asyncio.get_running_loop().time() + 4
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            await pilot.pause()
            return
        await pilot.pause(0.02)
    assert predicate()


def _app_with_result(source_type, source_id, *, backend=""):
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        media=[
            {
                "id": 17,
                "title": "Exact Media",
                "type": "document",
                "content": "Exact stored body.",
            }
        ],
    )
    app.library_rag_search_service = _StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "source_id": source_id,
                    "title": "Exact source",
                    "snippet": "stored body",
                    "runtime_backend": backend,
                    "provenance": {"source_type": source_type},
                }
            ]
        }
    )
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("source_type", ["media", "prompt"])
@pytest.mark.parametrize(
    "shape", ["{id}", "{kind}_{id}", "{kind}-{id}", "local:{kind}:{id}"]
)
@pytest.mark.parametrize("gesture", ["button", "keyboard"])
async def test_result_open_resolves_exact_local_record(
    tmp_path, source_type, shape, gesture
):
    prompts = PromptsDatabase(tmp_path / "prompts.db", client_id="rag-open-id-test")
    prompt_id, _, _ = prompts.add_prompt(
        name="Exact Prompt", author=None, details=None, user_prompt="Exact prompt body."
    )
    source_id = shape.format(
        kind=source_type, id=17 if source_type == "media" else prompt_id
    )
    app = _app_with_result(source_type, source_id)
    app.prompt_scope_service = PromptScopeService(
        local_service=LocalPromptService(prompts), server_service=None
    )
    host = LibraryProductionCSSHarness(app)
    try:
        async with host.run_test(size=(170, 50)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _run_library_search_and_wait_for_open_result(screen, pilot, "exact")
            if gesture == "button":
                screen.query_one("#library-rag-open-result-0", Button).press()
            else:
                screen.query_one("#library-rag-result-card-0").focus()
                await pilot.pause()
                await pilot.press("o")
            if source_type == "media":
                await _until(
                    pilot,
                    lambda: (
                        screen._media_state.reader_session.loaded_id == "local:media:17"
                    ),
                )
                assert screen._media_state.detail["content"] == "Exact stored body."
                assert "Exact Media" in str(
                    screen.query_one("#library-media-viewer-title").render()
                )
                assert (
                    app.media_reading_scope_service.detail_calls[-1]["media_id"] == 17
                )
            else:
                await _until(
                    pilot,
                    lambda: (
                        bool(screen.query("#library-prompt-name"))
                        and screen.query_one("#library-prompt-name", Input).value
                        == "Exact Prompt"
                    ),
                )
                assert screen._prompts_state.selected_prompt_id == prompt_id
    finally:
        prompts.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source_type", "source_id", "backend"),
    [
        ("media", "media_bad", ""),
        ("prompt", "prompt_bad", ""),
        ("media", "prompt_17", ""),
        ("prompt", "media_17", ""),
        ("media", "server:media:17", ""),
        ("prompt", "17", "server"),
    ],
)
async def test_unresolvable_result_explains_refusal_without_leaving_search(
    monkeypatch, tmp_path, request, source_type, source_id, backend
):
    app = _app_with_result(source_type, source_id, backend=backend)
    prompts = PromptsDatabase(tmp_path / "prompts.db", client_id="rag-invalid-id-test")
    request.addfinalizer(prompts.close_connection)
    prompts.add_prompt(
        name="Exact Prompt", author=None, details=None, user_prompt="Body"
    )
    app.prompt_scope_service = PromptScopeService(
        local_service=LocalPromptService(prompts), server_service=None
    )
    notify = Mock()
    monkeypatch.setattr(app, "notify", notify)
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _run_library_search_and_wait_for_open_result(screen, pilot, "exact")
        before = screen._library_selected_row_id
        card = screen.query_one("#library-rag-result-card-0")
        notify.reset_mock()
        card.focus()
        await pilot.pause()
        await pilot.press("o")
        await _until(pilot, lambda: notify.called)

        assert "Can't open" in notify.call_args.args[0]
        assert notify.call_args.kwargs["severity"] == "warning"
        assert screen._library_selected_row_id == before
        assert screen.query_one("#library-rag-result-card-0") is card
        assert screen.focused is card
        assert not app.media_reading_scope_service.detail_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("source_type", ["media", "prompt"])
async def test_missing_record_keeps_the_reader_recovery_visible(
    tmp_path, request, source_type
):
    app = _app_with_result(source_type, f"{source_type}_9999")
    prompts = PromptsDatabase(tmp_path / "prompts.db", client_id="rag-missing-id-test")
    request.addfinalizer(prompts.close_connection)
    prompts.add_prompt(name="Existing", author=None, details=None, user_prompt="Body")
    app.prompt_scope_service = PromptScopeService(
        local_service=LocalPromptService(prompts), server_service=None
    )
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _run_library_search_and_wait_for_open_result(screen, pilot, "exact")
        screen.query_one("#library-rag-open-result-0", Button).press()
        if source_type == "media":
            await _until(pilot, lambda: bool(screen._media_state.reader_session.error))
            assert (
                screen._media_state.reader_session.error == "Media item is unavailable."
            )
            assert screen._media_state.reader_session.loaded_id is None
        else:
            await _until(pilot, lambda: bool(screen._prompts_state.detail_error))
            assert (
                "Couldn't load the selected Prompt"
                in screen._prompts_state.detail_error
            )
            assert not screen.query_one("#library-prompt-detail-retry", Button).disabled
        painted = "\n".join(strip.text for strip in screen._compositor.render_strips())
        assert (
            "unavailable" in painted or "Couldn't load the selected Prompt" in painted
        )
