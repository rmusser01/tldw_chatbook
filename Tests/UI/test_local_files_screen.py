"""Mounted Local Files search and receipt validation.

The private-profile scans run in targeted UI verification. They are outside
the PR UI census, like the Workspace index Settings tests.
"""

import asyncio
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace

import pytest
from textual.app import App
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Select, Static

from Tests.private_profile import private_profile_test
from Tests.Workspaces.test_project_index_service import Rig


class LocalFilesApp(App):
    CSS_PATH = "../../tldw_chatbook/css/tldw_cli_modular.tcss"

    def __init__(self, rig):
        super().__init__()
        self.app_config = {}
        self.workspace_registry_service = SimpleNamespace(
            db=None,
            subscribe_project_index_changes=lambda _callback: lambda: None,
            list_workspaces=lambda: (
                SimpleNamespace(
                    workspace_id=rig.key.workspace_id, name="Test workspace"
                ),
            ),
            list_folder_bindings=lambda _workspace_id: (
                SimpleNamespace(binding_id=rig.key.binding_id, locator=str(rig.root)),
            ),
        )

    def on_navigate_to_screen(self, message):
        from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_target

        _, _, screen_type = resolve_screen_target(message.screen_name)
        assert screen_type is not None, "Local Files must have a registered route"
        self.push_screen(screen_type(self))


async def wait_for(pilot, condition):
    async with asyncio.timeout(45):
        while not condition():
            await pilot.pause(0.02)


@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
@private_profile_test
async def test_local_files_search_preview_and_superseded_query(
    tmp_path, request, monkeypatch, size
):
    rig = Rig(tmp_path)
    source = rig.root / "quartz.txt"
    source.write_text("intro\nbooking policy\n", encoding="utf-8")
    (rig.root / "amber.txt").write_text(
        "other keyword\nremaining preview context\n", encoding="utf-8"
    )
    rig.enable()
    rig.refresh()
    rig.settle(seconds=60)
    started, release, finished = Event(), Event(), Event()
    retry_started, release_retry = Event(), Event()
    original_search = rig.service.search_binding
    held_stop = None
    busy_once = True

    def search(key, query, *, stop, mode, keyword_query):
        nonlocal held_stop, busy_once
        if query == "amber.txt":
            held_stop = stop
            started.set()
            try:
                assert release.wait(45)
                return original_search(
                    key, query, stop=stop, mode=mode, keyword_query=keyword_query
                )
            finally:
                finished.set()
        if query == "booking policy":
            assert held_stop is not None and held_stop.is_set()
            if busy_once:
                busy_once = False
                raise RuntimeError("project_index_search_busy")
            retry_started.set()
            assert release_retry.wait(45)
        return original_search(
            key, query, stop=stop, mode=mode, keyword_query=keyword_query
        )

    monkeypatch.setattr(rig.service, "search_binding", search)
    from tldw_chatbook.Workspaces.project_index import runtime

    monkeypatch.setattr(runtime, "get_project_index_service", lambda _app: rig.service)
    monkeypatch.setattr(runtime, "project_index_key", lambda *args: rig.key)
    monkeypatch.setattr(
        runtime, "project_index_registry_connections", lambda _db: nullcontext()
    )

    from tldw_chatbook.app_command_providers import TabNavigationProvider
    from tldw_chatbook.Chat.citation_evidence_models import (
        EvidenceBundle,
        EvidenceReference,
    )
    from tldw_chatbook.Library.library_rag_answer_service import LibraryRagAnswer
    from tldw_chatbook.UI.Screens import local_files_screen
    from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
    from tldw_chatbook.Workspaces.project_index import file_answers

    destination = file_answers.FileAnswerDestination(
        "moonshot", "test-model", "https://example.invalid/v1/chat/completions", b""
    )
    answer_calls = []
    revealed = []
    monkeypatch.setattr(
        file_answers, "list_answer_destinations", lambda _config: (destination,)
    )
    monkeypatch.setattr(
        local_files_screen, "_reveal_file", lambda path: revealed.append(path) or True
    )

    async def answer_files(**kwargs):
        answer_calls.append((kwargs["approved"], kwargs["query"]))
        hit = kwargs["search"].result.hits[0]
        return LibraryRagAnswer(
            status="ready",
            text="The policy is in [S1].",
            evidence_bundle=EvidenceBundle(
                bundle_id="bundle",
                query=kwargs["query"],
                references=(
                    EvidenceReference(
                        evidence_id="S1",
                        source_id="source",
                        source_type="workspace_file",
                        title=f"{hit.relative_path} · lines {hit.line_start}–{hit.line_start}",
                        authority_label="Selected folder",
                        snippet=hit.excerpt.splitlines()[0],
                        metadata={
                            "hit_index": 0,
                            "relative_path": hit.relative_path,
                            "line_start": hit.line_start,
                            "line_end": hit.line_start,
                        },
                    ),
                ),
            ),
        )

    monkeypatch.setattr(file_answers, "answer_files", answer_files)
    app = LocalFilesApp(rig)
    try:
        async with app.run_test(size=size) as pilot:
            provider = TabNavigationProvider(app.screen)
            commands = [hit async for hit in provider.search("Local Files")]
            command = next(hit for hit in commands if hit.text == "Open Local Files")
            command.command()
            await pilot.pause()
            screen = app.screen
            folder = screen.query_one("#local-files-folder", Select)
            await wait_for(pilot, lambda: not folder.disabled)
            folder.value = rig.key.binding_id
            search_button = screen.query_one("#local-files-search", Button)
            await wait_for(pilot, lambda: not search_button.disabled)
            assert search_button.region.right <= size[0]
            query = screen.query_one("#local-files-query", Input)
            results = screen.query_one("#local-files-results", OptionList)
            mode = screen.query_one("#local-files-mode", Select)
            assert mode.value == "keyword"
            status = screen.query_one("#local-files-status", Static)
            mode.value = "semantic"
            query.value = "quartz.txt"
            await pilot.pause()
            search_button.press()
            await wait_for(
                pilot, lambda: "Semantic search unavailable" in str(status.render())
            )
            assert not results.option_count
            mode.value = "keyword"
            await pilot.pause()

            query.value = "amber.txt"
            query.focus()
            await pilot.press("enter")
            await wait_for(pilot, started.is_set)
            query.value = "booking policy"
            await pilot.press("enter")
            await wait_for(pilot, retry_started.is_set)
            await app.push_screen(ModalScreen())
            release_retry.set()
            await wait_for(pilot, lambda: results.option_count == 1)
            assert "quartz.txt" in str(results.get_option_at_index(0).prompt)
            await app.pop_screen()

            release.set()
            await wait_for(pilot, finished.is_set)
            await app.workers.wait_for_complete()
            assert "quartz.txt" in str(results.get_option_at_index(0).prompt)

            results.focus()
            results.highlighted = 0
            await pilot.press("enter")
            preview = screen.query_one("#local-files-preview", Static)
            await wait_for(pilot, lambda: "booking policy" in str(preview.render()))
            assert "quartz.txt" in str(preview.render())
            screen.query_one("#local-files-reveal", Button).press()
            await wait_for(pilot, lambda: revealed == [source])

            source.write_text("changed after search\n", encoding="utf-8")
            await pilot.press("enter")
            status = screen.query_one("#local-files-status", Static)
            await wait_for(pilot, lambda: "Search again" in str(status.render()))
            assert "booking policy" not in str(preview.render())

            select = screen.query_one("#local-files-destination", Select)
            await wait_for(pilot, lambda: not select.disabled)
            select.value = 0
            question = "What is the other keyword?"
            query.value = question
            ask = screen.query_one("#local-files-ask", Button)
            await wait_for(pilot, lambda: not ask.disabled)
            assert ask.region.right <= size[0]
            ask.press()
            await wait_for(
                pilot,
                lambda: isinstance(app.screen, ConfirmationDialog)
                and app.screen.query("#cancel-button"),
            )
            assert destination.label in app.screen.message
            app.screen.query_one("#cancel-button", Button).press()
            await wait_for(pilot, lambda: "Answer canceled" in str(status.render()))
            assert answer_calls == []

            ask.press()
            await wait_for(
                pilot,
                lambda: isinstance(app.screen, ConfirmationDialog)
                and app.screen.query("#confirm-button"),
            )
            app.screen.query_one("#confirm-button", Button).press()
            citations = screen.query_one("#local-files-citations", OptionList)
            await wait_for(pilot, lambda: citations.option_count == 1)
            assert answer_calls == [(True, question)]
            assert "[S1] amber.txt · lines 1–1" in str(
                citations.get_option_at_index(0).prompt
            )
            citations.focus()
            citations.highlighted = 0
            await pilot.press("enter")
            await wait_for(
                pilot, lambda: "remaining preview context" in str(preview.render())
            )
    finally:
        release.set()
        release_retry.set()
        await rig.close()
