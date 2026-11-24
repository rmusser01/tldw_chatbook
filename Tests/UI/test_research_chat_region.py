"""Exercise Grounded Chat through the production terminal destination."""

import asyncio
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input, Select, Static, TextArea

from Tests.app_module_patches import patch_app_global, set_app_global
from Tests.UI.app_factory import (
    _build_test_app,
    attach_chachanotes_db,
    swap_app_aliases,
)
from Tests.UI.library_quit_guard_support import _settings_without_splash, _until
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Research_Workspace.contracts import (
    QualifiedWorkspaceRef,
    WorkspaceDataSource,
)
from tldw_chatbook.Research_Workspace.layout_state import ResearchPanePreferences
from tldw_chatbook.UI.Screens.research_workspace_screen import ResearchWorkspaceScreen


pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


class _Provider:
    def __init__(self):
        self.started = asyncio.Event()
        self.hold = False
        self.requests = []
        self.answer = "Evidence-aware answer."

    async def resolve_for_send(self, selection):
        return SimpleNamespace(
            ready=True,
            provider=selection.provider,
            model=selection.explicit_model,
            resolved_destination=SimpleNamespace(
                egress_class="on_device", endpoint_identity="loopback"
            ),
        )

    async def stream_chat(self, resolution, messages, *, signals):
        self.requests.append(messages)
        yield self.answer
        self.started.set()
        if self.hold:
            await asyncio.Event().wait()

    async def aclose(self):
        pass


def build_research_app(tmp_path, monkeypatch):
    app = _build_test_app(configured_default="research_workspace")
    set_app_global(monkeypatch, "get_user_data_dir", lambda: tmp_path / "profile")
    db = attach_chachanotes_db(app, db_path=tmp_path / "notes.sqlite")
    swap_app_aliases(app, local_chat_conversation_service=ChatConversationService(db))
    app.workspace_registry_service.create_workspace(
        workspace_id="study", name="Study group"
    )
    gateway = _Provider()
    app.console_runtime.set_provider_gateway(gateway)
    return app, db, gateway


async def open_chat(app, pilot):
    await _until(
        pilot,
        lambda: isinstance(app.screen, ResearchWorkspaceScreen),
        "Research Workspace",
    )
    screen = app.screen
    composer = screen.query_one("#research-chat-composer", TextArea)
    await _until(pilot, lambda: not composer.disabled, "Grounded Chat ready")
    return screen, composer


async def test_grounded_chat_is_reachable_and_reopens_saved_conversation_and_draft(
    tmp_path, monkeypatch
):
    app, db, provider = build_research_app(tmp_path, monkeypatch)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(160, 40)) as pilot:
            screen, composer = await open_chat(app, pilot)
            screen.query_one("#research-chat-provider", Input).value = "fixture"
            screen.query_one("#research-chat-model", Input).value = "model"
            mode = screen.query_one("#research-chat-mode", Select)
            mode.value = "rag"
            composer.load_text("What does the source say?")
            screen.query_one("#research-chat-send", Button).press()
            status = screen.query_one("#research-chat-status", Static)
            await _until(
                pilot,
                lambda: "selected ready source" in str(status.render()),
                "Grounded readiness explanation",
            )
            assert provider.requests == []

            mode.value = "auto"
            screen.query_one("#research-chat-send", Button).press()
            await _until(
                pilot,
                lambda: "Did not retrieve" in str(status.render()),
                "completed Auto answer",
            )
            session_id = screen.query_one("#research-chat-session", Select).value
            rows = db.get_messages_for_conversation(session_id)
            assert [row["content"] for row in rows] == [
                "What does the source say?",
                "Evidence-aware answer.",
            ]
            assert composer.text == ""

            composer.load_text("A second unsent draft")
            await pilot.press("ctrl+1")
            await _until(pilot, lambda: app.current_tab == "home", "navigation to Home")
            await pilot.press("f5")
            screen, composer = await open_chat(app, pilot)
            assert composer.text == "A second unsent draft"
            assert (
                screen.query_one("#research-chat-session", Select).value == session_id
            )


async def test_stop_is_reachable_in_narrow_chat_and_commits_partial_answer(
    tmp_path, monkeypatch
):
    app, db, provider = build_research_app(tmp_path, monkeypatch)
    provider.hold = True
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(60, 20)) as pilot:
            screen, composer = await open_chat(app, pilot)
            screen.query_one("#research-chat-provider", Input).value = "fixture"
            screen.query_one("#research-chat-model", Input).value = "model"
            screen.query_one("#research-chat-mode", Select).value = "general"
            composer.load_text("Please answer briefly")
            screen.query_one("#research-chat-send", Button).press()
            await asyncio.wait_for(provider.started.wait(), 8)
            stop = screen.query_one("#research-chat-stop", Button)
            stop.focus()
            await pilot.press("enter")
            status = screen.query_one("#research-chat-status", Static)
            await _until(
                pilot, lambda: "Stopped" in str(status.render()), "stopped answer"
            )
            session_id = screen.query_one("#research-chat-session", Select).value
            rows = db.get_messages_for_conversation(session_id)
            assert rows[-1]["assistant_generation_state"] == "stopped"
            assert rows[-1]["content"] == "Evidence-aware answer."


async def test_workspace_picker_preserves_the_draft_and_blocks_switching_during_an_answer(
    tmp_path, monkeypatch
):
    app, _, provider = build_research_app(tmp_path, monkeypatch)
    app.workspace_registry_service.create_workspace(
        workspace_id="z-project", name="Z project"
    )
    study = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "study")
    project = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "z-project")
    provider.hold = True
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(100, 30)) as pilot:
            screen, composer = await open_chat(app, pilot)
            picker = screen.query_one("#research-workspace-picker", Select)
            if picker.value != study:
                picker.value = study
                await _until(
                    pilot,
                    lambda: picker.value == study and not composer.disabled,
                    "Study workspace",
                )
            composer.load_text("Study draft")
            picker.value = project
            await _until(
                pilot,
                lambda: (
                    picker.value == project
                    and not composer.disabled
                    and composer.text == ""
                ),
                "separate project draft",
            )
            screen.query_one("#research-chat-provider", Input).value = "fixture"
            screen.query_one("#research-chat-model", Input).value = "model"
            screen.query_one("#research-chat-mode", Select).value = "general"
            composer.load_text("Project question")
            screen.query_one("#research-chat-send", Button).press()
            await asyncio.wait_for(provider.started.wait(), 8)
            picker.value = study
            status = screen.query_one("#research-chat-status", Static)
            await _until(
                pilot,
                lambda: "Stop the active answer" in str(status.render()),
                "explanation for blocked workspace switch",
            )
            assert picker.value == project
            screen.query_one("#research-chat-stop", Button).press()
            await _until(pilot, lambda: not composer.disabled, "answer stopped")
            picker.value = study
            await _until(
                pilot,
                lambda: (
                    picker.value == study
                    and not composer.disabled
                    and composer.text == "Study draft"
                ),
                "restored Study draft",
            )


async def test_chat_only_overlay_update_does_not_conflict_with_pane_save(
    tmp_path, monkeypatch
):
    app, _, _ = build_research_app(tmp_path, monkeypatch)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(160, 40)) as pilot:
            screen, _ = await open_chat(app, pilot)
            ref = screen.controller.selected_ref
            store = screen.overlay_store
            store.save(
                ref,
                ResearchPanePreferences(),
                expected_revision=0,
                chat_draft="Saved separately",
            )
            screen.query_one("#research-sources-collapse", Button).press()
            await _until(
                pilot,
                lambda: store.load(ref).revision == 2,
                "pane preferences committed",
            )
            overlay = store.load(ref)
            assert not overlay.preferences.sources_open
            assert overlay.chat_draft == "Saved separately"
