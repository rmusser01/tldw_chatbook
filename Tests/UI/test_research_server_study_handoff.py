"""Verify that Studio opens the exact matching Server Study workspace and owner."""

from types import SimpleNamespace

import pytest
from textual.widgets import Select

from Tests.app_module_patches import patch_app_global
from Tests.UI.app_factory import swap_app_aliases
from Tests.UI.library_quit_guard_support import _settings_without_splash, _until
from Tests.UI.test_research_chat_region import build_research_app, open_chat
from tldw_chatbook.Research_Workspace.contracts import QualifiedWorkspaceRef
from tldw_chatbook.Research_Workspace.studio_models import WorkspaceOutputRef
from tldw_chatbook.Study_Interop.quiz_scope_service import QuizScopeService
from tldw_chatbook.Study_Interop.study_scope_service import StudyScopeService
from tldw_chatbook.UI.Navigation.research_output_handoffs import output_handoff
from tldw_chatbook.runtime_policy.server_event_scope import (
    event_principal_id_from_active_context,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize(
    ("kind", "owner_kind", "selector"),
    [
        ("flashcards", "server_study_deck", "#deck-select"),
        ("quiz", "server_quiz", "#quiz-select"),
    ],
)
async def test_open_server_study_owner_beyond_first_page(
    tmp_path, monkeypatch, kind, owner_kind, selector
):
    app, _, _ = build_research_app(tmp_path, monkeypatch)
    context = SimpleNamespace(
        active_server_id="test-server",
        auth_token="fixture",
        credential_source="test",
    )
    ref = QualifiedWorkspaceRef(
        "server",
        "exact-ws",
        context.active_server_id,
        event_principal_id_from_active_context(context),
    )
    output = WorkspaceOutputRef(
        ref, kind, owner_kind, "301", 1, (), "a" * 64, "complete"
    )
    opened = []

    class StudyPages:
        async def list_workspace_decks(self, workspace_id, **kwargs):
            assert workspace_id == ref.workspace_id
            return [{"id": 4, "workspace_id": workspace_id, "name": "First page"}]

        async def list_workspace_quizzes(self, workspace_id, **kwargs):
            return await self.list_workspace_decks(workspace_id, **kwargs)

        async def get_deck(self, owner_id):
            opened.append(str(owner_id))
            return {
                "id": int(owner_id),
                "workspace_id": ref.workspace_id,
                "name": "Exact output",
            }

        async def get_quiz(self, owner_id):
            return await self.get_deck(owner_id)

        async def list_flashcards(self, **kwargs):
            return []

        async def list_questions(self, *args, **kwargs):
            return []

        async def list_attempts(self, *args, **kwargs):
            return []

    pages = StudyPages()
    swap_app_aliases(
        app,
        study_scope_service=StudyScopeService(local_service=None, server_service=pages),
        study_quiz_scope_service=QuizScopeService(server_service=pages),
    )
    channel, _ = output_handoff(output)
    store = app.pending_handoffs
    staged = []
    stage = store.stage

    def capture_revision(staged_channel, value):
        revision = stage(staged_channel, value)
        if staged_channel is channel:
            staged.append(revision)
        return revision

    monkeypatch.setattr(store, "stage", capture_revision)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(120, 34)) as pilot:
            research, _ = await open_chat(app, pilot)
            monkeypatch.setattr(
                app, "get_authoritative_runtime_source", lambda: "server"
            )
            active = SimpleNamespace(
                active_server_id="different-server",
                auth_token=context.auth_token,
                credential_source=context.credential_source,
            )
            monkeypatch.setattr(
                app.server_context_provider, "get_active_context", lambda: active
            )

            research._open_studio_owner(output)
            assert app.screen is research and not staged
            active.active_server_id = context.active_server_id
            active.auth_token = "fixture-other-user"
            research._open_studio_owner(output)
            assert app.screen is research and not staged

            active = context
            research._open_studio_owner(output)
            await _until(
                pilot, lambda: app.current_tab == "study", "Server Study opens"
            )
            try:
                await _until(
                    pilot,
                    lambda: (
                        bool(app.screen.query(selector))
                        and app.screen.query_one(selector, Select).value
                        == output.owner_id
                        and staged
                        and store.exact_revision_status(channel, staged[-1])
                        == "settled"
                    ),
                    "exact Server Study owner selected",
                    timeout=8,
                )
            except AssertionError as exc:
                study = app.screen
                selected = (
                    study.query_one(selector, Select).value
                    if study.query(selector)
                    else None
                )
                raise AssertionError(
                    f"{study.scope_state=}, {study.current_section=}, {selected=}, {opened=}, "
                    f"pending={app.pending_handoffs.has_pending(channel)}, "
                    f"workers={list(app.workers)}"
                ) from exc
            assert app.screen.scope_state.workspace_id == ref.workspace_id
            assert opened and set(opened) == {"301"}
            back = app.screen.query_one("#study-back-to-workspace-button")
            back.press()
            await _until(
                pilot,
                lambda: app.current_tab == "research_workspace",
                "return to Research",
            )
