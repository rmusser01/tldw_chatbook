"""Verify exact canonical reopening and lifecycle controls from mounted Studio."""

from dataclasses import replace

import pytest
from textual.widgets import Button, Select, Static, TextArea

from Tests.app_module_patches import patch_app_global
from Tests.UI.app_factory import swap_app_aliases
from Tests.UI.library_quit_guard_support import _settings_without_splash, _until
from Tests.UI.test_research_chat_region import build_research_app, open_chat
from Tests.UI.test_research_studio_pane import prepare_studio, start_output
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.Study_Interop.local_quiz_service import LocalQuizService
from tldw_chatbook.Study_Interop.local_study_service import LocalStudyService
from tldw_chatbook.Study_Interop.quiz_scope_service import QuizScopeService
from tldw_chatbook.Study_Interop.study_scope_service import StudyScopeService
from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel


pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def test_studio_history_keeps_stale_sources_visible_and_edits_the_exact_artifact(
    tmp_path, monkeypatch
):
    app, db, provider = build_research_app(tmp_path, monkeypatch)
    notes = NotesInteropService(
        base_db_directory=tmp_path, api_client_id="studio-test", global_db_to_use=db
    )
    swap_app_aliases(
        app,
        notes_scope_service=NotesScopeService(
            local_notes_service=notes, server_service=None
        ),
    )
    provider.answer = "Original report [S1]."
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(160, 40)) as pilot:
            screen, composer = await open_chat(app, pilot)
            port, sources, report = await prepare_studio(
                screen, pilot, monkeypatch, "report"
            )
            await start_output(app, screen, pilot, report, "Versioned report")
            status = screen.query_one("#research-studio-status", Static)
            await _until(
                pilot,
                lambda: "Saved to Local Artifacts" in str(status.render()),
                "saved artifact",
            )
            first = (await port.list_outputs(screen.controller.selected_ref)).items[0]
            sources[0] = replace(sources[0], association_version=97)
            screen.query_one("#research-studio-check-route", Button).press()
            detail = screen.query_one("#research-studio-history-detail", Static)
            await _until(
                pilot,
                lambda: "Sources changed since generation" in str(detail.render()),
                "source drift",
            )
            actions = screen.query_one("#research-studio-output-actions", Button)
            notes_section = screen.query_one("#research-quick-notes-section")
            await _until(
                pilot, lambda: notes_section.can_save_note, "canonical Notes ready"
            )
            actions.press()
            await _until(pilot, lambda: app.screen is not screen, "output details")
            for key in (
                "edit",
                "discuss",
                "note",
                "reopen",
                "replace",
                "new-version",
                "delete",
                "export",
            ):
                assert app.screen.query_one(f"#research-studio-action-{key}", Button)
            assert not app.screen.query("#research-studio-action-append")
            app.screen.query_one("#research-studio-action-edit", Button).press()
            await _until(
                pilot,
                lambda: bool(app.screen.query("#research-studio-edit-body")),
                "editor",
            )
            app.screen.query_one("#research-studio-edit-body", TextArea).load_text(
                "Manually revised"
            )
            app.screen.query_one("#research-studio-edit-save", Button).press()
            await _until(
                pilot,
                lambda: "Output updated" in str(status.render()),
                "revised artifact saved",
            )
            updated = (await port.list_outputs(screen.controller.selected_ref)).items[0]
            assert (
                updated.owner_id == first.owner_id
                and updated.owner_version > first.owner_version
            )
            assert (await port.resolve_output(updated))["metadata"][
                "content"
            ] == "Manually revised"

            await _until(
                pilot,
                lambda: f"Version {updated.owner_version}" in str(detail.render()),
                "revised output history",
            )
            composer.load_text("Keep this earlier question")
            actions.press()
            await _until(
                pilot, lambda: app.screen is not screen, "revised output actions"
            )
            app.screen.query_one("#research-studio-action-discuss", Button).press()
            await _until(
                pilot,
                lambda: "Manually revised" in composer.text,
                "output in Chat draft",
            )
            assert composer.text.startswith("Keep this earlier question")
            assert len(provider.requests) == 1

            actions.press()
            await _until(
                pilot,
                lambda: app.screen is not screen,
                "output actions for Quick Notes",
            )
            app.screen.query_one("#research-studio-action-note", Button).press()
            await _until(
                pilot,
                lambda: notes_section.can_append_note and not notes_section.is_dirty,
                "saved Quick Note",
            )
            _, first_note = notes_section.capture_save_request()
            assert first_note.content == "Manually revised"
            actions.press()
            await _until(
                pilot, lambda: app.screen is not screen, "saved note append action"
            )
            app.screen.query_one("#research-studio-action-append", Button).press()
            body = notes_section.query_one("#research-quick-note-body", TextArea)
            await _until(
                pilot,
                lambda: (
                    body.text == "Manually revised\n\nManually revised"
                    and not notes_section.is_dirty
                ),
                "appended Quick Note",
            )
            _, appended = notes_section.capture_save_request()
            assert appended.note_id == first_note.note_id
            assert appended.expected_version > first_note.expected_version
            screen.query_one("#research-studio-reopen-last", Button).press()

            def exact_artifact_open() -> bool:
                catalog = getattr(app.screen, "_artifacts_controller", None)
                detail = catalog.detail if catalog else None
                return (
                    detail is not None
                    and detail.key.source == "chatbook"
                    and str(detail.key.native_id) == updated.owner_id
                )

            await _until(
                pilot, exact_artifact_open, "canonical Library artifact reopened"
            )


@pytest.mark.parametrize(
    ("kind", "selector", "answer", "channel"),
    [
        (
            "flashcards",
            "#deck-select",
            '{"items":[{"front":"Who?","back":"Rowan","citations":["S1"]}]}',
            HandoffChannel.RESEARCH_STUDY_DECK_TARGET,
        ),
        (
            "quiz",
            "#quiz-select",
            '{"items":[{"question":"Who?","options":["Rowan","Sam"],"correct_index":0,"citations":["S1"]}]}',
            HandoffChannel.RESEARCH_QUIZ_TARGET,
        ),
    ],
)
async def test_studio_reopens_only_the_exact_saved_study_owner(
    tmp_path, monkeypatch, kind, selector, answer, channel
):
    app, db, provider = build_research_app(tmp_path, monkeypatch)
    study = LocalStudyService(db)
    quizzes = LocalQuizService(db)
    study.create_deck(name="Existing unrelated deck")
    quizzes.create_quiz(name="Existing unrelated quiz")
    study_scope = StudyScopeService(local_service=study, server_service=None)
    quiz_scope = QuizScopeService(local_service=quizzes)
    if kind == "flashcards":

        async def first_deck_page(**kwargs):
            return [
                row
                for row in await StudyScopeService.list_decks(study_scope, **kwargs)
                if row["name"].startswith("Existing")
            ]

        monkeypatch.setattr(study_scope, "list_decks", first_deck_page)
    else:

        async def first_quiz_page(**kwargs):
            return [
                row
                for row in await QuizScopeService.list_quizzes(quiz_scope, **kwargs)
                if row["name"].startswith("Existing")
            ]

        monkeypatch.setattr(quiz_scope, "list_quizzes", first_quiz_page)
    swap_app_aliases(
        app,
        local_study_service=study,
        local_quiz_service=quizzes,
        study_scope_service=study_scope,
        study_quiz_scope_service=quiz_scope,
    )
    provider.answer = answer
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(160, 40)) as pilot:
            screen, _ = await open_chat(app, pilot)
            port, _, create = await prepare_studio(screen, pilot, monkeypatch, kind)
            await start_output(app, screen, pilot, create, "Canonical study output")
            reopen = screen.query_one("#research-studio-reopen-last", Button)
            await _until(pilot, lambda: not reopen.disabled, "saved Study output")
            output = (await port.list_outputs(screen.controller.selected_ref)).items[0]
            reopen.press()
            await _until(
                pilot,
                lambda: app.current_tab == "study",
                "canonical Study route",
                timeout=6,
            )
            try:
                await _until(
                    pilot,
                    lambda: (
                        bool(app.screen.query(selector))
                        and app.screen.query_one(selector, Select).value
                        == output.owner_id
                        and not app.pending_handoffs.has_pending(channel)
                    ),
                    "exact Study item selected",
                    timeout=6,
                )
            except AssertionError as exc:
                choice = app.screen.query_one(selector, Select)
                raise AssertionError(
                    f"Expected {output.owner_id}; selected {choice.value}; options {choice._options}; "
                    f"handoff {app.pending_handoffs.exact_revision_status(channel, 1)}"
                ) from exc
