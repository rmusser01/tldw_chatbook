"""Product maturity Phase 3.4 source-selected Study generation contract."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import Mock

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesScopeService,
    _build_test_app,
    _wait_for_library_snapshot,
)
from Tests.UI.test_study_dashboard import (
    DashboardStudyScopeService,
    StudyDashboardTestApp,
    _build_app_instance,
)
from tldw_chatbook.UI.Screens.study_scope_models import (
    MATERIAL_SOURCE_LIBRARY,
    MATERIAL_TITLE_LIBRARY_SOURCES,
    StudyScopeContext,
    StudySourceItem,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_3_README = Path("Docs/superpowers/qa/product-maturity/phase-3/README.md")
PHASE_3_4_EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-3/2026-05-07-phase-3-4-source-study-generation.md"
)
TASK_10 = Path("backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md")
TASK_10_4 = Path(
    "backlog/tasks/task-10.4 - Product-Maturity-Phase-3.4-Source-Selected-Study-Generation.md"
)


class RecordingSourceStudyService(DashboardStudyScopeService):
    """Dashboard service stub that records source-selected study pack jobs."""

    async def create_study_pack_job(
        self,
        *,
        mode: str | None = None,
        title: str,
        source_items: list[dict[str, object]],
        workspace_id: str | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            (
                "create_study_pack_job",
                mode,
                title,
                workspace_id,
                source_items,
            )
        )
        return {"job": {"id": 42, "status": "queued"}}

    async def get_study_pack_job_status(self, job_id: int) -> dict[str, object]:
        self.calls.append(("get_study_pack_job_status", job_id))
        statuses = getattr(self, "study_pack_job_statuses", [])
        if statuses:
            return statuses.pop(0)
        return {"job": {"id": job_id, "status": "queued"}}


async def _wait_for_source_generation_call(
    service: RecordingSourceStudyService,
    pilot,
    *,
    timeout: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(call[0] == "create_study_pack_job" for call in service.calls):
            return
        await pilot.pause(0.01)
    raise AssertionError(f"Timed out waiting for study pack generation call: {service.calls}")


async def _wait_for_service_call(
    service: RecordingSourceStudyService,
    pilot,
    call_name: str,
    *,
    timeout: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(call[0] == call_name for call in service.calls):
            return
        await pilot.pause(0.01)
    raise AssertionError(f"Timed out waiting for {call_name}: {service.calls}")


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _static_text(widget: Static) -> str:
    return str(widget.render())


@pytest.mark.asyncio
async def test_library_source_context_carries_study_pack_source_items() -> None:
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService(
        [{"title": "Research Note", "id": "note-1"}]
    )
    app.media_reading_scope_service = StaticLibraryMediaScopeService(
        [{"title": "Transcript A", "id": "media-1"}]
    )
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(
        [{"title": "Planning Chat", "id": "chat-1"}]
    )
    app.open_study_screen = Mock()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_library_snapshot(screen, pilot)
        await pilot.click("#library-open-study")
        await pilot.pause(0.1)

    scope_context = app.open_study_screen.call_args.args[0]

    assert scope_context.source_items == (
        StudySourceItem(source_type="note", source_id="note-1", label="Research Note"),
        StudySourceItem(source_type="media", source_id="media-1", label="Transcript A"),
    )
    assert scope_context.material_titles == (
        "Research Note",
        "Transcript A",
        "Planning Chat",
    )


@pytest.mark.asyncio
async def test_server_study_dashboard_launches_source_selected_study_pack_job() -> None:
    service = RecordingSourceStudyService()
    app_instance = _build_app_instance()
    app_instance.study_scope_service = service
    app_instance.current_runtime_backend = "server"
    app_instance.runtime_backend = "server"
    app_instance.notify = Mock()
    app_instance.pending_study_scope_context = StudyScopeContext(
        material_source=MATERIAL_SOURCE_LIBRARY,
        material_title=MATERIAL_TITLE_LIBRARY_SOURCES,
        material_summary="Local Library source snapshot staged for Study.",
        material_titles=("Research Note", "Transcript A"),
        source_items=(
            StudySourceItem(source_type="note", source_id="note-1", label="Research Note"),
            StudySourceItem(source_type="media", source_id="media-1", label="Transcript A"),
        ),
    )
    app = StudyDashboardTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        generate_button = app.screen.query_one("#study-generate-source-pack", Button)

        assert generate_button.disabled is False
        assert "selected Library sources" in str(generate_button.tooltip)

        await pilot.click("#study-generate-source-pack")
        await _wait_for_source_generation_call(service, pilot)

        status = app.screen.query_one("#study-source-generation-status", Static)

        assert "queued" in _static_text(status).lower()
        assert "42" in _static_text(status)

    assert (
        "create_study_pack_job",
        "server",
        "Local Library Sources",
        None,
        [
            {"source_type": "note", "source_id": "note-1", "label": "Research Note", "locator": {}},
            {"source_type": "media", "source_id": "media-1", "label": "Transcript A", "locator": {}},
        ],
    ) in service.calls
    app_instance.notify.assert_called_with("Study pack generation queued.", severity="information")


@pytest.mark.asyncio
async def test_server_study_dashboard_observes_completed_source_pack_for_reuse() -> None:
    service = RecordingSourceStudyService()
    service.study_pack_job_statuses = [
        {
            "job": {"id": 42, "status": "completed"},
            "study_pack": {
                "id": 9,
                "title": "Research Note Study Pack",
                "deck_id": 7,
                "status": "active",
                "deleted": False,
                "client_id": "server-client",
                "version": 1,
            },
        }
    ]
    app_instance = _build_app_instance()
    app_instance.study_scope_service = service
    app_instance.current_runtime_backend = "server"
    app_instance.runtime_backend = "server"
    app_instance.notify = Mock()
    app_instance.pending_study_scope_context = StudyScopeContext(
        material_source=MATERIAL_SOURCE_LIBRARY,
        material_title=MATERIAL_TITLE_LIBRARY_SOURCES,
        material_titles=("Research Note",),
        source_items=(StudySourceItem(source_type="note", source_id="note-1", label="Research Note"),),
    )
    app = StudyDashboardTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        await pilot.click("#study-generate-source-pack")
        await _wait_for_service_call(service, pilot, "get_study_pack_job_status")

        status = app.screen.query_one("#study-source-generation-status", Static)
        recent_decks = app.screen.query_one("#study-recent-decks", Static)
        resume_button = app.screen.query_one("#study-resume-last", Button)

        assert "ready" in _static_text(status).lower()
        assert "Research Note Study Pack" in _static_text(status)
        assert "deck 7" in _static_text(status)
        assert "Research Note Study Pack" in _static_text(recent_decks)
        assert resume_button.disabled is False
        assert "flashcards" in str(resume_button.label).lower()

        await pilot.click("#study-resume-last")
        await pilot.pause(0.2)

        assert app.screen.current_section == "flashcards"

    app_instance.notify.assert_any_call("Study pack ready.", severity="information")


@pytest.mark.asyncio
async def test_server_study_dashboard_keeps_failed_source_pack_generation_recoverable() -> None:
    service = RecordingSourceStudyService()
    service.study_pack_job_statuses = [
        {
            "job": {"id": 42, "status": "failed"},
            "error": "Embedding service unavailable",
        }
    ]
    app_instance = _build_app_instance()
    app_instance.study_scope_service = service
    app_instance.current_runtime_backend = "server"
    app_instance.runtime_backend = "server"
    app_instance.notify = Mock()
    app_instance.pending_study_scope_context = StudyScopeContext(
        material_source=MATERIAL_SOURCE_LIBRARY,
        material_title=MATERIAL_TITLE_LIBRARY_SOURCES,
        material_titles=("Research Note",),
        source_items=(StudySourceItem(source_type="note", source_id="note-1", label="Research Note"),),
    )
    app = StudyDashboardTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        await pilot.click("#study-generate-source-pack")
        await _wait_for_service_call(service, pilot, "get_study_pack_job_status")

        status = app.screen.query_one("#study-source-generation-status", Static)
        generate_button = app.screen.query_one("#study-generate-source-pack", Button)

        assert "failed" in _static_text(status).lower()
        assert "Embedding service unavailable" in _static_text(status)
        assert generate_button.disabled is False
        assert "Retry source study-pack generation" in str(generate_button.tooltip)

    app_instance.notify.assert_any_call(
        "Study pack generation failed. Embedding service unavailable",
        severity="error",
    )


@pytest.mark.asyncio
async def test_local_study_dashboard_explains_source_generation_server_requirement() -> None:
    service = RecordingSourceStudyService()
    app_instance = _build_app_instance()
    app_instance.study_scope_service = service
    app_instance.pending_study_scope_context = StudyScopeContext(
        material_source=MATERIAL_SOURCE_LIBRARY,
        material_title=MATERIAL_TITLE_LIBRARY_SOURCES,
        material_summary="Local Library source snapshot staged for Study.",
        material_titles=("Research Note",),
        source_items=(StudySourceItem(source_type="note", source_id="note-1", label="Research Note"),),
    )
    app = StudyDashboardTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        generate_button = app.screen.query_one("#study-generate-source-pack", Button)
        status = app.screen.query_one("#study-source-generation-status", Static)

        assert generate_button.disabled is True
        assert "server mode" in str(generate_button.tooltip).lower()
        assert "server mode" in _static_text(status).lower()

    assert not any(call[0] == "create_study_pack_job" for call in service.calls)


def test_phase_3_4_source_study_generation_evidence_is_tracked() -> None:
    tracker = _text(TRACKER)
    readme = _text(PHASE_3_README)
    evidence = _text(PHASE_3_4_EVIDENCE)
    parent_task = _text(TASK_10)
    task = _text(TASK_10_4)

    assert "Phase 3.4" in tracker
    assert "TASK-10.4" in tracker
    assert PHASE_3_4_EVIDENCE.name in tracker
    assert PHASE_3_4_EVIDENCE.name in readme
    assert "Source-Selected Study Generation" in evidence
    assert "Library -> Study Dashboard -> Generate Source Study Pack" in evidence
    assert "No P0/P1 defects found" in evidence
    assert "Tests/UI/test_product_maturity_phase3_source_study_generation.py" in evidence
    assert "status: In Progress" in parent_task
    assert "TASK-10.4" in parent_task
    assert "Product Maturity Phase 3.4: Source-Selected Study Generation" in task
    assert "status: Done" in task
    for ac_number in range(1, 5):
        assert f"- [x] #{ac_number}" in task
